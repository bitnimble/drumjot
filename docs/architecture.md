# Architecture

Detailed architecture reference. The top-level [AGENTS.md](../AGENTS.md)
(symlinked as `CLAUDE.md`) is the index; pull this in when you need the
full picture of how the pieces fit.

## What Drumjot is

A browser-based drum notation tool with three deeply integrated layers,
sharing the DSL as their lingua franca:

1. **A DSL** ([SPEC.md](../SPEC.md)) for representing drum patterns
   compactly: notes are single letters, `.` is a rest, `(...)_N` groups
   with weighted durations, `[Name=(...)]` patterns, `+` for
   simultaneity, `||` for parallel voices, inline `{{...}}` for metadata
   changes (bpm, time sig, instrument mapping, etc.). See
   [dsl-reference.md](dsl-reference.md) for the quick reference.
2. **A React/MobX renderer** that lays a parsed Jot out as
   per-instrument-lane staves with bar lines, pattern brackets, accent
   modifiers, etc. Stack: Vite + React 18 + MobX (`mobx-react-lite`) +
   CSS modules. Bun is the package manager and test runner.
3. **A transcriber service** (separate Python/FastAPI backend,
   Docker-deployed) that turns arbitrary audio into a predicted-onsets
   MIDI file. See [transcriber-pipeline.md](transcriber-pipeline.md).

The in-memory model of a parsed Jot is `Jot` from [src/dsl.ts](../src/dsl.ts);
the rendered (positioned) form is `ResolvedJot` from [src/jot.ts](../src/jot.ts).
There are also bidirectional converters for **MIDI** (`src/midi/`) and
**Paradiddle RLRR** (`src/rlrr/`, a JSON song-chart format from
[ParadiddleUtilities](https://github.com/emretanirgan/ParadiddleUtilities)).

## Repo layout

```
drumjot/
├── SPEC.md                         The DSL grammar. Source of truth.
├── AGENTS.md / CLAUDE.md           Index (CLAUDE.md is a symlink to AGENTS.md).
├── package.json                    Bun-driven; type: module. `browserslist` field
│                                   is the source of truth for browser targets.
├── tsconfig.json                   Path alias `src/*` -> `src/*`.
├── vite.config.ts                  /api proxy -> http://localhost:8001 (transcriber).
├── index.html                      Vite entry; mounts #app.
├── src/                            Frontend (TS, all client + library code).
│   ├── design_tokens.css           Global :root design tokens. See design-system.md.
│   ├── typography.module.css       Use-case typography classes (composed via `composes:`).
│   ├── dsl.ts                      Types: Note, Rest, Group, Simultaneity,
│   │                               PatternRef, Bar, Voice, Pattern, Jot,
│   │                               Metadata, Instrument, Modifier, Sticking, Limb.
│   ├── jot.ts                      RenderedJot + layout pipeline; per-bar pixel
│   │                               positions. Holds LANE_COLORS data palette.
│   ├── jot_view.tsx                React renderer + JotViewStore (MobX).
│   ├── jot_view.module.css         Shared form chrome CSS modules.
│   ├── jot_view/                   Per-component chrome split out of jot_view.tsx.
│   │   ├── toolbar.tsx             Header strip + dropdowns + DebugPanel.
│   │   ├── playback.tsx            Bottom transport bar + master volume + playhead.
│   │   ├── mixer.tsx               Unified mixer (audio tracks + drum pitch rows).
│   │   ├── score.tsx               Timeline header, bars, notes, brackets,
│   │   │                           note-label popovers, filtered-onset ghosts.
│   │   ├── store.ts                JotViewStore (MobX) + TrackKey + constants.
│   │   ├── contexts.ts             React contexts (NoteProvenanceContext, …).
│   │   └── components/             Shared React UI primitives + design-system
│   │                               module CSS (button, icon_button, modal, form,
│   │                               spinner, dropdown, tabs, number_stepper, …).
│   ├── fakes.ts                    EXAMPLE_JOTS (rockJot, tripletJot).
│   ├── geom.ts                     Tiny Point/Box helpers.
│   ├── selection.ts                Marquee + pattern selection store.
│   ├── transcriber.ts              HTTP client for the transcriber service.
│   ├── playback/                   Browser drum playback via smplr.
│   │   ├── player.ts               JotPlayer singleton (MobX-observable).
│   │   ├── events.ts               RenderedJot -> PlaybackEvent[].
│   │   ├── timeline.ts             buildTimeline + timeToX (reads LIVE bar.x/width).
│   │   ├── drums.ts                MIDI note -> drum role -> smplr kit group (TR-808).
│   │   ├── stems.ts                StemPlaybackController (audio tracks in sync).
│   │   └── index.ts                Public exports.
│   ├── index.tsx                   Vite entry; exposes window.drumjot / window.jotPlayer.
│   ├── parser/                     Recursive-descent DSL parser.
│   │   ├── parser.ts               parseJot, suffixes, simultaneity, patterns,
│   │   │                           bar/voice slicing, per-bar metadata snapshots.
│   │   ├── preprocess.ts           Macro substitution ([$name=...] / [$name]).
│   │   ├── metadata.ts             {{...}} and {...} block parsing.
│   │   ├── cursor.ts / errors.ts   Text cursor + ParseError with line/col.
│   │   └── __tests__/              parser.test.ts, preprocess.test.ts.
│   ├── midi/                       MIDI <-> Jot.
│   │   ├── from_midi.ts            fromMidi(bytes, opts). Quantizes to 16th grid;
│   │   │                           preserves raw note/velocity on note.metadata.midi.
│   │   ├── to_midi.ts              toMidi(jot, opts). Channel 10, 480 PPQN.
│   │   ├── gm.ts                   GM percussion mapping (note -> letter+mods).
│   │   └── __tests__/              synthetic tests + fixture harness.
│   ├── rlrr/                       RLRR <-> {Jot, MIDI}.
│   └── */tests/*.e2e.ts            Playwright e2e specs, co-located per feature
│                                   (separate runner; see AGENTS.md §Build).
├── tests/fixtures/                 Shared e2e fixtures (tone.wav, song.jot, …).
├── playwright.config.ts            Headless Chromium; webServer = bun run dev.
├── bunfig.toml                     Scopes `bun test` to src/ (runner split).
├── .stylelintrc.json               Design-token lint config. See design-system.md.
├── .mcp.json                       Project-scoped @playwright/mcp server.
├── docs/                           This folder (detailed docs).
├── research/                       Research notes (HIHAT, MODELS, alignment, lyrics).
├── transcriber/                    Python backend (FastAPI + Docker). See
│                                   transcriber-pipeline.md for the full pipeline tree.
└── training/                       Learned drum-onset model (frozen MERT encoder +
                                    per-lane heads). Pure Python, offline, not in the
                                    transcriber runtime. See training/README.md.
```

## Architectural decisions

### The DSL is the lingua franca

Every conversion (audio, MIDI, RLRR) targets the DSL. It was designed
before the transcriber existed, so anything we want from audio
transcription has to be expressible in it. The DSL already covered
triplets (`(...)_N`), tempo changes (inline `{{bpm}}`), time-signature
changes (inline `{{time}}`), velocity (`:a` / `:g` / `vol`), and pattern
reuse (`[Name=(...)]`), so the transcriber chooses the right DSL
constructs, it doesn't invent new ones.

### The transcriber is pure Python (no bun bridge)

The DSL parser and all DSL/MIDI/RLRR manipulation live in TypeScript on
the **frontend only** (`src/parser/`, `src/midi/`, `src/rlrr/`). The
transcriber does **not** parse, interpret, or emit Jots: it produces
MIDI directly (`app/pipeline/onsets_midi.py`), and `src/midi/from_midi.ts`
converts that MIDI into a Jot in the browser. There is no `bun` and no
`src/` in the transcriber runtime.

**If Python ever needs to read MIDI again** (e.g. the MIDI↔audio
alignment scorer, [research/midi-audio-alignment-score.md](../research/midi-audio-alignment-score.md)):
port the canonical `src/midi/gm.ts` GM→pitch mapping to Python with a
drift-guard fixture test, rather than resurrecting the bridge. `gm.ts` /
`from_midi.ts` stay the source of truth for MIDI semantics.

### Beat-relative coordinates, not fixed grids

The transcriber operates in `(bar, beat_in_bar)` space throughout
(backed by madmom). Chosen over a 1/16 grid because triplets become a
property of intra-beat fractions (0.000 / 0.333 / 0.667), tempo changes
work naturally (each beat has its own absolute time anchor), time-sig
changes are detected from downbeat gaps, and tolerances stay musically
sensible at any tempo. See [transcriber-pipeline.md](transcriber-pipeline.md).

### LLM-in-the-loop, deterministic everywhere else

The LLM's only job is the **symbolic filtering** layer, applied **one
instrument at a time**: filter each instrument's candidate onsets,
rejecting artifacts. Separation, onset detection, beat tracking, MIDI
render, and the whole React UI / MIDI / RLRR export are all
deterministic. This keeps LLM cost bounded (~$0.05–0.30/song) and makes
failure modes inspectable.

### What doesn't exist (deliberately)

- **No audio-native LLM step.** The pipeline never feeds raw audio to an
  LLM; only structured candidate lists. (Explicitly out of scope; first
  lever to revisit if accuracy plateaus.)
- **No cross-instrument musical context in the filter.** Each
  per-instrument call sees only its own onsets + the shared beat frame.
  This is the main accuracy trade-off of the per-instrument split,
  accepted for reliability/parallelism/isolatable failures. Designed but
  unbuilt mitigation: feed each call a read-only summary of other
  instruments' onset positions.
- **No cross-instrument pattern factoring or `@stick` generation.** The
  MIDI→Jot conversion is mechanical; sticking is cross-hand and can't
  come from monophonic per-instrument data.

### Frontend performance model

**No DOM layout reads in hot paths.** The score uses a virtualised
scroll model: `.jotContainer` is `overflow: hidden`, an inner
`.scrollViewport` moves via `transform: translate(-scrollX, -scrollY)`,
and offsets live on `JotViewStore` as MobX observables (`scrollX`,
`scrollY`). Viewport/content extents are cached on the store, fed by a
`ResizeObserver` in JotView.

In any render, effect, MobX reaction, or per-frame/scroll/zoom path, **do
not read DOM layout metrics** (`scrollLeft`, `clientWidth`,
`getBoundingClientRect`, `getComputedStyle` for layout, etc.), a layout
read forces a synchronous reflow; per-frame that's jank. Read from the
store. If a needed dimension isn't on the store, add an observable and
feed it from the existing JotView `ResizeObserver`.

**Legitimate exceptions** (stay as DOM reads): synchronous user-input
handlers that need a fresh rect at gesture time (click→seek, marquee
drag start, dropdown anchoring) and `ResizeObserver` callbacks that feed
the store.

**Frame budget is 120 fps / 8.3 ms** (the user runs a 165 Hz monitor),
not 60 fps / 16.6 ms. Apply to every per-frame budget judgement and
frame source comments as 120 Hz / 8.3 ms.

### Browser target: evergreen, last 2 years

Source of truth is `package.json`'s `browserslist` (`last 2 years`, `not
dead`, `not op_mini all`). Use all modern web platform features without
polyfills or fallbacks (`OffscreenCanvas`, `ResizeObserver`,
`structuredClone`, top-level `await`, `Intl.Segmenter`, `:has()`,
container queries, `color-mix()`, `content-visibility`, subgrid, etc.).
Don't add feature-detection for features universally supported ≥18
months. `browserslist` is not yet wired into Vite's `build.target`
(defaults to `'modules'` ≈ Chrome 87+); integrate via
`browserslist-to-esbuild` if output-side enforcement ever matters.

### Score stacking / overlay clipping

The score's stacking model is subtle (transform root stacking context,
equal-z sibling rows, `content-visibility: auto` clipping). Selection
popovers are portaled to `document.body` via `PopoverPortal` to escape
clipping. **Before adding any overlay/popover/badge that escapes its bar
or row, read [score-stacking.md](score-stacking.md)** (full z-index
ladder + portal + clipping contract). Regression test:
`e2e/popover-visibility.spec.ts`.
