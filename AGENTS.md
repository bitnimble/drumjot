# Agent handover

This document is the single entry point for any AI agent (or human)
inheriting this repo. It captures the project's purpose, current state,
architectural decisions, the conversational history that shaped the
code, plans that exist but haven't been executed yet, and a list of
gotchas worth knowing before making changes. Read it top to bottom
once; refer back as needed.

Last updated by the assistant: May 2026, after the legacy-pathway
purge (timeline #26): the DSL-output transcribe endpoint and the
librosa onset backend are gone. The backend now produces only MIDI
(via the filter LLM that rejects artifact onsets per instrument),
and `src/midi/from_midi.ts` on the frontend converts that to a Jot.
Useful techniques from the deleted DSL pipeline (per-instrument
prompting, deterministic recompose, F1-gated multi-level refine,
critic triage, best-of-K, pattern-aware suppression) are captured in
`transcriber/docs/ai-midi-to-jot-notes.md` for any future AI-assisted
MIDI → Jot work. Prior session was the frontend design-system pass
(timeline #25): global color / radius / shadow tokens in
`src/design_tokens.css`, use-case typography classes in
`src/typography.module.css` (composed via CSS-modules `composes:`),
shared mixer-button primitives under `src/jot_view/components/`, and
a stylelint design-token lint (configured in `.stylelintrc.json`,
chained into `bun run build`). **Read §5.8 before touching any module CSS**; hardcoded colors will fail the lint.

**Workflow override (superpowers plugin):** the `superpowers:writing-plans`
skill is **off by default for this project**. Once `superpowers:brainstorming`
converges on requirements, go straight to implementation; no separate spec /
plan doc. Reach for `writing-plans` only when the work genuinely needs review
checkpoints (multi-day, multi-agent, or risky cross-cutting changes); ask the
user first if unsure.

---

## 1. What Drumjot is

A browser-based drum notation tool with three deeply integrated layers:

1. **A DSL** ([SPEC.md](SPEC.md)) for representing drum patterns
   compactly: notes are single letters, `.` is a rest, `(...)_N` groups
   with weighted durations, `[Name=(...)]` patterns, `+` for
   simultaneity, `||` for parallel voices, inline `{{...}}` for
   metadata changes (bpm, time sig, instrument mapping, etc.).
2. **A React/MobX renderer** that takes a parsed Jot and lays it out as
   per-instrument-lane staves with bar lines, pattern brackets, accent
   modifiers, and so on. Frontend stack: Vite + React 18 + MobX (via
   `mobx-react-lite`) + CSS modules. Bun is the package manager and
   test runner; **do not use npm**.
3. **A transcriber service** (separate Python backend, Docker-deployed)
   that takes arbitrary audio and produces a predicted-onsets MIDI
   file via: BS-Roformer SW separation -> MDX23C DrumSep -> madmom
   beat tracking -> ADTOF Frame_RNN per-stem onset detection -> Claude
   filter LLM that rejects artifact onsets per instrument. The kept
   onsets render straight to MIDI with their original un-quantized
   times; `src/midi/from_midi.ts` on the frontend converts that MIDI
   into a Jot. (The earlier DSL-output / refinement pathway was
   deleted in May 2026 — see
   `transcriber/docs/ai-midi-to-jot-notes.md` for the captured
   techniques.)

The DSL is the lingua franca that all three layers share. The
in-memory model of a parsed Jot is `Jot` from [src/dsl.ts](src/dsl.ts);
the rendered (positioned) form is `ResolvedJot` from
[src/jot.ts](src/jot.ts).

There are also bidirectional converters for **MIDI** and **Paradiddle
RLRR** (the latter is a JSON song-chart format from
[ParadiddleUtilities](https://github.com/emretanirgan/ParadiddleUtilities)).

---

## 2. Repo layout

```
drumjot/
├── SPEC.md                         The DSL grammar. Source of truth.
├── AGENTS.md                       This file.
├── package.json                    Bun-driven; type: module.
├── tsconfig.json                   Path alias `src/*` -> `src/*`.
├── vite.config.ts                  /api proxy -> http://localhost:8001 (transcriber).
├── index.html                      Vite entry; mounts #app.
├── src/                            Frontend (TS, all client + library code).
│   ├── design_tokens.css           Global :root design tokens (colors,
│   │                               radii, shadows). Loaded once from
│   │                               index.tsx. See §5.8.
│   ├── typography.module.css       Use-case typography classes (heading,
│   │                               body, label, …) pulled in via
│   │                               `composes:` by other modules. See §5.8.
│   ├── dsl.ts                      Types: Note, Rest, Group, Simultaneity,
│   │                               PatternRef, Bar, Voice, Pattern, Jot,
│   │                               Metadata, Instrument, Modifier, Sticking, Limb.
│   ├── jot.ts                      RenderedJot + layout pipeline; computes
│   │                               per-bar pixel positions for the renderer.
│   ├── jot_view.tsx                React renderer. Contains JotViewStore (MobX),
│   │                               Toolbar (example picker, transcribe button,
│   │                               refine checkbox), staff/bar/note components,
│   │                               pattern brackets, status pill.
│   ├── jot_view.module.css         CSS modules for the above (shared form
│   │                               chrome — toolbarCheckbox, samplesSelect,
│   │                               offsetInput, statusPill*, global slider).
│   ├── jot_view/                   Per-component chrome split out of jot_view.tsx.
│   │   ├── toolbar.tsx + .module.css   Header strip + dropdowns + DebugPanel.
│   │   ├── playback.tsx + .module.css  Bottom transport bar + master volume
│   │   │                               + playhead line/label.
│   │   ├── mixer.tsx + .module.css     Unified mixer (audio tracks +
│   │   │                               drum-instrument pitch rows).
│   │   ├── score.tsx + .module.css     Timeline header, bars, notes, brackets,
│   │   │                               note label popovers, filtered-onset ghosts.
│   │   ├── store.ts                    JotViewStore (MobX) + TrackKey + constants.
│   │   ├── contexts.ts                 React contexts (NoteProvenanceContext, …).
│   │   └── components/                 Cross-module shared React UI primitives
│   │                                   (icon_button.tsx + .module.css today; the
│   │                                   home for future shared widgets).
│   ├── fakes.ts                    EXAMPLE_JOTS (rockJot, tripletJot).
│   ├── geom.ts                     Tiny Point/Box helpers.
│   ├── selection.ts                Marquee + pattern selection store.
│   ├── recompose.ts                Deterministic merge of per-instrument
│   │                               monophonic fragments into one Jot
│   │                               (hands `+`, polyrhythm `(A)+(B)`,
│   │                               kick = 2nd `||` voice). Used by the
│   │                               transcriber via tools/recompose_jot.ts.
│   ├── recompose.test.ts           bun-test coverage for the above.
│   ├── transcriber.ts              HTTP client for the transcriber service.
│   ├── playback/                   Browser drum playback via smplr.
│   │   ├── player.ts               JotPlayer singleton (MobX-observable
│   │   │                           state/currentTime/timeline/playbackSpeed;
│   │   │                           live setFilter for mute/solo;
│   │   │                           live setPlaybackSpeed for slow practice).
│   │   ├── events.ts               RenderedJot -> PlaybackEvent[] (carries
│   │   │                           DSL pitch letter for mute/solo filtering).
│   │   ├── timeline.ts             buildTimeline + timeToX; reads LIVE
│   │   │                           bar.x/width so zoom changes during
│   │   │                           playback keep the playhead in sync.
│   │   ├── drums.ts                MIDI note -> drum role -> smplr kit-group
│   │   │                           resolution. Hardcoded for TR-808 today.
│   │   └── index.ts                Public exports.
│   ├── index.tsx                   Vite entry; constructs Drumjot, mounts the
│   │                               View, exposes window.Drumjot for the console.
│   ├── parser/                     Recursive-descent DSL parser.
│   │   ├── parser.ts               parseJot, parseElement, suffixes (:, @, _N,
│   │   │                           *N, ~, {meta}), simultaneity, patterns,
│   │   │                           bar/voice slicing, per-bar metadata snapshots.
│   │   ├── preprocess.ts           Macro substitution ([$name=...] / [$name]).
│   │   ├── metadata.ts             {{...}} and {...} block parsing.
│   │   ├── cursor.ts               Text cursor utility.
│   │   ├── errors.ts               ParseError with line/col.
│   │   ├── index.ts                Public: parse, ParseError, preprocessMacros.
│   │   └── __tests__/              parser.test.ts (47 tests),
│   │                               preprocess.test.ts (6 tests).
│   ├── midi/                       MIDI <-> Jot.
│   │   ├── from_midi.ts            fromMidi(bytes, opts). Quantizes to 16th
│   │   │                           grid; preserves raw MIDI note/velocity on
│   │   │                           note.metadata.midi for lossless round-trip.
│   │   ├── to_midi.ts              toMidi(jot, opts). Channel 10, 480 PPQN.
│   │   ├── gm.ts                   GM percussion mapping (note -> letter+mods).
│   │   ├── index.ts                Public API.
│   │   └── __tests__/              7 synthetic tests + fixture harness
│   │                               (drop .mid files in __tests__/fixtures/).
│   └── rlrr/                       RLRR <-> {Jot, MIDI}.
│       ├── schema.ts               RlrrFile types + DEFAULT_INSTRUMENTS kit
│       │                           (copied verbatim from defaultset.rlrr).
│       ├── drums.ts                BP_<Class>_C <-> pitch+mods+midi mapping.
│       ├── rlrr_to_jot.ts          RLRR -> Jot.
│       ├── jot_to_rlrr.ts          Jot -> RLRR.
│       ├── midi_to_rlrr.ts         Port of midiconvert.py (one-way upstream).
│       ├── rlrr_to_midi.ts         Inverse of the above.
│       └── __tests__/              ~10 synthetic tests + fixture harness.
├── e2e/                            Playwright e2e (separate runner; see §3.1).
│   ├── smoke.spec.ts               Boot, load .jot, mocked transcribe.
│   ├── music-stems.spec.ts         Stem load/decode/waveform/mute/solo/play.
│   ├── fixtures/                   loop.jot, tone.wav (+ gen-tone.mjs).
│   └── helpers/transcriber-mock.ts page.route stub for /api/transcribe.
├── playwright.config.ts            Headless Chromium; webServer = bun run dev.
├── bunfig.toml                     Scopes `bun test` to src/ (runner split).
├── .stylelintrc.json               Design-token lint config: bans hex / rgb /
│                                   rgba / hsl / hsla in any `src/**/*.css`
│                                   outside `src/design_tokens.css`. Run via
│                                   `bun run lint:design`; chained into
│                                   `bun run build`. See §5.8.
├── .mcp.json                       Project-scoped @playwright/mcp server.
└── transcriber/                    Python backend (FastAPI + Docker).
    ├── README.md                   Service-level docs (formats, API, perf).
    ├── Dockerfile                  CUDA 12.4 base; installs bun + madmom.
    ├── docker-compose.yml          Build context = REPO ROOT (so src/ is
    │                               available to the bun bridge).
    ├── pyproject.toml              Deps: fastapi, audio-separator[gpu],
    │                               adtof_pytorch, madmom (from git),
    │                               mir_eval, anthropic.
    ├── .env.example                ANTHROPIC_API_KEY, LLM_MODEL,
    │                               INSTRUMENT_CONCURRENCY, DEBUG_DIR.
    ├── docs/ai-midi-to-jot-notes.md   Captured techniques from the now-
    │                               deleted DSL pathway; reference for any
    │                               future AI-assisted MIDI → Jot pass.
    ├── prompts/                    Markdown prompt templates with placeholders.
    │   ├── filter_onsets.md         Filter-stage artifact rejection prompt.
    │   ├── split_cymbals.md         Ride/crash classification.
    │   └── split_hihat.md           Closed/open hi-hat classification.
    └── app/
        ├── main.py                 FastAPI: GET /health, POST /transcribe,
        │                           POST /transcribe/resume.
        ├── config.py               Pydantic-settings (env-driven).
        ├── debug.py                DebugSink + serializers (beats_dump,
        │                           onsets_dump). Request-scoped ContextVar
        │                           so deep callees can dump prompts
        │                           without threading the sink.
        ├── models.py               Request/response schemas:
        │                           TranscribeResponse, OnsetCandidate (with
        │                           bar/beat_in_bar), BarSummary,
        │                           TranscriptionSummary.
        └── pipeline/
            ├── runner.py           Stage enum + PipelineContext +
            │                       run_pipeline(). Both endpoints dispatch
            │                       through here; StageError carries the
            │                       failing stage so HTTP layer maps to
            │                       500 (local) / 502 (transcribe = LLM).
            ├── resume.py           hydrate_context_from_resume(folder,
            │                       start_stage): loads on-disk artifacts
            │                       for stages strictly before start_stage,
            │                       raises FileNotFoundError naming the
            │                       missing piece for HTTP 400.
            ├── separate.py         Two-stage separator with INDEPENDENT
            │                       run_stems_all() / run_stems_per() so
            │                       resume can re-run either alone.
            │                       BS-Roformer SW for `stems_all`, Jarredou
            │                       MDX23C 5-stem DrumSep for `stems_per`.
            │                       Owns STEM_NAME_TO_PITCH +
            │                       PITCH_DISPLAY_NAMES.
            ├── beats.py            madmom RNN+DBN downbeat tracker (default)
            │                       OR Beat Transformer activations into the
            │                       shared DBN (selectable via `beat_tracker`
            │                       setting). BeatStructure with BarInfo
            │                       (tempo, time sig, feel); intra-beat
            │                       fraction analysis for feel detection.
            │                       `_summarize` excludes bar 0 when ≥2
            │                       bars present (anacrusis handling).
            ├── adtof_onsets.py     ADTOF Frame_RNN onset detection per stem
            │                       (the sole onset detector). Noisy lanes
            │                       (hihat / merged cymbals) read off the
            │                       in-distribution drum stem.
            ├── cymbal_split.py     Splits the merged `c` cymbals lane into
            │                       ride (`d`) / crash (`c`) via features +
            │                       a small LLM classification pass.
            ├── hihat_split.py     Splits the hi-hat lane into closed (`h`)
            │                       / open (`H`) via features + LLM.
            ├── filter_llm.py       Per-instrument Claude filter that
            │                       rejects artifact onsets. Tool-use
            │                       channel for structured output.
            ├── onsets_midi.py      Render kept onsets to `prediction.mid`
            │                       with per-bar tempo + time-sig map.
            ├── note_provenance.py  Per-note debug sidecar listing every
            │                       detected onset (kept and rejected) so
            │                       the UI can annotate notes + render
            │                       rejected onsets as ghost overlays.
            └── llm_util.py         Refusal/content-filter retry helper +
                                    code-fence strip.
```

Debug folder layout (when DEBUG_DIR is set or debug=true on a request):

```
/debug/<timestamp>_<id>_<slug>/
├── input.<ext>           # raw upload
├── stems_all/            # stage `stems_all` output (Demucs)
│   ├── drum_stem.wav     # isolated drums (consumed by stems_per)
│   ├── no_drums.wav      # bass+other+vocals summed; ready-to-play
│   │                     # drumless mix (e.g. for backing-track practice)
│   └── bass/other/vocals # individual sibling stems (audit aid)
├── stems_per/            # stage `stems_per` output (MDX23C)
│   ├── k.wav, s.wav, h.wav, d.wav, c.wav, t.wav
│   └── residual.wav      # drum_stem − sum(per-instrument stems).
│                         # Carries auxiliary percussion the 5-class
│                         # model has no lane for (cowbell, tambourine,
│                         # shaker, claps, woodblock) plus the model's
│                         # own reconstruction error. Diagnostic-only —
│                         # no downstream stage consumes it.
├── beats.json            # stage `beats` output (BeatStructure dump)
├── onsets.json           # stage `onsets` output (per-pitch candidates)
├── onsets_only.mid       # auto-emitted alongside onsets.json; one MIDI hit
│                         # per detected onset (channel 10, per-pitch
│                         # percentile-normalised velocity). Drop into a DAW
│                         # to hear *exactly* what the detector heard, with
│                         # no LLM filtering or quantization in the way.
├── prediction.mid        # stage `transcribe` output: kept-onset MIDI (the score)
├── note_provenance.json  # per-note debug sidecar (kept + rejected onsets)
├── filter/kept_onsets.json  # per-pitch list of onsets the filter kept
├── llm/NN_<purpose>.txt              # full hydrated prompt for one LLM call
├── llm/NN_<purpose>.response.json    # parsed Anthropic response paired with
│                                     # the prompt above (same NN): content
│                                     # blocks incl. tool_use.input,
│                                     # stop_reason, usage. Essential for
│                                     # diagnosing forced-tool truncation:
│                                     # `stop_reason=="max_tokens"` plus an
│                                     # empty `input` is the signature.
└── request.json          # filename + options + timings summary
```

Outputs folder layout (always populated, regardless of the `debug` flag).
Each FLAC is written the instant its producing stage finishes — never
deferred to the end of the request — so the deliverables are downloadable
while the slow beats/onsets/transcribe/refine stages are still running.
The request directory is created lazily on the first successful write, so
an `/outputs/<...>/` folder that exists at all is guaranteed non-empty
(an empty one would have meant "died before producing any deliverable").

Crucially, the in-stage writes only fire when those stages actually run.
A `/transcribe/resume` from `beats` (or later) skips `stems_all`/
`stems_per` entirely, so `app.outputs.materialize_pending` runs once
after the pipeline (both endpoints, before the temp work_dir is torn
down) and backfills every missing deliverable from the hydrated context
or — for `no_drums` and `residual`, which no stage carries on the
context — by scavenging `<resume_dir>/stems_all/no_drums.<ext>` and
`<resume_dir>/stems_per/residual.<ext>` respectively. No recomputation;
already-present FLACs are left as the stage wrote them. `final.jot` is
always (re)written so it tracks the latest refine result.

```
/outputs/<timestamp>_<id>_<slug>/   # same slug as the debug folder
├── drum_stem.flac        # `stems_all` (or resume backfill): isolated
│                         # drum mix (lossless re-encode; FLAC keeps
│                         # size modest). Surfaced as
│                         # TranscribeResponse.drum_stem_url.
├── no_drums.flac         # `stems_all` (or resume backfill, scavenged
│                         # from the debug folder): bass+other+vocals
│                         # summed; the "music minus drums" backing track
│                         # surfaced as TranscribeResponse.no_drums_url.
├── stem_<k|s|h|d|c|t>.flac  # `stems_per` (or resume backfill): one FLAC
│                         # per recovered per-instrument stem (only the
│                         # pitches the drum-piece separator found).
├── residual.flac         # `stems_per` (or resume backfill, scavenged
│                         # from `<resume_dir>/stems_per/residual.<ext>`):
│                         # drum_stem − sum(per-instrument stems).
│                         # Diagnostic-only — captures aux percussion
│                         # the 5-class model has no lane for + the
│                         # separator's own reconstruction residue.
│                         # Not surfaced as a URL on the response; only
│                         # included in the debug bundle's MP3 set.
├── prediction.mid        # `transcribe` output: the kept-onset MIDI
│                         # (the score). Surfaced as
│                         # TranscribeResponse.prediction_midi_url.
└── debug.zip             # full debug bundle (prediction.mid +
                          # note_provenance.json + MP3-encoded audio
                          # tracks + JSON manifest with stage timings +
                          # captured logs). Loaded back into the UI
                          # via "Load debug bundle".
```

The FastAPI app serves this folder via `StaticFiles` at `/outputs/...`,
so `TranscribeResponse.drum_stem_url` and `no_drums_url` (paths with
leading `/`) compose against the configured `TRANSCRIBER_BASE` (`/api`
in dev → routes through the Vite proxy; absolute URL in prod). See
`src/transcriber.ts::stemUrl` for the canonical client-side helper.

---

## 3. Build, test, run

**After any code change, run one of `scripts/check*` instead of
invoking ruff / pytest / tsc / stylelint / bun test directly.** The
scripts get venv activation, working directory, and autofix flags right
so agents don't have to re-derive them every turn:

| Script | What it does |
|---|---|
| `scripts/check` | Both sides — runs `check-py` then `check-ts`. The default after any cross-cutting change. |
| `scripts/check-py [pytest args]` | Python: `ruff check --fix` + `pytest`. Args pass through to pytest, so `scripts/check-py tests/test_hihat_split.py::test_label_for_priority` works. Requires `transcriber/.venv`. |
| `scripts/test-py [pytest args]` | Python tests ONLY, pytest without the ruff step. Same passthrough as `check-py`, so `scripts/test-py tests/test_hihat_split.py::test_label_for_priority` runs a single test. Use this when iterating on a failing test or after a change you know is lint-clean; reach for `check-py` when you actually want the full post-change loop. Requires `transcriber/.venv`. |
| `scripts/check-ts [bun test args]` | Frontend: `stylelint --fix` + `tsc --noEmit` + `bun test`. Args pass through to `bun test`. |

Scripts autofix where possible (ruff `--fix`, stylelint `--fix`) and
exit non-zero on the first failure. They surface pre-existing lint debt
the same way they surface new debt; fix or escalate, don't silently
ignore.

**Direct pytest invocations are denied by the Claude permission config**
(`pytest`, `python3 -m pytest`, `.venv/bin/pytest`, `uv run pytest`, etc.; see `.claude/settings.local.json`). Go through `scripts/test-py` or
`scripts/check-py`; the scripts' own internal pytest call isn't
intercepted because the permission check only fires on the top-level
Bash invocation.

**Direct TS test / build / typecheck invocations are denied the same
way** (`bun test`, `bunx tsc`, `bunx vite …`, `bunx stylelint …`,
`bunx playwright …`, `bun run test|e2e|lint:design`; see
`.claude/settings.local.json`). Go through `scripts/check-ts`; it
chains `stylelint --fix` + `tsc --noEmit` + `bun test` in the right
cwd with the right flags, and any args pass through to `bun test` so
single-file iteration still works. Same nesting trick as the pytest
denies: the script's own internal `bun test` / `bunx tsc` calls aren't
intercepted because the permission check only fires on the top-level
Bash invocation. `bun run build` stays allowed as a separate top-level
shortcut for the full production build chain (`lint:design` + tsc +
Vite). Without these denies, agents reach for a bare `bunx tsc --noEmit`
or `bun test` to "quickly verify" a change and skip the stylelint pass, that lets design-token lint failures (see §5.8) into commits.

**`bun run dev` is for humans, not agents; don't run it.** It's a
long-running interactive Vite watch process; agents can't keep it
alive between turns, and the dev server isn't needed to verify a
change compiles. When you want a "does this build?" smoke test on
top of `scripts/check-ts`, run **`bun run build`** instead; it
chains `lint:design` + `tsc --noEmit` + the production Vite build
and exits with a clear pass/fail. Leave `bun run dev` to the user;
if you genuinely need browser verification, ask the user to start
the dev server themselves.

The rest of this section documents the direct commands the scripts
wrap, for reference / human-driven workflows.

Always use **bun**, not npm. From the repo root:

| Command | What it does |
|---|---|
| `bun install` | Install npm deps + project. |
| `bun run dev` | Start Vite dev server on http://localhost:5173. **Human-only, agents must not run this** (long-running watch process); use `bun run build` for a "does this compile?" check. |
| `bun run build` | Run `lint:design` + `tsc --noEmit` then Vite production build. Agent-friendly smoke test on top of `scripts/check-ts`. |
| `bun test` | Run all tests via Bun's test runner. |
| `bunx tsc --noEmit` | Typecheck only (fast iteration). |
| `bun run lint:design` | Design-token lint: fails on hex / rgba literals in any `src/**/*.css` outside `src/design_tokens.css`. See §5.8. |
| `bun run preview` | Serve the production build. |
| `bunx --bun vite build` | Direct Vite build (used inside `bun run build`). |

Similarly, never use npx, use bunx instead.

For the transcriber (Python service) container workflow:

| Command | What it does |
|---|---|
| `cd transcriber && cp .env.example .env` | Bootstrap config; fill in `ANTHROPIC_API_KEY`. |
| `docker compose up --build` | Build + run the service. First-ever build downloads ~3 GB of separation model weights into a Docker volume; later restarts skip that. |
| `docker compose logs -f transcriber` | Tail logs. The container is ready when you see `Startup complete in N.NNs - service is ready`. |
| `curl http://localhost:8001/health` | Readiness probe; only returns 200 after eager-load completes. |

The frontend's Vite dev server proxies `/api/*` to
`http://localhost:8001`, so the "Transcribe audio" toolbar button works
end-to-end as long as the Docker service is up.

**Python env for agents (read before touching `transcriber/`):**
The dev box has a local uv-managed venv at `transcriber/.venv` with the
full dep set (torch cu128, madmom, audio-separator[gpu], adtof_pytorch),
which is now the **primary dev loop** — Docker is only required for
clean / reproducible builds. `scripts/check-py` activates it
automatically. When working in the venv directly:

- Invoke Python as **`python3`** on the bare system (no `python`
  symlink); inside an activated venv, plain `python` works.
- **Do NOT install or upgrade deps unprompted.** The install graph is
  ordering-sensitive (numpy/cython before madmom; torch from the cu128
  index before everything else). If a change adds a runtime dep, flag
  it and let the user run `uv pip install`. See the user's notes on
  install ordering before proposing dependency changes.
- `scripts/check-py` is the canonical way to lint + test. If you must
  bypass it: `cd transcriber && source .venv/bin/activate` first.

Current `bun test` count: **103 passing, 2 skipped** (the skipped
tests are fixture suites — drop `.mid` files in
`src/midi/__tests__/fixtures/` or `.rlrr` files in
`src/rlrr/__tests__/fixtures/` to activate them). `src/recompose.test.ts`
(12 tests) is the authoritative correctness coverage for
recomposition and runs locally. The Python suite (`transcriber/tests/`,
Docker-only) gained `test_llm_spec_subset.py` this session.
(`bun test` is scoped to `src/` via `bunfig.toml` so it never picks up
the Playwright e2e specs — see below.)

### 3.1 End-to-end tests (Playwright)

Playwright drives headless Chromium against the Vite dev server.
**Two runners, deliberately separate**: `bun test` owns the unit tree
(`src/`), `playwright test` owns `e2e/`. They never overlap.

| Command | What it does |
|---|---|
| `bun run e2e` | Run the Playwright suite (auto-spawns `bun run dev`, or reuses a running one). |
| `bun run e2e:report` | Serve the last HTML report on `0.0.0.0:9323`. |
| `bunx playwright test e2e/foo.spec.ts:42` | Run a single test by file:line. |

Setup on a fresh box (the dev box is a **headless container**):

1. `bun install` pulls `@playwright/test`.
2. `bunx playwright install chromium` downloads the browser binary
   (per-user `~/.cache/ms-playwright`, not in-repo).
3. **System libs are the one gotcha** — Chromium won't launch without
   `libnspr4`/`libnss3`/etc. Run `sudo bunx playwright install-deps
   chromium` (or bake the apt set into the devbox image). Verify with a
   bare `chromium.launch()`; `error while loading shared libraries:
   libnspr4.so` means the libs are missing.

Container specifics already encoded in `playwright.config.ts`:
`--no-sandbox` is set; `--disable-dev-shm-usage` is deliberately NOT
(the container's `/dev/shm` is sized to 2 GB — forcing shm through
`/tmp` is a measurable perf hit). If you see opaque "Target closed"
crashes under parallelism, check shm size before reaching for that flag.

**Debugging is trace-viewer driven** — no display for `--headed` or the
Inspector. `trace: 'on-first-retry'` + `video: 'retain-on-failure'` are
configured; `bun run e2e:report` + port-forward 9323 is the forensic
surface.

**Probe surface**: `src/index.tsx` exposes `window.jotPlayer`
(playback singleton) and `window.drumjot` (live app instance, has
`.store`). Assert JS-only state (decoded `AudioBuffer`s, player state)
through those rather than scraping the DOM; use `data-testid`
(`stem-row-*`, `stem-mute-*`, `stem-waveform-*`, ...) for element
scoping. The transcriber backend is stubbed via
`e2e/helpers/transcriber-mock.ts` (`page.route`), so e2e needs neither
the Python service nor a GPU.

**Audio caveat**: headless Chromium runs the Web Audio graph (you can
assert scheduling, gain, `currentTime` advance), but you cannot verify
audio *fidelity* — "does it sound in sync" still needs a human
ear-check. The `playback starts...` test depends on the smplr sample
CDN (`smpldsnds.github.io`); a network-restricted box fails it with a
real load error, not flake — that's intentional, not masked.

### 3.2 Playwright MCP

`.mcp.json` registers a project-scoped `@playwright/mcp` server
(headless, `--no-sandbox`, persistent profile under
`.playwright/mcp-profile`, `--save-session` for replayable capture,
output to `.playwright/mcp-output`). Same headless story as the test
framework: the agent perceives via accessibility tree + screenshots
(both produced correctly headless); you can't spectate live or
intervene by hand. Co-located with Claude Code on the dev box, so no
networking config. All `.playwright/` paths are gitignored.

---

## 4. Session timeline

The conversation built up the current repo in this rough order. Each
phase produced concrete files you can find in the layout above.

1. **Vite migration + DSL data model rewrite**. The repo started as an
   esbuild-based React app with a hand-built drum data model. We
   migrated to Vite + Bun and rewrote `src/dsl.ts` to match the DSL
   spec.
2. **DSL parser**. Recursive-descent parser in `src/parser/` with full
   support for notes, rests, groups, simultaneity, weights, repeats,
   rolls, modifiers, sticking, metadata, patterns (definitions + refs +
   substitutions), macros, and per-bar metadata snapshots.
3. **MIDI <-> Jot**. `src/midi/` with assumption-tagged converters
   (channel 10 only, 16th-note quantization on read, raw MIDI note +
   velocity preserved on `note.metadata.midi` for lossless round-trip,
   fixed 480 PPQN on write, GM percussion mapping).
4. **RLRR <-> Jot, RLRR <-> MIDI**. `src/rlrr/` ported from
   ParadiddleUtilities's `midiconvert.py`, plus the inverse direction
   (Python tool is one-way; we added the missing direction).
5. **Renderer polish**. Pattern brackets (solid for definitions,
   dashed for usages, click to highlight via selection store);
   lane gutter for instrument labels; instrument mapping rename
   (`mapping` -> `instrumentMapping`, `NoteMapping` -> `Instrument`);
   bar padding so notes don't overlap bar lines.
6. **Renamed UI/data terms** for clarity:
   - "Voice" is strictly "one side of `||`". Voices can have a `name`
     for display ("Hands", "Feet" in `rockJot`).
   - "Instrument" replaces "drum part" in vocabulary.
   - "Pitch" stays as the DSL letter; `Instrument` is what a pitch
     resolves to.
7. **Transcriber MVP** (`transcriber/`). FastAPI service with eager
   model loading; Docker with GPU passthrough; Vite proxy + toolbar
   "Transcribe audio" button. Uses Demucs + Jarredou + librosa +
   Claude.
8. **Refinement pipeline**. Per-stem onset diff against the source
   stems; LLM-based revision with score gating (only accept revisions
   that improve onset F1); critic LLM (Haiku) for issue triage;
   best-of-K wrapper that generates K candidates and picks the
   best; constrained DSL output via retry-on-parse-error; multi-level
   refinement (macro / structure / onsets / velocity).
9. **Beat-aware overhaul**. Replaced the constant-tempo + 1/16 grid
   model with madmom beat tracking. Onsets are now positioned in
   `(bar, beat_in_bar)` coordinates. Per-bar feel detection
   (straight16 / straight8 / triplet / shuffle / sparse / mixed).
   Parser now snapshots active `time` + `bpm` onto each bar's metadata
   so per-bar tempo / time-signature changes survive the AST. Bun
   bridge honours per-bar `bpm` when computing onset times. Triplets
   and tempo changes are now first-class.
10. **Audio format support**. Frontend `accept` attribute explicitly
    lists wav, flac, mp3, aac/m4a, opus, ogg; backend goes through
    librosa + ffmpeg so anything ffmpeg understands works.
11. **Naming cleanup**. `self_consistency_samples` was renamed to
    `best_of_k` end-to-end (Python + TS + UI + CLI + docs); the old
    term was misleading (it's plain best-of-K with a score function,
    not self-consistency in the LLM-paper sense).
12. **Browser drum playback**. `src/playback/` added. Play button
    schedules events through smplr (TR-808 default; samples fetched
    from a CDN on first click). Travelling playhead reads live bar
    widths so zoom changes during playback don't desync. Per-row mute
    / solo buttons in the gutter take effect mid-playback by cancelling
    and rescheduling. Stop button actually stops (smplr's
    `drums.stop()` only halts currently-sounding notes; we collect the
    per-note stop fns from `drums.start({ time })` and invoke them all).
    Playback-speed dropdown (0.25×–1.25×) re-anchors mid-flight and
    spaces audio times by `1/speed` so pitch is unchanged.
13. **`/transcribe/resume` endpoint**. Re-run from a chosen pipeline
    stage using a previous debug folder's intermediates. Cuts iteration
    cost when debugging the LLM stage (the 30–60 s separation step is
    skipped). See §5.6.
14. **Named-stage pipeline runner**. `pipeline/runner.py` unifies the
    pipeline into six explicit stages (`stems_all`, `stems_per`,
    `beats`, `onsets`, `transcribe`, `refine`). Both endpoints
    dispatch through `run_pipeline(start_stage=...)`. See §5.7.
15. **Anacrusis-aware initial time signature**. `_summarize` in
    `beats.py` now excludes bar 0 when ≥2 bars are present. Madmom
    correctly emits beat_in_bar=2,3,4 for a 3-beat pickup, but the
    old code derived `initial_time_signature` from `bars[0]` and
    labelled the whole song as 3/4 with `has_time_sig_changes=true`.
    Bar 0's own `time_signature` is left as-is (accurate for its
    actual beat count).
16. **Onset detector retuning**. `pre_max`/`post_max` went 20→3,
    `wait` 5→3, `pre_avg`/`post_avg` 100→50. The wide windows were
    over-defensive for full-mix detection; we run on per-instrument
    stems where transients are well isolated, and the previous
    ±232 ms local-max window was suppressing double-kicks and
    hi-hat 16ths.
17. **Beat-onset alignment (global offset, not per-beat snap)**.
    Neural beat trackers (Beat Transformer especially) report each
    beat at its activation peak, which lags the actual transient by
    ~30–50 ms. Symptom: downbeat kicks land at `beat_in_bar≈1.18`
    instead of `1.00`, the LLM then either drops them as "off-grid"
    or snaps them to the wrong slot.
    `pipeline/beats.py::align_beats_to_onsets` computes, for each
    beat, the delta to the **strongest** drum onset within ±50 ms
    (strongest, not closest — a quiet ghost hat shouldn't outrank a
    louder kick further out), takes the **median of those deltas**,
    and shifts the *entire* grid by that single offset. It does
    **not** re-time beats individually. An earlier version snapped
    each beat to its own nearest onset; that removed the lag but also
    folded the drummer's micro-timing into the grid, so inter-beat
    gaps varied beat-to-beat and per-bar `60/mean(gap)` tempo wobbled
    5–10 BPM on dead-steady songs (the LLM then emitted a `{{ bpm }}`
    change between nearly every bar). A uniform shift removes the
    systematic lag while leaving inter-beat gaps — hence per-bar
    tempo and any genuine accelerando the DBN tracked — untouched.
    The shift is gated by `MIN_ALIGN_COVERAGE` (.30): if fewer than
    30 % of beats had a nearby onset the grid is left as the tracker
    produced it. `_rebuild_bar_fields` then refreshes per-bar
    `start_time` / `end_time` / `tempo_bpm` and the global initial
    tempo. Drum-stem onsets are detected once on `ctx.drum_stem`
    inside `_do_beats` and passed into `analyze_beats(..., align_onsets=...)`.
18. **Per-bar tempo stabilization**. `analyze_beats` runs
    `_finalize_bar_tempos` after alignment and before trailing-bar
    padding. It median-filters the raw per-bar tempos
    (`TEMPO_SMOOTHING_WINDOW`=5) only to *decide* whether the song
    has real tempo motion: if the smoothed reference-bar span is
    under `SUSTAINED_TEMPO_CHANGE_BPM` (8.0) every bar is pinned to a
    single global tempo (median of raw bar tempos, bar 0 excluded as
    a likely anacrusis) and `has_tempo_changes` is set False;
    otherwise the smoothed contour is kept and the flag is True. This
    replaced the old `len({round(bpm,1) …}) > 1` test that flipped
    `has_tempo_changes` true on 0.1 BPM jitter. The transcribe
    prompt's per-bar `{{ bpm }}` threshold was also relaxed (2 → 4
    BPM). Net effect with #17: a constant-tempo song yields a flat
    `beats.json` and zero inline tempo blocks.
19. **`missing_onset` is pattern-aware for repetitive pitches**.
    `pipeline/diff.py::_is_subpulse_flicker` suppresses a
    `missing_onset` issue when an unmatched source onset on a
    repetitive pitch (`REPETITIVE_PITCHES = {"h","d"}`) sits between
    two predicted hits ~one local pulse apart and is weaker than the
    median of the kept hits — i.e. it would only re-introduce 1/16
    hi-hat spam the transcription deliberately thinned. Kick/snare
    are excluded (a missing one is almost always real). A ~2× bracket
    (a genuinely skipped pulse position) is still reported. The
    onsets refine prompt (`refine_onsets.md`) was also reframed:
    flagged issues are *evidence the LLM may overrule on musical
    grounds*, not commands — the old "address every issue, don't
    selectively fix a subset" constraint was removed so the loop
    stops fighting the transcription. The F1 accept-gate makes
    "reject the bogus issues" a safe no-op that ends the level
    rather than a regression.
20. **Pattern definitions are always silent**. `[Name=(...)]` defines
    a pattern but does not play it at the definition's position; only
    `[Name]` references play it. The `?`-prefixed silent form
    (`[?Name=(...)]`) and the older "plays at its position" semantics
    have both been removed — there is now only one definition form.
    To play a pattern at the same time as defining it, follow the
    definition with an explicit reference: `[Name=(...)][Name]`.
21. **Stem deliverables + in-app stem tracks**. `stems_all` now also
    sums bass+other+vocals into a drumless mix; both it and the drum
    stem are FLAC-encoded into a per-request `/outputs/<id>/` folder
    served by FastAPI `StaticFiles`, surfaced as
    `TranscribeResponse.{drum_stem_url,no_drums_url}`. Frontend gained
    "Load music stem" / "Load drum stem" toolbar buttons, a
    music-tracks header above the score (per-stem gutter + Canvas
    waveform aligned to the bar timeline + mute/solo), and
    `StemPlaybackController` so loaded stems play through the shared
    AudioContext in sync with the MIDI (speed changes + mute via
    per-stem GainNode). See `src/playback/stems.ts`.
22. **Playwright e2e + MCP**. Headless Chromium against the Vite dev
    server; `bun test` scoped to `src/` so the two runners don't
    collide. `window.jotPlayer` / `window.drumjot` added as the e2e
    probe surface; `data-testid`s on the stem rows. See §3.1 / §3.2.
23. **Renderer UX pass (tuplets, density width, seek, lead-in)**.
    Four related `src/jot.ts` + `src/jot_view.tsx` + `src/playback/`
    changes:
    - **Non-straight indicators**. `layoutBarContents` flags any
      non-pattern `group` whose internal weight fractions aren't
      dyadic as a `TupletSpan` (count = slot count) → a numbered
      bracket over the notes; plus a per-note `straight` flag
      (`isDyadic(beat)`) drives a dotted-ring fallback for off-grid
      notes not under a bracket. Bracket `endBeat` is the *last
      slot's onset* (not group end) so the right leg sits on the
      final notehead; `coveredByTuplet` upper bound is therefore
      inclusive. Label is the bare slot count — see §8.17 caveat.
    - **Density-driven bar width**. `measureOnsetDensity` (max
      onsets/beat across voices) → a clamped `densityFactor`
      threaded through `beatsToPx`, so sparse scores compress and
      busy ones expand. Reference is 2 onsets/beat → factor 1.0
      (typical rock unchanged). Default `barWidth` also dropped
      640→448.
    - **Click-to-seek**. `JotPlayer.seek()` (idle = park + cue next
      play; playing = live re-anchor/reschedule; paused = re-anchor
      vs frozen clock). `xToTime` inverse of `timeToX`. Bars row +
      stem waveforms get a click handler; notes / pattern label
      carry `data-noseek`. `cued` makes the playhead show before
      play; `JotPlayer.stop()` on jot replace.
    - **Lead-in pre-bars**. A jot with `globalMetadata.drumsT0Sec`
      reserves `leadInPx` of hatched pre-roll before bar 1, scaled
      at the bars' exact px/s so a loaded music-stem waveform lines
      up with the notation (drum entrance under bar 1). `timeToX`
      maps `[-drumsT0Sec,0)` into the lead-in; the rAF clamp now
      allows negative `currentTime` so the playhead travels it.
      See §8.7 / §8.18.
24. **Per-instrument transcription + deterministic recomposition.**
    The transcribe stage no longer makes one whole-chart LLM call.
    `llm.transcribe_all_instruments` makes **one call per drum pitch**
    (parallel, `instrument_concurrency`), each emitting a single
    monophonic line — no `+`, no `||`, no `{{}}` block — against the
    shared `beats.py` frame; the model picks only its own subdivision
    (so genuine polyrhythm is expressible). Recomposition lives in
    **`src/recompose.ts`** (single-sourced with the parser, §5.2; local
    `bun test` coverage) and is called via `tools/recompose_jot.ts`;
    `app/pipeline/recompose.py` is a thin subprocess wrapper. It merges
    the lines in `(bar,beat)` space from the parsed resolved positions
    (not string-splicing): concurrent hands become `a+b` on the
    *minimal common* grid; hands whose only common grid is an LCM
    blow-up of genuinely different subdivision families (e.g. straight
    8ths vs triplets) become `(A) + (B)` polyrhythm groups (SPEC
    §"Notes…"/ex.4); kick (`recompose.FEET_PITCHES`, just `{"k"}` —
    the separator has no hi-hat-pedal stem) becomes the second `||`
    voice. **`||` is gone
    only from the LLM-facing spec** (`llm._load_spec_subset` strips the
    "Global simultaneity" section at load time; `SPEC.md` is untouched
    on disk and `||`/`+`/`Voice` remain fully in the DSL, parser, and
    renderer). Refinement mirrors this: `refine.refine_jot_per_instrument`
    runs the F1-gated ONSETS/VELOCITY loop per instrument (scored on
    that pitch alone, parallel), recomposes, then runs LINT+MACRO once
    on the merged Jot via the existing `refine_jot`. STRUCTURE
    (cross-instrument pattern factoring) and `@stick` generation are
    deliberately dropped (neither was previously generated; flatter but
    correct output — see §5.5). Best-of-K is now **per instrument**
    (K candidates per pitch, each scored on its own F1);
    `BestOfKLog.per_instrument` carries the breakdown. `initial.jot`
    stays the stage artifact (plus `initial_<pitch>.jot` per pitch);
    resume-from-refine has no per-pitch fragments on disk so it falls
    back to whole-chart `refine_jot`. The build/test constraint in §3
    (Python is Docker-only on the dev box) was added this session.
25. **Frontend design system + lint** (see §5.8). Three new
    source-of-truth files:
    `src/design_tokens.css` (global `:root` color / radius / shadow
    tokens, loaded via `index.tsx`),
    `src/typography.module.css` (use-case typography classes pulled in
    by other modules via `composes:` — `heading`, `body`, `bodyMd`,
    `bodyEmphasis`, `label`, `labelSm`, `metaLabel`, `metaCaption`,
    `readout`, `readoutSm`, `subtle`, `mono`),
    and `src/jot_view/components/icon_button.tsx` (shared
    `IconButton` + `MuteButton` / `SoloButton` / `ClearButton` used by
    both AudioTrackRow and PitchRow in the mixer, replacing
    ~50 lines of duplicated JSX).
    Every hex / rgba / shadow / radius literal across the five
    `*.module.css` files was migrated to `var(--…)` references; the
    standalone slider thumb restyle landed alongside this so the
    knob sits flush inside the track on all sliders. Stylelint
    (configured via `.stylelintrc.json` — `color-no-hex` +
    `function-disallowed-list` for `rgb`/`rgba`/`hsl`/`hsla`, with an
    `overrides` block exempting `src/design_tokens.css`) enforces "no
    naked color literals" and runs at `bun run lint:design` /
    `bun run build`. Stylelint is configured **without a preset** —
    presets like `stylelint-config-standard` would fire on
    CSS-modules conventions (`composes:` as unknown property,
    camelCase class names vs default kebab-case selector pattern).

---

## 5. Architectural decisions and their rationale

### 5.1 The DSL is the lingua franca

Every conversion (audio, MIDI, RLRR) targets the DSL. The DSL was
designed before the transcriber existed, so anything we want from
audio transcription has to be expressible in it. The DSL spec already
covered triplets (`(...)_N` groups), tempo changes (inline `{{bpm}}`),
time-signature changes (inline `{{time}}`), velocity (`:a` / `:g` /
`vol`), and pattern reuse (`[Name=(...)]`) — so the transcriber's job
is to choose the right DSL constructs, not invent new ones.

### 5.2 Single source of truth for the parser (and DSL manipulation)

The DSL parser exists in TypeScript only. The Python transcriber
service shells out to `bun run transcriber/tools/jot_to_onsets.ts`
when it needs to interpret a Jot (extract predicted onsets for the
diff). This avoids maintaining two parsers in lockstep at the cost of
adding `bun` to the Docker image (~50 MB).

The same principle now extends to **recomposition**: merging the
per-instrument monophonic lines into one Jot is a DSL-manipulation
task, so it lives in `src/recompose.ts` (next to the parser, with
local `bun test` coverage) and the Python side calls it through
`tools/recompose_jot.ts` exactly like the onset bridge.
`app/pipeline/recompose.py` is a thin subprocess wrapper that owns
only the domain facts the TS layer can't know (`FEET_PITCHES`,
display names) and shapes the `BeatStructure` into the bridge's JSON.
There is deliberately **no recomposition algorithm in Python**.

The Docker `build context` is the **repo root**, not the transcriber
folder, so `src/` (TS parser) and `transcriber/` (Python service) end
up in the same image. The Dockerfile synthesises a `tsconfig.json`
at `/app/` inline so bun can resolve the `src/*` path alias in the
container. Locally, bun finds the repo-root tsconfig automatically
by walking upward from the script path.

### 5.3 Beat-relative coordinates, not fixed grids

The transcriber operates in `(bar, beat_in_bar)` space throughout.
This was chosen over a 1/16 grid (or even a 1/48 LCM grid) because:

- Triplets become a property of intra-beat fractions
  (0.000 / 0.333 / 0.667) and the LLM can recognise them by feel.
- Tempo changes mid-track work naturally — each beat has its own
  absolute time anchor.
- Time-signature changes are detected from gaps between downbeats.
- Tolerances stay musically sensible at any tempo (a "beat fraction"
  is invariant to bpm).

The implementation lives in `transcriber/app/pipeline/beats.py` and is
backed by madmom's `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`.

### 5.4 LLM-in-the-loop, deterministic everywhere else

The LLM's job is the **symbolic translation** layer, now applied
**one instrument at a time** (timeline #24): filter that instrument's
candidate onsets and emit a single monophonic line. Polyphony assembly
(hands `+`, hands/feet `||`, polyrhythm groups) is *deterministic*
recomposition, not an LLM responsibility. Everything below
(separation, onset detection, beat tracking, scoring) is deterministic.
Everything above (the rendered React UI, MIDI/RLRR export) is also
deterministic. This shape keeps the LLM cost bounded (~$0.05–0.30 per
song) and makes the failure modes inspectable.

The refinement loop is **monotone-improving**: each iteration computes
issues from a diff against the source stems, asks the LLM to revise,
re-extracts onsets via the bun bridge, scores via mir_eval F1, and
**only accepts** revisions that improve the score. The loop stops
when score plateaus or after ~3 onset iterations.

The cheap **critic** LLM (Haiku by default) triages the deterministic
issue list before the expensive **generator** LLM (Opus 4.7) sees it
— that's the standard two-LLM pattern for cost reduction (~10x).

### 5.5 What doesn't exist (deliberately)

- **No audio-native LLM step**. The user explicitly excluded Gemini /
  GPT-4o audio review from the current scope. The pipeline does not
  feed raw audio to any LLM; only structured candidate lists.
- **No cross-instrument musical context in transcription** (timeline
  #24). Each per-instrument call sees only its own onsets + the shared
  beat frame, so it can't use groove context (e.g. "the kick is on the
  & so this hi-hat gap is intentional") to disambiguate. This is the
  main accuracy trade-off of the per-instrument split, accepted in
  exchange for reliability/parallelism/isolatable failures. The
  mitigation (feed each call a read-only summary of other instruments'
  onset positions) was designed but **not built** — it's the first
  lever if real-audio accuracy disappoints.
- **No cross-instrument pattern factoring or `@stick` generation**
  (timeline #24). Per-instrument lines have no shared bar patterns to
  factor (`[Groove=…]`), and sticking is inherently cross-hand so it
  can't come from monophonic calls. Neither was generated before the
  refactor either, so output is flatter (more verbose DSL) but not a
  regression. v2 candidates: a deterministic identical-consecutive-bar
  → repeat/pattern pass, and a deterministic alternating-sticking pass,
  both post-recompose.
- **No FluidSynth round-trip render-and-diff**. We compare predicted
  onsets (from the bun bridge) to source onsets directly in
  seconds-space; this is functionally equivalent to rendering and
  re-onset-detecting but cheaper and lower-noise.
- **No per-bar tempo issue type** (yet). The macro pass only diffs the
  global initial tempo; the onset pass picks up per-bar mismatches
  implicitly through missing/extra onsets. Per-bar tempo issues are a
  reasonable follow-up. Note this is *deliberately* paired with
  `_finalize_bar_tempos` (timeline #18): per-bar tempos are either
  pinned to one global value or a smoothed contour, so there is no
  raw-wobble signal for a per-bar tempo diff to chase even if one
  existed. Any future per-bar tempo issue type must diff against the
  *finalized* tempos, not the raw `60/mean(gap)` estimates.

### 5.6 The `/transcribe/resume` endpoint

Re-runs the pipeline from a chosen stage onward, hydrating earlier
stages' artifacts from a previous run's debug folder. Form params:

- `resume_folder` — absolute path or bare folder name under
  `DEBUG_DIR` (default `/debug`). A path-traversal guard rejects
  anything outside the configured base.
- `resume_stage` — one of `stems_all`, `stems_per`, `beats`, `onsets`,
  `transcribe`. Required.
- `beat_input`, `include_candidates` — same as `/transcribe`.

Required upstream artifacts depend on `resume_stage`:

- `stems_per` needs `stems_all/drum_stem.<ext>`.
- `beats` needs `stems_per/*.<ext>` (consumed downstream by `onsets`).
- `onsets` needs `stems_per/*.<ext>` + `beats.json`.
- `transcribe` needs `beats.json` + `onsets.json` + `stems_per/*.<ext>`.

Missing artifacts surface as HTTP 400 with a message naming which
`resume_stage` value would regenerate them. Output overwrites the
artifacts produced by the chosen stage and any stage after it;
upstream artifacts are preserved so re-resuming the same folder is
idempotent.

### 5.7 Named-stage pipeline runner

Both endpoints dispatch through `pipeline/runner.run_pipeline()`. The
six stages and their data dependencies:

```
stems_all  -> drum_stem               (consumed by stems_per)
stems_per  -> per_instrument_stems    (consumed by onsets)
beats      -> structure               (consumed by onsets, filter, transcribe)
onsets     -> onsets_by_pitch         (consumed by filter, transcribe)
filter     -> kept_by_pitch           (consumed by transcribe)
transcribe -> predicted_midi          (kept-onsets MIDI deliverable)
```

`filter` runs the per-instrument filter LLM
(`filter_onsets_all_instruments`, one parallel Anthropic call per drum
pitch) and persists the surviving onsets to `filter/kept_onsets.json`.
`transcribe` is the deterministic render: `onsets_to_midi_bytes` +
`build_note_provenance` against `kept_by_pitch`, writing
`prediction.mid` and `note_provenance.json`. The split lets the
operator iterate on the render / provenance code without re-paying
for LLM calls; `resume_stage=transcribe` re-hydrates
`kept_by_pitch` from `filter/kept_onsets.json` (with identity re-
threaded against `onsets.json` so `build_note_provenance`'s `id(c)`
kept-vs-rejected match still works) and re-runs only the render.

There is no refine stage: the legacy F1-gated multi-level refinement
loop was removed in May 2026 along with the DSL-output pathway it
depended on. See `transcriber/docs/ai-midi-to-jot-notes.md` for the
captured techniques.

Each `_do_<stage>(...)` function reads its inputs from a shared
`PipelineContext`, writes its output back, and persists artifacts via
the debug sink. Stage failures wrap into `StageError(stage,
original)`; the HTTP layer maps `StageError.stage == FILTER` to
502 (LLM is external) and everything else to 500.

The runner is the single place where the pipeline order lives.
Adding a stage means: extend the enum, add a `_do_*`, add a case to
`_run_stage`, add a hydration entry in `resume.py`.

### 5.8 Frontend design system

The frontend has a small, deliberate design system. **Use it.** Don't
hardcode colors, radii, shadows, or shared text styles in module CSS —
that's exactly the kind of drift the design-token lint catches.

**Three source-of-truth files**:

- **[src/design_tokens.css](src/design_tokens.css)** — global `:root`
  custom properties for every color, border-radius, and shadow used in
  the app. Loaded once from [src/index.tsx](src/index.tsx) as a plain
  (non-module) stylesheet so the vars are visible to every module.
  Grouped semantically: `--color-bg-*`, `--color-border-*`,
  `--color-text-*`, `--color-accent-*`, `--color-error-*`,
  `--color-success-*`, `--color-busy-*`, `--color-toggle-active-*`,
  `--radius-{xs,sm,md,lg,xl,circle,pill}`, `--shadow-{sm,md,lg,note,
  playhead-label,accent-glow}`.
- **[src/typography.module.css](src/typography.module.css)** — one
  CSS-modules class per typography **use case**, not per atomic
  property: `heading`, `body`, `bodySm`, `bodyMd`, `bodyEmphasis`,
  `label`, `labelSm`, `metaLabel`, `metaCaption`, `readout`,
  `readoutSm`, `subtle`, `mono`. Each bundles font-size + font-weight
  + letter-spacing + text-transform + line-height + family for its
  use case in one place.
- **[src/jot_view/components/](src/jot_view/components/)** — shared
  React UI primitives. Today: `icon_button.tsx` (the 18×18
  `IconButton` base + `MuteButton` / `SoloButton` / `ClearButton`
  used by the mixer rows). New shared chrome goes here, not inlined
  into the page-level modules.

**How consumers use it**:

```css
/* module CSS: pull in a use case, then layer your own chrome. */
.toolbarLabel {
    composes: labelSm from '../typography.module.css';
    color: var(--color-text-muted-alt);
}

.transcribeButton {
    composes: bodyEmphasis from '../typography.module.css';
    padding: 5px 12px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-accent);
    color: var(--color-bg-base);
    /* …rest of per-component chrome */
}
```

**`composes:` constraint** — CSS Modules' `composes:` directive only
works in a rule whose selector is a *single bare class*: `.foo` is
fine, `.foo dt` / `.foo:hover` / `.foo > .bar` are NOT (postcss-modules
errors with *"composition is only allowed when selector is single
:local class name"* and the build fails). When typography is needed on
an element targeted by a compound selector (an `<h4>` inside a card, a
`<dt>` inside a `<dl>`), inline the props on the compound rule and
leave a comment naming which use case it mirrors so future edits to
the use case can be propagated by hand. The current exceptions are
`.debugPanelColumn h4` (toolbar.module.css) and `.debugDetailsList dt`
(score.module.css). The alternative — adding a `className` to every
inner element in JSX — is cleaner for one or two call sites but quickly
becomes verbose (11 `<dt>` inserts in NoteProvenanceDetails was the
tipping point).

In React inline styles — JSX `style={{ … }}` props accept `var(...)`
strings, so colors still flow through tokens:

```tsx
<div style={{ color: 'var(--color-text-faint-strong)' }}>…</div>
```

**The rules** (rule 1 enforced by stylelint via
[`.stylelintrc.json`](.stylelintrc.json) — run with `bun run lint:design`,
chained into `bun run build`):

1. **No hex literals (`#xxx` / `#xxxxxx`) or `rgb()`/`rgba()`/`hsl()`/`hsla()`
   calls in any `src/**/*.css` outside `src/design_tokens.css`.** Every
   color is a named token. If you genuinely need a new color, add it
   to `design_tokens.css` first and reference the var. Caught by
   stylelint's `color-no-hex` + `function-disallowed-list` rules; the
   `overrides` block in `.stylelintrc.json` is the one place those
   are permitted.
2. **Prefer `composes:` over reinventing typography.** When you find
   yourself writing `font-size: 11px; font-weight: 700;
   letter-spacing: 0.08em; text-transform: uppercase;` from scratch,
   stop — that's a use case (`labelSm`), compose it. Stylelint
   doesn't enforce this (legitimate per-site overrides exist) but
   reviewers will.
3. **Shared visual chrome goes through a primitive.** A 3rd mixer
   row reaching for a 4th hand-rolled M/S button pair is a smell —
   it belongs as a variant of `IconButton`.

**Carve-outs** (these stay as raw values, by design):

- `LANE_COLORS` in [src/jot.ts](src/jot.ts) — an 8-color data
  palette consumed by JS to set per-lane note colors, not styling
  chrome. Lives in TS, not CSS.
- Canvas drawing in [src/jot_view/mixer.tsx](src/jot_view/mixer.tsx)
  (`ctx.fillStyle = '#…'` for waveform painting). The Canvas2D API
  requires literal color strings; using `var(--…)` would mean a
  `getComputedStyle` lookup per paint. Not worth it.
- True one-off typography (the tuplet number's italic, the 22px
  transport-button glyph, single-use `font-size: 14px` empty-state
  text, etc.) — the typography file's header comment lists which
  are deliberately inline. Promoting them to a class for one caller
  would add indirection without consolidation.

**Adding a new visual primitive**:

1. If it's a *color/radius/shadow*, add the token to
   `src/design_tokens.css`. Pick a semantic name; descriptive
   suffixes (`-strong`, `-alt`, `-overlay-medium`) are fine when
   close-but-distinct variants are genuinely needed.
2. If it's a *typography use case* that appears in ≥2 places, add a
   class to `src/typography.module.css` and have consumers
   `composes:` it.
3. If it's a *React UI widget* used by ≥2 pages, drop it under
   `src/jot_view/components/`.
4. Run `bun run lint:design` before committing.

---

## 6. The two paths for accuracy improvement

The conversation researched two parallel approaches to maximum
transcription accuracy. The user picked Path A as the implementation,
Path B remains researched-only.

### 6.1 Path A — Off-the-shelf + LLM (IMPLEMENTED)

Status: shipped in `transcriber/`. Awaiting real-world testing.

Expected accuracy on real-world music: **F1 0.85–0.92** after
refinement, dominated by the upstream-model ceiling. ADTOF caps around
F1 0.87 on clean drum stems; Demucs/Jarredou separation adds residual
bleed that the LLM cleanup partially recovers.

Per-song cost (3-min track on Replicate-tier serverless GPU):
- Separation: ~$0.05
- LLM (single-shot + refinement, no best-of-K): ~$0.15
- **Total: ~$0.20 per song.**

### 6.2 Path B — Train own N2N on Google Cloud TPUs (RESEARCHED ONLY)

The Noise-to-Notes paper (arXiv 2509.21739, Sept 2025) is the current
academic SOTA for drum transcription: 0.897 F1 on E-GMD, 0.879 on
MDB, 0.949 on IDMT. It's a diffusion-based generative model on
EDGE-style transformer + MERT music foundation model features. The
paper has no public code or weights at time of writing; expected drop
around ICASSP 2026 (Feb camera-ready) or ISMIR 2026 (Aug).

User has confirmed they want to train their own N2N. Key facts from
the research:

**Architecture (all public components)**:
- EDGE transformer decoder: github.com/Stanford-TML/EDGE (MIT)
- MERT-330M music foundation model encoder: huggingface.co/m-a-p/MERT-v1-330M (Apache-2.0)
- EDM diffusion framework (Karras 2022): github.com/NVlabs/edm
- Annealed Pseudo-Huber loss (fully specified in the paper)

**Training data plan (all open)**:

| Dataset | Hours | License |
|---|---|---|
| E-GMD | 444 | CC-BY 4.0 |
| STAR Drums | 125 | research |
| ADTOF-YT | 202 | research |
| Slakh drum stems | 118 | CC-BY 4.0 |
| **User's 2000-song corpus** | ~117 base, ~585 after 5-kit rendering | user's own |
| **Total** | **~1500 h after augmentation** | mostly open |

**Rendering of the 2000-song corpus**: via **Drumgizmo** (GPL,
headless CLI, multi-mic kits like DRSKit / MuldjordKit / CrocellKit /
Aasimonster / MorbidStudioKit, all free). ~$50 in GCP CPU spot to
render 2000 songs × 5 kits = 10,000 audio files. Could also add
EZdrummer 3 or BFD3 via Wine + yabridge + Reaper Linux for an
additional commercial-kit timbre.

**Augmentation suite** (proven gains in piano transcription, expected
+3–5 F1 on the harder datasets):
- Mixing augmentation: blend with non-drum stems (from MUSDB18) at
  randomised dB ratios. Directly attacks DTM weakness.
- Standard: SpecAugment (time/freq masking), gain randomisation
  (±10 dB), pitch shift (±2 st), time stretch (±10%), reverb (p=0.2).

**TPU configuration**:
- Recommended: **v6e-1 (single chip) on spot pricing** (~$0.95/chip-hr).
- Compute budget: ~440 chip-hours for the scaled-up training run
  (1500 h data, ~60 epochs, after MERT feature pre-extraction).
- **One training run on spot v6e-1: ~$420.**
- Full ablation program (5–10 runs): **~$3,000–6,500**.
- Mix of on-demand (debugging, final ship-it run) + spot (ablations)
  hits ~$5,000–6,500 total.
- All-on-demand budget: ~$14,000.
- Wall-clock per run on v6e-1 spot: ~22 days. Parallelize ablations
  across multiple single-chip instances to finish in ~3 weeks.

**Expected outcome** if executed well:
- MDB F1 0.92–0.94 (vs N2N paper's 0.879)
- RBMA F1 0.70–0.78 (vs the field's current ~0.56)
- E-GMD F1 0.91–0.93 (already near ceiling)

**Engineering plan** (rough):
1. Re-implement N2N architecture from the paper (1–2 weeks).
2. Build Drumgizmo Docker pipeline + render 2000-song corpus (3 days).
3. Pre-extract MERT features for all data (~$50 GCP).
4. Run ablations (~$5K cloud, ~3 weeks wall-clock parallelised).
5. Final tuned run on full data + best augmentation (~$1200).
6. Total engineering effort: ~6–10 weeks for one person.

**Verification path**: build the F1 eval harness immediately
(`mir_eval` 50ms tolerance on E-GMD / MDB / IDMT / RBMA test splits)
and run ADTOF locally as a proxy baseline. ADTOF is the only modern
SOTA model with public weights — N2N paper reports ADTOF's numbers
too, so the gap is triangulable. ADTOF: github.com/MZehren/ADTOF
(install via `pip install adtof`).

**Email the authors first**. Often gets you weights or sample outputs
within ~30% probability for ICASSP-class papers.

---

## 7. The DSL spec — quick reference for agents

Single-letter pitches `a`–`z` resolve to instruments via
`globalMetadata.instrumentMapping`. Suffixes attach to a primary
element tightly (no whitespace between primary and suffix is the
canonical form, but whitespace is allowed):

| Suffix | Meaning |
|---|---|
| `:mod` | Modifier (a/g/c/h/o/f/s/r/x/z/k/m/l, or multi: fl/dr/rf). Chain with `:a:r`. |
| `@stick` | Sticking (r/l/rf/lf). Only on notes. |
| `_N` | Weight (relative duration in a sequence). |
| `*N` | Repeat the element N times in place. |
| `~` | Roll/buzz. |
| `{...}` | Note/group metadata. |

Top-level structure:

```
{{ globalMetadata }}
[Pattern=(...)] (silent definitions; play via [Pattern] references)
| bar1 elements | bar2 elements |
||
| voice 2 bar 1 | voice 2 bar 2 |
```

Things that look fiddly but matter:

- **Macros vs patterns**: `[$name=...]` is a textual preprocessor
  substitution; `[Name=...]` (no `$`) is a parsed pattern with
  position-aware substitutions (`[Name#3=(x)]`, `[Name#5-8=...]`).
- **The `time` key** in metadata is written as a string (`"4/4"`)
  in DSL but normalises to `{ count, unit }` in the AST.
- **Voice == one side of `||`**. Nothing else calls anything a
  "voice" — see the `Voice` doc comment.
- **Per-bar metadata** lives in `Bar.metadata` (populated by the
  parser snapshot logic). Renderer and bun bridge use it; consumers
  fall back to `globalMetadata`.

The parser is ~600 lines of recursive descent + ~150 lines of macro
preprocessing. Tests cover every spec example.

---

## 8. Known limitations and gotchas

1. **madmom installs from a git main branch**, not from PyPI — the
   last PyPI release (0.16.1) predates modern Python. The Dockerfile
   pins this. If install breaks in a future Python, fallback paths in
   `beats.py` will use librosa beat tracking (no downbeat detection),
   degrading per-bar feel detection to default-`straight16`. A second
   backend, **Beat Transformer**, is also available via
   `settings.beat_tracker = "beat_transformer"`; its activations feed
   the same madmom DBN postprocessor and respect the same `BeatStructure`
   shape, with an optional tempo-window narrowing around its tempo
   head's prediction.
2. **Vector size limits**. A 5-minute song's per-bar prompt is
   currently within Opus's 200k context budget, but a 10-minute song
   could push it. If you hit token limits, batch by song-section.
3. **F1 is noisy**. Sample variation in LLM output can mask small
   real gains. Always run multiple seeds when measuring whether a
   pipeline change helped.
4. **Refinement loop assumes tempo is roughly correct**. If the macro
   pass fails to fix a half/double-tempo confusion, the onset pass
   will see hundreds of bogus issues and the LLM may give up. Watch
   for this in early testing on songs at unusual tempos.
5. **`audio-separator[gpu]` install size**. ~3 GB of PyTorch + CUDA
   wheels. First Docker build is slow.
6. **Mid-bar tempo changes** aren't handled. The DSL spec allows it
   (per the "remain in effect" semantic) but the parser snapshots
   metadata only at bar boundaries. Sub-bar `{{bpm:...}}` blocks
   technically parse but don't affect timing inside the bar they
   appear in.
7. **The TS renderer doesn't apply per-bar tempo to its pixel layout**
   (it only uses `barBeats` for bar widths, which depends on time
   signature not bpm). So a Jot with tempo changes renders all bars
   at the same visual width even if real durations differ. Fine for
   notation; would need work for "playback-accurate" rendering.
   *Two song-global scalars now sit on top of this*: `densityFactor`
   (timeline #23) scales every bar by the song's onset density, and
   `leadInPx` reserves pre-roll space before bar 1 from
   `globalMetadata.drumsT0Sec`. Both are derived at the bars' exact
   px/second (`(barWidth·densityFactor/4)·(bpm/60)`) — that
   equivalence with `buildTimeline` is load-bearing for waveform /
   playhead alignment; change one side and you must change the other.
8. **Stop semantics for browser playback need both halves**. smplr's
   `drums.stop()` only halts notes already sounding; future-scheduled
   notes (via `drums.start({ time })`) keep firing until they reach
   their start time. `JotPlayer.stop()` collects the per-note stopFns
   returned by each `start` call and invokes them all, otherwise Stop
   only stops the playhead animation and the song keeps playing.
   Anyone adding new scheduling paths must push their stop fns onto
   `this.scheduledStops` for Stop to remain truthful.
9. **`<input type="file">` accept is best-effort** across browsers.
   Safari especially can be loose about which MIME types it tags
   non-Apple formats with. The `audio/*` wildcard catches most edge
   cases but not all; backend doesn't enforce format and uses ffmpeg
   for anything libsndfile rejects.
10. **The TS parser tracks active metadata as full snapshots per
    bar**. If you add new fields to `Metadata` that need to propagate
    per-bar (beyond `time` + `bpm` today), update the `BarMeta` type
    in `src/parser/parser.ts` accordingly.
11. **`JotPlayer.currentTime` is in JOT time, not real time.** With
    `playbackSpeed < 1.0`, real elapsed seconds ≠ reported
    `currentTime`. The rAF loop computes
    `startJotTime + (ctx.currentTime - startContextTime) * playbackSpeed`,
    and `setPlaybackSpeed` mid-flight re-anchors both fields so the
    playhead doesn't jump. Anything reading `currentTime` for visual
    sync should keep working; anything timing wall-clock events
    against it needs to divide by `playbackSpeed` first.
12. **Per-instrument stems are the input to `onsets.detect_onsets`,
    not the full drum mix**. The detector windows are tuned tight
    (`pre_max`=`post_max`=3, `wait`=3) on this assumption. If a future
    refactor ever runs onset detection on the full drum stem (or
    full mix), the windows need to widen back to ~20 or you'll get
    spurious double-detections of single transients.
13. **smplr's TR-808 group names are non-obvious.** The kit uses
    `hihat-close` (no trailing `d`), `mid-tom` (not `tom-mid`), and
    has no separate `ride` group — `drums.ts` falls back to
    `cymbal` for ride hits. If you swap to a different kit
    (`Casio-RZ1`, `LM-2`, `MFB-512`, `Roland CR-8000`) those mappings
    will need re-verifying against the new kit's `getGroupNames()`.
14. **The frontend's playback module assumes one global BPM**. The
    timeline anchors to `globalMetadata.bpm` because `toMidi` only
    emits one `setTempo` at tick 0 — per-bar `{{ bpm: ... }}`
    overrides in the DSL aren't carried into the MIDI bytes that drive
    playback, so honouring them in the playhead would drift the
    visual relative to the audio. If MIDI export ever emits multiple
    tempo events, `playback/events.ts` and `playback/timeline.ts` need
    updating in lockstep.
15. **Beat-grid finalization order is load-bearing**. If you change
    the order of operations in `analyze_beats` or `_do_beats`,
    preserve the sequence:
    `tracker → align_beats_to_onsets → _finalize_bar_tempos →
    _pad_trailing_bars`.
    - `align_beats_to_onsets` shifts the **whole grid** by one median
      offset (it no longer per-beat-snaps — see timeline #17). Doing
      it after padding would let the synthetic fadeout bars pull the
      median; doing it before is correct because the pads are built
      from the already-corrected last-real-bar tempo.
    - `_finalize_bar_tempos` must run **before** padding so the pads
      inherit the pinned/smoothed tempo, and **before** the
      `has_tempo_changes` flag is consumed — it overwrites the
      naive flag `_summarize`/`_rebuild_bar_fields` set. Re-running
      `_rebuild_bar_fields` after it would reintroduce the raw wobble.
    - Alignment is also why `detect_onsets(ctx.drum_stem)` runs in
      `_do_beats` before `_do_onsets` re-detects per-stem — these are
      different stems (combined drum vs. per-instrument) and the
      duplicate cost is cheap relative to the accuracy win.
16. **Pattern definitions never play at their position.**
    `[Name=(...)]` only declares the pattern; the body plays exclusively
    through `[Name]` references. To play a pattern at the same point
    you define it, write the reference explicitly: `[Name=(...)][Name]`.
    The older `?`-prefixed silent form has been removed.
17. **Tuplet bracket labels are the bare slot count.** Detection is
    robust (any non-dyadic weight fraction → tuplet), so equal-note
    triplets/quintuplets/etc. label correctly (3/5/6/7…). But the
    label is `el.elements.length`, so a weight-expressed feel like a
    swung pair `(a_2 a)` is flagged and labeled **2** when musically
    it's triplet-based (should read 3, or no number). `(a_3 a)`
    (3:1 dotted) is correctly *not* flagged (dyadic). If this bites,
    options are ratio form (`5:4`) or labeling weighted groups by
    subdivision base; deliberately left simple for now.
18. **Lead-in / seek coordinate invariants.** Click-to-seek and the
    lead-in both assume `bar.x` is the single source of truth for the
    bars-row pixel space (origin after the gutter). The score's bars
    row needs the `leadIn` spacer as its first flex child or the
    flex-positioned bars desync from `bar.x` / the absolutely-placed
    playhead. `seekFromClick` bails on `[data-noseek]` ancestors
    (notes, pattern label) — new clickable score chrome that
    shouldn't seek must opt out the same way. `xToTime` clamps the
    lead-in region to time 0 (seek won't scrub into the pre-roll);
    `timeToX` *does* map negative time into it for the playhead.
    Anything timing wall-clock against `JotPlayer.currentTime` must
    now tolerate negative values during the lead-in (see §8.11).
19. **No naked color literals in CSS modules — the lint will catch you.**
    `bun run lint:design` (also chained into `bun run build`) fails on
    any hex or `rgba()` literal in `src/**/*.css` outside
    `src/design_tokens.css`. If you genuinely need a new color, add
    a named token there first and reference it via `var(--…)`. See
    §5.8 for the broader design-system rules: typography goes through
    `composes:` from `src/typography.module.css`, and shared UI
    primitives live under `src/jot_view/components/`. The lint is
    deliberately scoped to CSS — TSX inline styles (`style={{ color:
    '...' }}`) and Canvas/data-color literals are *not* checked, but
    the same intent applies: prefer `'var(--token)'` strings in JSX
    inline styles where the browser will resolve them.

---

## 9. Things to test next

In rough priority order. Most of these are real-audio tests that
weren't possible in the build/test sandbox.

1. **Run the transcriber end-to-end on a 4/4 rock track at a known
   tempo**. Verify F1 is in the 0.80+ range (initial) and refinement
   lifts it. The success pill in the toolbar reports the F1 delta.
2. **Run on a song with a tempo change** (e.g. song that accelerates
   into a chorus). Verify the LLM emits inline `{{ bpm: ... }}`
   blocks at the right bar boundaries. Watch the
   `has_tempo_changes: true` flag in the response.
3. **Run on a song with triplets / swing** (jazz waltz, shuffle blues,
   triplet fill). Verify `feel=triplet` or `feel=shuffle` shows up on
   the relevant bars and the LLM emits `(...)_N` groups.
4. **Run on a 7/8 or 3/4 song**. Verify `time_signature: "7/8"` or
   `3/4` is detected and time-sig changes (if any) emit
   `{{ time: "..." }}` between bars.
5. **Best-of-K**: set samples=3 or 5 in the UI, verify the
   F1 of the chosen sample is meaningfully better than samples=1 for
   tricky audio.
6. **Try opus/aac/flac/wav files** to confirm the format list works
   end-to-end after our last change.
7. **Drop a few `.mid` files in `src/midi/__tests__/fixtures/`** to
   activate the fixture suite — useful for proving MIDI round-trip
   on real material.
8. **Email the N2N authors** asking for weights or sample outputs on
   E-GMD/MDB tracks; while waiting, set up the eval harness (see Path
   B section above) so a future N2N run can be benchmarked directly.

---

## 10. User communication style and preferences

Observed across the session, useful for tone calibration:

- **Direct, technical, detailed**. They want the rationale behind
  recommendations, not just the conclusion.
- **Decisive**. They say "just do it" or "do the proper complete
  approach now" when they want execution rather than more discussion.
- **Pragmatic over academic**. They care about real-world accuracy on
  their inputs, not benchmark scores in isolation.
- **Comfortable with substantial implementation work in single turns**.
  They explicitly asked for things like "implement the full multi-level
  refinement now" as one task.
- **Has compute budget but watches efficiency**. Quotes for GCP/TPU
  costs are scrutinised; cheap-but-slow is acceptable ("no deadline").
- **Will hand off mid-task**. Multiple `Continue` instructions kept
  the same plan going across interruptions. This very handover is
  proof — they're transferring the session to a remote agent.
- **No emojis in responses unless explicitly asked**. Matches the
  system rules already.
- **Prefers `bun` over npm/yarn**. Verified, baked into project setup.

---

## 11. References / things worth bookmarking

- The DSL spec: [SPEC.md](SPEC.md).
- Parser test cases as the most authoritative DSL examples:
  [src/parser/\_\_tests\_\_/parser.test.ts](src/parser/__tests__/parser.test.ts).
- Transcriber service architecture overview:
  [transcriber/README.md](transcriber/README.md).
- 2509.24853 — the "Enhanced ADT via Drum Stem Source Separation"
  paper, which is essentially a blueprint for our Path A pipeline.
- 2509.21739 — N2N paper; what Path B aims to reproduce + scale.
- ADTOF: github.com/MZehren/ADTOF — runnable proxy baseline for
  benchmarking against N2N's reported numbers.
- E-GMD dataset: magenta.tensorflow.org/datasets/e-gmd — primary
  training data for any Path B work.
- STAR Drums: zenodo.org/records/... (TISMIR 2025 dataset article;
  124.5 h with 18 drum classes).
- madmom: github.com/CPJKU/madmom — beat / downbeat tracking.
- audio-separator: pypi.org/project/audio-separator/ — wrapper around
  Demucs, MDX-Net, BS-RoFormer.
- Jarredou MDX23C 6-stem drum separator:
  github.com/jarredou/models/releases (the `aufr33-jarredou_MDX23C_DrumSep_model_v0.1.ckpt`).
- Drumgizmo + free kits: drumgizmo.org/wiki/doku.php?id=kits — used
  in the Path B rendering pipeline.
- ParadiddleUtilities upstream:
  github.com/emretanirgan/ParadiddleUtilities — origin of the RLRR
  format and the `midiconvert.py` we ported.

---

## 12. Quick sanity checklist if anything's broken

If `bun run build` fails: usually a TS error in a type rename ripple.
Run `bunx tsc --noEmit` to get the full error list.

If a test in `src/parser/__tests__/` fails: the parser is the most
load-bearing piece of the TS code; revert your last parser change
and re-run.

If the transcriber Docker build fails on madmom: switch to a librosa
fallback by setting an env var `DISABLE_MADMOM=1` (would need to be
plumbed through; currently `beats.py` falls back automatically on
ImportError, so madmom can just be removed from `pyproject.toml`
temporarily).

If the bun bridge errors with "Cannot find module": ensure tsconfig
has `paths: { "src/*": ["src/*"] }` and that bun is finding it. The
repo-root tsconfig handles local dev; the Dockerfile synthesises one
at `/app/tsconfig.json` for the container.

If the LLM returns invalid DSL repeatedly: check that
`transcribe.md` prompt's `{SPEC}` placeholder is being filled (look
for the spec text in the generator's first message). The retry-on-
parse-error logic gives one more shot before giving up.

If `bun test` shows 0 tests run: check that `package.json` has the
`test` script pointing to `bun test` and that test files match
`*.test.ts` glob (Bun's default).

If frontend can't reach the transcriber: confirm `docker compose up`
finished startup ("service is ready to accept requests"), and that
`curl http://localhost:8001/health` returns 200. The Vite proxy
target is in `vite.config.ts`.

---

## 13. Open questions for the user

Things you may want to clarify with the user before making sweeping
changes:

1. **N2N training timeline**. Has the user committed to starting Path
   B yet, or is it still "research-complete, execution-pending"?
   At time of writing it's the latter.
2. **The 2000-song MIDI corpus** — they mentioned it but the agent
   hasn't seen it. If they want to start the rendering pipeline,
   ask for the source folder structure.
3. **Refinement defaults**: refinement is currently on by default
   (`REFINE_BY_DEFAULT=true`). For batch workflows where cost
   matters, they might want to flip this.
4. **Audio-native LLM step** was explicitly excluded. If accuracy
   plateaus below the user's target after refinement runs, this is
   the next lever to discuss.

---

End of handover. Good luck.
