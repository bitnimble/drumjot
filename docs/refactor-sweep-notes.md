# Refactor sweep, working notes & findings

Autonomous session against TODO.md "refactor sweep" (bullets 20–26) +
unit-test cleanup + dedup audit + Storybook. Branch
`refactor/frontend-editing-prep`. This file accumulates:

- **Flagged** items: things I judged behaviour-/quality-risky to change
  autonomously, left for human review.
- **Dedup audit** findings (bullet 26).
- Per-feature extraction progress.

Each completed slice is a green-gated commit (`bun run build` +
`scripts/check-ts` + `bun run e2e`).

> **Perf-gate caveat (re-run on a quiet box).** Later in the session the
> shared host got busy (load ~6-7 on 12 cores, lots of background work, the
> Jun-dated multi-hundred-minute `python3` zombies suggest training runs), and
> the 3×120fps perf specs (`src/playback/tests/perf.e2e.ts`) started flaking:
> they need the box quiet to measure the tight 8.3ms median / ≤10%-slow-frame
> budget. **The functional gates are unaffected**, `bun run build`,
> `scripts/check-ts` (291 unit), and all 23 functional e2e stay green for every
> commit. For perf-adjacent structural moves made while the box was loaded I
> verified perf-neutrality by *baseline comparison*: stash the change, run the
> perf spec on the pre-change commit under the same load, confirm the change is
> equal-or-better. (e.g. the F4 dedup measured *better* than its baseline:
> 7.5ms/13-slow vs 9.7ms/16-slow.) **Please re-run `bun run e2e` on a quiet box
> to get a clean 26/26 perf stamp.**

---

## Flagged for review (NOT changed autonomously)

> **Update:** after review the user directed fixes for F2/F3/F5/F6 (all now
> done, see each entry) and asked about F1 (answered below; left as-is, it's
> deliberate). F4 was already resolved during the sweep. So everything in this
> section is now either resolved or a deliberate keep.

### F1. Filter flow inverted: engine PULLS from PlaybackStore. DONE
Originally flagged as a "keep" (the player held pushed snapshots). After
review the user chose to invert it (and confirmed it's fine for the engine to
depend on a store). Now:
- `PlaybackStore` takes `MixerStore` as a dependency and exposes the
  engine-facing computeds `pitchFilter` / `audioTrackFilter` /
  `audioMasterAudible` / `drumMasterAudible`, delegating to the mixer's
  existing computeds (the mixer also consumes `pitchFilter`/`audioTrackFilter`
  for per-row audibility, so the build logic stays in one place).
- `jotPlayer.currentFilter` etc. are now GETTERS that read the late-bound
  `PlaybackStore` (`attachPlayback`); the audio path pulls the computed
  directly instead of caching a snapshot. The `setX` methods became
  parameterless `applyPitchFilter` / `applyAudioTrackFilter` /
  `apply{Drum,Audio}BusGain`.
- The four reactions moved from `MixerPresenter` to `PlaybackPresenter`; they
  carry no data now, just firing the imperative audio-graph re-apply when the
  pulled computed changes (still `reaction`s, not `autorun`s, the reschedule
  reads+writes player observables). If the scheduler ever moves to a worker,
  these become the reaction that postMessages the computed across.

Also moved the whole playback engine from `src/playback/` into
`src/jot_view/playback/` (the jot viewer is its only consumer).

Caveat: audio *fidelity* isn't e2e-verifiable (the suite asserts
scheduling/audibility wiring, which the `audio_tracks` mute/solo-live spec
covers, not sound), worth an ear-check.

### F2. `audioTrack.pitch` (TODO bullet 15), DONE
`AudioTrack.pitch` is now a computed derived from the mixer group: when the
audio row shares a group with instrument row(s) it reports that pitch (the
user's grouping is the source of truth). A private `_pitchOverride` holds the
load-time mapping, used only as the tiebreaker when one file maps to several
pitches in a group, and as the fallback when the row is solo. On drag-OUT
(grouped → solo) `MixerPresenter.moveTrack` calls `detachPitch()` to bake the
group pitch into `_pitchOverride` before the group is gone. The group walk is a
shared free function `groupInstrumentPitches` (tracks.ts), kept a free
function, not a MobX action, so the `trackOrder` read stays tracked inside the
`pitch`/`color` computeds. +8 unit tests.

### F3. `StructuralTrack` vs `RenderedJot` shared source of truth (TODO bullet 16). DONE
Each `Resolved*` layout type now extends its zoom-invariant `Structural*` base
(`ResolvedNote = StructuralNote & { x, width }`, `ResolvedBar = Omit<StructuralBar,
'tracks'|'patternSpans'|'tupletSpans'> & {…}`, etc.), so the beat-only base
fields are declared once. Type-only change (erases at compile).

### F4. `LyricsRowDragProps` re-declared to avoid a circular import. RESOLVED
~~`lyrics/lyrics_row.tsx` re-declares a subset of `MixerRowDragProps`...~~
**Resolved** (commit `refactor(lyrics): dedupe drag props + drop-target hook`).
Once the mixer rows were extracted, `mixer/mixer_drag.tsx` became a leaf
(imports only classnames/react/css), so `lyrics_row.tsx` now imports
`MixerRowDragProps` + `useMixerRowDropTarget` from it directly, no cycle,
and the byte-identical local `useDropTarget` hook + the subset prop type are
gone (−62 lines). Verified perf-neutral by baseline comparison (see perf-gate
note below).

---

## Dedup audit (bullet 26)

Read-only audit of the MobX stores/presenters + `src/playback`/`src/lyrics`/
`src/selection` singletons for state held in two places that can drift.

**Result:** no unsafe, drift-prone duplication found beyond F1-F4 above. The
following were checked and judged SAFE (deliberate, can't drift):
- `PlaybackStore.drumOffsetBeats`, read-through getter over `currentJot`, no copy.
- Follow-playhead flags, isolated UI state, single writer (PlaybackPresenter).
- `jotPlayer.currentFilter` / `currentAudioTrackFilter`, intentional snapshots
  pushed by MixerPresenter for the non-React render loop.
- `ProvenanceStore.provenanceContextValue` + sub-computeds, proper MobX computeds.
- Lyrics tracks vs `mixer.trackOrder`, the latter holds id references only.
- Document→player drum-offset seeding; controlled one-way push, user-adjustable after.
- `jotPlayer.audioTracks` + derived getters, single writer, observers don't mutate.

No safe auto-consolidation was applied (the one real candidate, F1, is
architectural and flagged).

---

## Per-feature extraction progress

| feature | status | notes |
|---|---|---|
| playback | **done** | split `playhead.tsx` out of `playback.tsx`; pure label logic → `playhead_label.ts` (+ unit test) |
| provenance | **done** | `DebugPanel` moved out of `toolbar` → `provenance/debug_panel.{tsx,module.css}` |
| lyrics | **done** | pure beat-positioning → `lyric_layout.ts`. `lyrics_row.tsx` 776 → 342: chips/WordText/WindowedLines + word-debug tooltip → `lyric_chips.tsx`; overflow menu + LRC export → `lyrics_overflow_menu.tsx`; the duplicated drag props + drop-target hook removed (F4 resolved, now imports `mixer/mixer_drag`). LyricsRow keeps the row composition + positioning/shift memos. |
| viewport | not started | `vertical_scrollbar` + store/presenter already small/clean; nothing obvious to extract. |
| transcribe | not started | `recent_transcriptions` is small; toolbar holds the transcribe form (see toolbar). |
| minimap | **done (conservative)** | `minimap.tsx` 532 → 495. The file is densely perf-critical (canvas paint, rAF-coalesced pointer drag, per-frame MinimapViewportBox/MinimapPlayhead observers) and already internally decomposed, so only the genuinely-pure, perf-neutral math was extracted: `computeBarLayouts` (jot-time → minimap px) + `noteMarksEqual` + the NoteMark types → `minimap_layout.ts`, with `minimap_layout.test.ts` (16 cases). These run in `useMemo`/`reaction`, never in the paint, so the paint/pointer/per-frame code is byte-identical. The waveform-peaks effect + canvas paint were deliberately left in place (touching them risks the 120fps budget, which can't be re-measured on the currently-loaded box). |
| mixer | **done** | `mixer.tsx` 1605 → **201 lines** (just the `MixerView` composition root + the control-type re-export), every slice perf-gated (all 26 e2e incl. 3×120fps specs green throughout). Extracted leaf clusters: `overflow_menus.tsx` (audio + instrument overflow menus + split-state, which mixer.test.ts covers), `gutter_controls.tsx` (GutterMasterRow + RowVolumeSlider), `mixer_drag.tsx` (MixerRowDragProps + useMixerRowDropTarget + drop zones + drag handle), `mixer_controls.ts` (VoiceControls/AudioTrackControls types), `use_live_px_per_beat.ts`, `waveform_chunks.ts`, then the two big row components: `instrument_row.tsx` (InstrumentRow + WindowedBarList, 362) and `audio_track_row.tsx` (AudioTrackRow + AudioTrackWaveformCanvas/Chunk, 497). Also extracted `reorderTrackOrder` → `src/tracks.ts` + 7 unit tests (drag-reorder logic). |
| toolbar | **mostly done** | `DebugPanel` extracted earlier. `toolbar.tsx` 1136 → 823: leaf clusters that read stores directly (no prop threading) pulled into `playback_menu.tsx` (kit/speed/latency), `view_menu.tsx` (ZoomControl + ThemeSection), `toolbar_status.tsx` (DrumLoadingIndicator + busy pills + the pure sample-progress/stage-label helpers). Added `toolbar_status.test.ts` (14 cases) for the extracted pure helpers. **Remaining (optional):** the File + Transcribe dropdown bodies are still inline in `Toolbar` (~400 lines of JSX); they're cohesive with the composition root and need heavy prop/ref threading to extract, so left as-is. |
| score | **done** | `score.tsx` 2825 → **123 lines**, split into 6 focused files, every slice perf-gated (E2E_DEBUG_BUNDLE live; all 26 e2e incl. 3×120fps perf specs green): `note_provenance_details.tsx` (debug-details, 1671), `bar_view.tsx` (BarView/NoteView/brackets/grace + note-desc helpers, 599, per-frame hot path, no perf regression), `popover_portal.tsx` (160), `timeline_header.tsx` (+WindowedTicks/TickDescriptor, 226), `filtered_onset_view.tsx` (119). score.tsx is now a 123-line leaf (seekFromClick + title/subtitle helpers + Legend). `WAVEFORM_PAINT_COLOR` → `utils/waveform_color.ts`. No import cycles. |
| contexts.ts split | not started | Splitting each React context next to its feature is low-risk but high import-churn (every context consumer). Cross-cutting contexts (NoteProvenance, BarTimings, RenderedJot, GridLineSettings, UniformWaveforms, FollowPlayhead, Selection) have no single feature home. Deferred; recommend a dedicated mechanical pass.

**Summary:** the safe behaviour-preserving extractions were done (playback,
provenance/DebugPanel, lyrics layout). `score.tsx` was fully broken up
(2825 → 123 lines across 6 files). `mixer.tsx` is now fully broken up too
(1605 → 201 lines: just `MixerView` + the control-type re-export), across
the leaf clusters (overflow menus, gutter controls, drag primitives, control
types, live-px hook, waveform chunks) plus the two big row components
(`instrument_row.tsx`, `audio_track_row.tsx`); `reorderTrackOrder` lives in
`src/tracks.ts` with 7 unit tests. Every slice verified against the live perf
gate (E2E_DEBUG_BUNDLE) including the 3×120fps specs. Remaining flagged work:
the `contexts.ts` split (low-risk but high import-churn), and the smaller
optional splits noted above (lyrics_row, toolbar menus, minimap helpers).

---

## Unit-test cleanup (bullet 24)

Approach: as each extraction lands, the now-pure logic gets a focused unit
test (no React mocking needed; the whole point of pulling logic into
presenters/helpers). Added so far:
- `playback/playhead_label.test.ts` (earlier)
- `tracks.test.ts` `reorderTrackOrder` (7 cases, drag-reorder logic)
- `toolbar/toolbar_status.test.ts` (14 cases: sampleProgressWidth/Label,
  samplePct, formatMb, formatStageLabel)
- `minimap/minimap_layout.test.ts` (16 cases: computeBarLayouts, noteMarksEqual)
- `lyrics/lyric_layout.test.ts` (10 cases: positionLyricLines)

Suite grew 282 → 324 pass, green throughout.

**Existing-test review (the "simplify / remove roundabout tests" half):**
walked all `src/**/*.test.ts`. The core-domain suites (parser, midi, rlrr,
lyrics, linter) are legitimate behaviour tests untouched by the React
refactor; the `jot_view` tests are all right-sized and focused
(`mixer.test.ts` is a clean role to enable matrix, correctly repointed to
`overflow_menus.ts` during the extraction). No roundabout or now-duplicated
tests were found, so nothing was removed or rewritten; the cleanup here was
additive (test the freshly-extracted pure logic) rather than subtractive.

Still-untested pure target if you want more: `lyrics_measure.ts`
(`computeLyricShifts`), but it measures glyphs via an off-screen canvas, so
it needs a DOM/canvas (bun's test env has none); better covered by an e2e or
a jsdom-canvas shim than a plain unit test.

---

## Storybook (done, foundation)

Storybook 9 (`@storybook/react-vite`) installed + configured
(`.storybook/main.ts` re-applies `patchCssModules()` + the `src` alias;
`preview.ts` loads the design tokens). Scripts: `bun run storybook` (dev),
`bun run build-storybook` (static build, green). Stories live in per-feature
`stories/` subfolders.

Coverage so far (all three requested categories represented):
- **Primitives** (`components/stories/`): IconButton (+Mute/Solo/Clear),
  Checkbox, NumberStepper, Tabs, Logo, ColorPot, ColorPicker, variants +
  interactive, handlers routed to the Actions panel via `fn()`.
- **Major components**:
  - `playback/stories/`: PlaybackBar driven by a real DocumentStore +
    PlaybackStore + PlaybackPresenter trio (with-jot / no-jot).
  - `mixer/stories/`: **InstrumentRow** (the note-track row) driven by real
    Document/Mixer/Viewport stores (trackOrder seeded from `jotPitches`,
    voiceControls stubbed), Default + Muted. Verified rendering in headless
    Chromium off the static build, gutter + bar noteheads paint.
  - `toolbar/stories/`: the busy pills (LyricsAlignBusyPill,
    TranscribeBusyPill) + the ThemeSection picker.
- **Library sandbox** (`src/stories/jot_loader.stories.tsx`): pick a
  `.jot`/`.mid` (or built-in example) → `parse()`/`fromMidi()` → view the Jot
  as text + a live JotView.

Remaining (not done):
- **AudioTrackRow**, needs a decoded `AudioBuffer` + the waveform worker to
  show its waveform; the gutter chrome alone would story, but a faithful one
  needs an audio fixture. Recipe is the same as InstrumentRow otherwise
  (real stores + stubbed AudioTrackControls).
- **Full mixer / MixerView**, would reuse the InstrumentRow harness plus an
  audio + lyrics row; mechanical follow-up now that the rows are exported.
- **Toolbar File/Transcribe dropdowns**, still inline in `Toolbar` (not
  separately exported), so no isolated story yet.

---

## Broader src/ review (bullet 25, non-jot_view libraries)

Reviewed all of `src/` outside `jot_view` (parser, dsl, jot, midi, rlrr,
playback engine, lyrics, tempo, selection, etc.). Verdict: generally
well-factored. Large files are large because their domain is genuinely complex
(player.ts, jot.ts, the converters), not because concerns are conflated. The
parser/formatter pipeline, MIDI/RLRR round-trip pairs, tempo math (centralised
in `tempo.ts`), drum mappings (gm/drums/instruments are distinct domains, not
dupes), and the lyrics module are all clean. No edits made (review-only; the
candidates below carry behaviour risk).

### F5. Velocity/dynamics constants duplicated across converters. FIXED (was a bug)
Resolved: extracted one shared `src/dynamics.ts` (DEFAULT_VELOCITY,
ACCENT_BOOST, GHOST_REDUCTION, VOLUME_TO_VELOCITY + the from_midi
ACCENT/GHOST thresholds); playback / to_midi / from_midi / jot_to_rlrr all
import it. Canonical `accentBoost = 36`: an accent must clear the loudest
non-accent volume (ff = 96) so `from_midi` (threshold 100) can recover it on
import, `mf:a` = 64 + 36 = 100 round-trips, 24 → 88 < 96 does not. So 24 was
the buggy value; playback + RLRR accents are now +36 to match MIDI export.
Original analysis below.

The accent/ghost/default velocity mapping is redefined in three places:
- `src/playback/events.ts` (~L35-45): DEFAULT_VELOCITY / ACCENT_BOOST / GHOST_REDUCTION / VOLUME_TO_VELOCITY
- `src/midi/to_midi.ts` (~L58-67): `DEFAULTS` with **accentBoost: 36**
- `src/rlrr/jot_to_rlrr.ts` (~L62-76): `DEFAULTS` with **accentBoost: 24**

Playback (`events.ts`) and RLRR export use accentBoost 24; MIDI export
(`to_midi.ts`) uses 36 (its comment says 36 was chosen for MIDI round-trip
fidelity). So an accent plays back at +24 but exports to MIDI at +36, i.e. a
play to export to re-import round-trip changes velocity. This MAY be
intentional (playback feel vs. export fidelity), which is exactly why I did
NOT unify them: extracting to one shared `dynamics.ts` forces a single value
and would change behaviour for at least one consumer. **Action for you:**
decide whether the 24/36 split is intentional; if yes, document it at both
sites; if no, pick the canonical value, consolidate, and re-check the MIDI
round-trip tests. Until then it's a real drift risk.

### F6. `src/jot.ts` (~1490 lines), splittable, DONE
Split into `view_config.ts` (Pixels brand + ViewConfig), `resolved_jot.ts`
(the Structural*/Resolved* types + the RenderedJot layout engine + drum-offset
pass), and `pattern_expansion.ts` (pattern/repeat tree-rewriting + element
weight / straightness / type-guards), all re-exported from a 12-line `jot.ts`
barrel so every `from 'src/jot'` importer is unchanged. `sumWeights` is now
exported (resolved_jot uses it). The pre-existing jot↔playback/timeline cycle
is preserved (type-only / lazy). Done by codemod (byte-for-byte) + full gate.
