## DevX
- add Prettier and autoformat hook

## Bugs

## UX
- mobile friendly
- pending tasks dropdown in top right - any active backend tasks (stem splitting, transcription, lyrics alignment, etc). see if we can add ETAs or progress bars.
- make a settings schema, move all persistent user settings into it, have it saved to localstorage

## Performance
- Add test map + regression tests

## Refactoring
- [DONE] fix audioTrack.pitch - now a computed derived from the mixer group (own `_pitchOverride` only as tiebreaker/solo-fallback; detach bakes in on drag-out). Shared `groupInstrumentPitches` helper in tracks.ts.
- [DONE] StructuralTrack type and co now share one source of truth with RenderedJot: each `Resolved*` extends its `Structural*` via `&` / `Omit` (no duplicated base fields).
- refactor sweep all code to split out more, be less monolithic:
  (progress + flagged items: docs/refactor-sweep-notes.md)
  - [DONE] split out files even more by feature/domain/etc., per-concern data stores + per-domain presenters (settings/viewport/mixer/playback/provenance/lyrics/document/transcribe).
  - [DONE] if each feature has multiple files, group them into a folder (store + presenter + tsx). All jot_view features are now `src/jot_view/<feature>/`.
    - [DONE] move `DebugPanel` out of `toolbar.tsx` into the `provenance/` feature.
    - [DONE] break up the large `score.tsx`: 2825 to 123 lines, split into 6 files (note_provenance_details, bar_view, popover_portal, timeline_header, filtered_onset_view + the score.tsx leaf), each verified against the perf e2e gate (E2E_DEBUG_BUNDLE; 120fps specs green).
    - [DONE] split the central `contexts.ts` and deleted it: every React context now lives next to its type/store. Feature-routing → `<feature>/<feature>_contexts.ts` (mixer/viewport/lyrics/provenance/playback); SelectionContext → `src/selection`; RenderedJot+BarTimings → `document/document_contexts`; GridLineSettings → `settings/settings_contexts`; UniformWaveforms → `mixer/mixer_contexts`. All consumers repointed.
  - [PARTIAL] pull out business logic / pure logic into a presenter/util file; keep .tsx scoped to instantiation + React + wiring. Done: playback (playhead + label logic), lyrics (beat-positioning), score (2825→123, 6 files), mixer (1605→201: just MixerView; rows + leaves split into overflow_menus/gutter_controls/mixer_drag/mixer_controls/instrument_row/audio_track_row/waveform_chunks/use_live_px_per_beat, + reorderTrackOrder→tracks.ts w/ tests). Remaining: toolbar (leaf pieces), minimap (canvas helpers). All perf-gated.
  - [PARTIAL] pull out component state into stores that the presenter acts over (only *persistable* state; transient UI state stays React-local). Domain state already lives in stores from the store carve-up.
  - react rendering is now based on the component store state
  - [PARTIAL] once logic has been moved into presenter files, unit tests can directly test logic without mocking React. (Added playhead_label.test.ts; dedicated test-simplification pass not done.)
  - [DONE] mock data + render components in isolation: Storybook 9 set up. Stories: all primitives; major components, PlaybackBar (real store+presenter), InstrumentRow / note-track row (real Document/Mixer/Viewport stores, verified rendering in headless Chromium), toolbar busy pills + ThemeSection; jot-loader library sandbox. Remaining (optional, noted in refactor-sweep-notes.md): AudioTrackRow (needs an audio-buffer fixture), full MixerView, File/Transcribe dropdowns.
  - [PARTIAL] move helper files, functions, and classes into a utils/ directory. `windowing` → `jot_view/utils/`; feature-specific helpers placed with their primary consumer per agreed rule.
  - [DONE] explore all the code for unnecessary duplication, esp. state that can drift out of sync. Audited; findings + one architectural flag (section-audibility mirror) in docs/refactor-sweep-notes.md. No unsafe drift-prone duplication found beyond pre-listed items.
- prepare for editing support. review all presenters and stores for any state management issues that may arise from editing.


## Converters
- redo midi/rlrr -> jot conversion, using smart stuff, e.g. tuplet inference
- add Export to the web ui
- test bpm changes in a song (also test round trip)

## Transcription
- fix hihats:
  - stage 1: improve open/closed classification
  - stage 2: splash
  - use energy injection filter from crashes to improve open/closed classification
- fix snares:
  - stage 1: correctly do all onsets at high accuracy (deterministic, need to adjust thresholds)
  - stage 2: modifiers and grace notes (needs inference): roll, flam
- fix cymbals:
  - stage 1: improve crash/ride classification

## Training (learned drum-onset model)
Architecture is locked to frozen MERT + per-frame per-lane heads. MuQ and a
two-stage propose→classify arch were both evaluated and ruled out (see
training/RESULTS.md). The open problem is the cymbal ceiling (crash/ride/misc),
which is data/separation-bound, not architecture-bound.
- [done 2026-06-14] `cym` sub-6 kHz timbre block A/B (`--cym`): **no benefit** at
  natural dist AND on a cymbal-balanced set (crash 0.000 even with 4x crash; ride
  −0.044) → **removed entirely** (feature + flag + cym_features). See RESULTS.md.
- [done 2026-06-14] dropped-neg A/B (`--dropped-neg` vs `--no-dropped-neg`): **no
  precision gain, mild ride/crash F1 loss → removed entirely** (separation already
  strips aux perc; right idea, wrong stage). See RESULTS.md / CHANGELOG #6.
- [done 2026-06-14] per-lane `keep_best` re-baseline: **validated**, over old
  global-best it adds crash +0.089 / mc +0.062 / hp +0.025 (lanes peak off the
  macro); huge over final-epoch on overfitters. Kept as default.
- better cymbal SEPARATION is the real ceiling lever, NOT more crash data: a
  cymbal-balanced A/B (crash 1:12→1:3 vs ride) left baseline crash F1 flat (0.645
  vs 0.657), so crash F1 is separation/ambiguity-bound, not crash-volume-starved.
- higher-cap confirmation of the per-stem best layers (the sweep was cap-30: s→L1,
  c→L10).
- (bigger, only if the above stall) fine-tune the MERT encoder.

## Lyrics
 - swap out non-en model for something commercial license friendly
 - do line-level language detection
 - ruby text for japanese
 - add support for import/export of word aligned lyrics with durations (WebVTT or Extended LRC)
