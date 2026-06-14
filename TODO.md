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
- fix audioTrack.pitch - it shouldn't exist, it should be computed from the track group instead?
- refactor StructuralTrack type and co to share same source of truth as RenderedJot (or vice versa), using type unions or Omit
- refactor sweep all code to split out more, be less monolithic:
  (progress + flagged items: docs/refactor-sweep-notes.md)
  - [DONE] split out files even more by feature/domain/etc., per-concern data stores + per-domain presenters (settings/viewport/mixer/playback/provenance/lyrics/document/transcribe).
  - [DONE] if each feature has multiple files, group them into a folder (store + presenter + tsx). All jot_view features are now `src/jot_view/<feature>/`.
    - [DONE] move `DebugPanel` out of `toolbar.tsx` into the `provenance/` feature.
    - [DONE] break up the large `score.tsx`: 2825 to 123 lines, split into 6 files (note_provenance_details, bar_view, popover_portal, timeline_header, filtered_onset_view + the score.tsx leaf), each verified against the perf e2e gate (E2E_DEBUG_BUNDLE; 120fps specs green).
    - [TODO] split the central `contexts.ts` so each React context lives next to its feature's store/presenter. Low-risk but high import-churn; deferred.
  - [PARTIAL] pull out business logic / pure logic into a presenter/util file; keep .tsx scoped to instantiation + React + wiring. Done: playback (playhead + label logic), lyrics (beat-positioning), score (2825→123, 6 files), mixer (1605→201: just MixerView; rows + leaves split into overflow_menus/gutter_controls/mixer_drag/mixer_controls/instrument_row/audio_track_row/waveform_chunks/use_live_px_per_beat, + reorderTrackOrder→tracks.ts w/ tests). Remaining: toolbar (leaf pieces), minimap (canvas helpers). All perf-gated.
  - [PARTIAL] pull out component state into stores that the presenter acts over (only *persistable* state; transient UI state stays React-local). Domain state already lives in stores from the store carve-up.
  - react rendering is now based on the component store state
  - [PARTIAL] once logic has been moved into presenter files, unit tests can directly test logic without mocking React. (Added playhead_label.test.ts; dedicated test-simplification pass not done.)
  - [DONE, foundation] mock data + render components in isolation: Storybook 9 set up with stories for primitives, a major component (PlaybackBar w/ real store+presenter), and a jot-loader library sandbox. More feature-component stories pending the score/mixer splits.
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

## Lyrics
 - swap out non-en model for something commercial license friendly
 - do line-level language detection
 - ruby text for japanese
 - add support for import/export of word aligned lyrics with durations (WebVTT or Extended LRC)
