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
  - split out files even more by feature/domain/etc. for example, viewport (consisting of the bounds of the viewport, converters between coordinate systems like pixels to beats, DPI awareness, etc) should probably be its own store and presenter.
  - if each feature has multiple files, group them into a folder. it's okay if each folder only has three files (store + presenter + tsx).
  - pull out business logic / pure logic stuff into a foo_presenter.ts file. Keep the *.tsx file scoped to (a) instantiation (b) React stuff (hooks, events, callbacks, VDOM+JSX) and (c) wiring between stores, presenters, and React.
  - pull out component state into stores that the presenter acts over
  - react rendering is now based on the component store state
  - once logic has been moved into presenter files, unit tests can now directly test logic without needing to mock React components.
  - with component state moved out into stores, we can also now mock data and render the component with less depenedencies required, e.g. in a Storybook or visual diff (don't actually implement Storybook or visual diffing, that's just an example)
  - move helper files, functions, and classes into a utils/ directory
  - explore all the code to see if there's anything that's duplicated unnecessarily, especially focusing on data structures that may get out of sync. if something is duplicated it is not necessarily bad - it's only bad if they can "get out of sync". i.e. if one implementation changes but not the other, would it lead to a product bug or regression? if so, they should share the same source of truth or implementation.
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
- [in progress] `cym` sub-6 kHz timbre block A/B (`--cym`): does the ride-ping-vs-
  crash-wash block lift cymbal F1? If yes, wire it into inference (currently raises
  NotImplementedError for cym).
- dropped-neg A/B (`--dropped-neg` vs `--no-dropped-neg`): confirm the hard-negative
  `x` lane actually improves precision on the leak-prone lanes (hc/rd/cr/mc); revert
  if it doesn't pull its weight.
- more crash / misc-cym training data + better cymbal separation, the fundamental
  ceiling levers (crash is data-starved vs ride; mc has ~50 val onsets).
- re-baseline with the new per-lane `keep_best`: quantify the per-lane-F1 lift over
  the old global-best-epoch behaviour.
- higher-cap confirmation of the per-stem best layers (the sweep was cap-30: s→L1,
  c→L10).
- (bigger, only if the above stall) fine-tune the MERT encoder.

## Lyrics
 - swap out non-en model for something commercial license friendly
 - do line-level language detection
 - ruby text for japanese
 - add support for import/export of word aligned lyrics with durations (WebVTT or Extended LRC)
