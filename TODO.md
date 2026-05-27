## DevX
- add Prettier and autoformat hook
- use branches, and write a skill for merging 

## UX
- Beat and audio alignment controls: stack them vertically on top of each other, keeping them in the right hand side of the bottom playback bar
- pending tasks dropdown in top right - any active backend tasks (stem splitting, transcription, lyrics alignment, etc). see if we can add ETAs or progress bars.
- add colour picker to track overflow menus. changes colour of notes for instrument tracks, and colour of waveforms for audio tracks

## Performance
- Worker pool for multi-threaded waveform compute and render on audio tracks

## Refactoring
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
- fix snares:
  - stage 1: correctly do all onsets at high accuracy (deterministic, need to adjust thresholds)
  - stage 2: modifiers and grace notes (needs inference): roll, flam
- fix cymbals:
  - stage 1: improve crash/ride classification

## Lyrics
 - test multi track
 - swap out non-en model for something commercial license friendly
 - do line-level language detection
