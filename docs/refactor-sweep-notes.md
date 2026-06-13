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

---

## Flagged for review (NOT changed autonomously)

### F1. Section-audibility state mirrored into `jotPlayer` (architectural)
`MixerStore.isAudioSectionAudible` / `isDrumSectionAudible` (computeds over the
authoritative mute/solo state) are pushed into `jotPlayer.audioMasterAudible` /
`drumMasterAudible` via two `reaction`s in `MixerPresenter` (mixer_presenter.ts
~L71-82). The player fields are 100% derivable from the mixer store, so they
*could* become computeds reading the mixer; eliminating the reaction sync.

**Why flagged, not changed:** this inverts the dependency direction (today the
player is deliberately decoupled from the mixer and the mixer *pushes* filters
to it; see also `currentFilter`/`currentAudioTrackFilter` snapshots, same
pattern). Making the player read the mixer is an architectural change with
real behaviour/perf surface (the player is the per-frame audio path). The
current reaction-push is a deliberate decoupling, not an accident. Leaving for
your call. Low real-world drift risk today since the reactions are the sole
writers and `fireImmediately` seeds them.

### F2. `audioTrack.pitch` (already in TODO.md, bullet 15)
Confirmed still present, `AudioTrack.pitch` duplicates info derivable from the
track's mixer group. Pre-existing TODO; not touched.

### F3. `StructuralTrack` vs `RenderedJot` shared source of truth (TODO bullet 16)
Pre-existing TODO; not touched.

### F4. `LyricsRowDragProps` re-declared to avoid a circular import
`lyrics/lyrics_row.tsx` re-declares a subset of `MixerRowDragProps` from
`mixer/mixer.tsx` with a comment noting it's to dodge a circular import. Minor
drift risk (the two prop shapes must stay compatible). Could be resolved by
hoisting the shared drag-props type into a small shared module (e.g.
`jot_view/utils/` or a `mixer/row_drag.ts`). Low priority; flagged.

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
| playback | pending | |
| viewport | pending | |
| transcribe | pending | |
| lyrics | pending | |
| minimap | pending | |
| mixer | pending | |
| toolbar | pending | incl. DebugPanel → provenance |
| score | pending | big; split sub-components |
| contexts.ts split | pending | per-feature contexts |

---

## Unit-test cleanup

_(pending)_

---

## Storybook

_(pending)_
