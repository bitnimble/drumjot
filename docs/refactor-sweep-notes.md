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
| playback | **done** | split `playhead.tsx` out of `playback.tsx`; pure label logic → `playhead_label.ts` (+ unit test) |
| provenance | **done** | `DebugPanel` moved out of `toolbar` → `provenance/debug_panel.{tsx,module.css}` |
| lyrics | partial | pure beat-positioning → `lyric_layout.ts`. The big `lyrics_row.tsx` (~770 lines) still bundles several components (chips, WordText, WindowedLines, useDropTarget) that could be split. |
| viewport | not started | `vertical_scrollbar` + store/presenter already small/clean; nothing obvious to extract. |
| transcribe | not started | `recent_transcriptions` is small; toolbar holds the transcribe form (see toolbar). |
| minimap | not started | `minimap.tsx` (~530) has pure peak/tick canvas-prep logic that could move to a tested helper. Perf-adjacent (canvas paint), review before splitting. |
| mixer | **FLAGGED** | `mixer.tsx` (~1600), InstrumentRow / AudioTrackRow / waveform canvas / gutter rows. Perf-critical (per-frame waveform + windowed rows). Mechanical sub-component splits are safe in principle but high-blast-radius; recommend doing as its own reviewed slice. |
| toolbar | partial | `DebugPanel` extracted. The menu code (~1130 lines) remains; could split leaf pieces (busy pills, ThemeSection, PlaybackKitSubmenu, sample-progress helpers), lower risk, not yet done. |
| score | **mostly done** | `score.tsx` 2825 → **585 lines**, split into 3 perf-gated files (E2E_DEBUG_BUNDLE live; all 26 e2e incl. 3×120fps perf specs pass after each slice): (1) `score/note_provenance_details.tsx`, per-note debug-details cluster (1692 lines); (2) `score/bar_view.tsx`, bars-row render (BarView/NoteView/brackets/grace + note-desc helpers, 611 lines), the per-frame hot path, verified no perf regression. `WAVEFORM_PAINT_COLOR` → `utils/waveform_color.ts`; `PopoverPortal` exported from score (shared by NoteView + FilteredOnsetView), bars-row consumer is the mixer's InstrumentRow (repointed). Remaining in score.tsx (585 lines, no longer monolithic; optional further splits): seekFromClick, PopoverPortal, title/subtitle helpers, Legend, TimelineHeader (+WindowedTicks/TickDescriptor), FilteredOnsetView. |
| contexts.ts split | not started | Splitting each React context next to its feature is low-risk but high import-churn (every context consumer). Cross-cutting contexts (NoteProvenance, BarTimings, RenderedJot, GridLineSettings, UniformWaveforms, FollowPlayhead, Selection) have no single feature home. Deferred; recommend a dedicated mechanical pass.

**Summary:** the safe behaviour-preserving extractions were done (playback,
provenance/DebugPanel, lyrics layout). `score.tsx` was then halved (2825 →
1172) by extracting its debug-details cluster, verified against the now-live
perf gate (E2E_DEBUG_BUNDLE). Still flagged for a reviewed pass: `mixer.tsx`,
the remaining perf-critical score sub-components (BarView/NoteView/TimelineHeader/
PopoverPortal), and the contexts split, all mechanically splittable, each a
careful slice best done with the perf specs enabled.

---

## Unit-test cleanup (bullet 24)

Not started as a dedicated pass. Added one new focused unit test alongside an
extraction (`playback/playhead_label.test.ts`). The existing suite (282 pass)
stays green throughout. Good future targets now that logic is extracted:
`lyric_layout.ts` (`positionLyricLines`) and the toolbar sample-progress
helpers.

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
- **Major component** (`playback/stories/`): PlaybackBar driven by a real
  DocumentStore + PlaybackStore + PlaybackPresenter trio (with-jot / no-jot).
- **Library sandbox** (`src/stories/jot_loader.stories.tsx`): pick a
  `.jot`/`.mid` (or built-in example) → `parse()`/`fromMidi()` → view the Jot
  as text + a live JotView.

Remaining (not done): stories for the bigger feature components (audio-track
row, instrument/note row, toolbar menus, full mixer), these depend on the
`mixer.tsx`/`score.tsx` sub-components being exported, which ties into the
flagged breakups above.

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

### F5. Velocity/dynamics constants duplicated across converters (POSSIBLE BUG)
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

### F6. `src/jot.ts` (~1490 lines), splittable, low priority
Three separable responsibilities: `ViewConfig` (layout config), the resolved
data model (`ResolvedNote`/`ResolvedTrack`/`RenderedJot` structure), and the
pattern-expansion tree-rewriting. Could split into `view_config.ts` /
`resolved_jot.ts` / `pattern_expansion.ts` re-exported from `jot.ts`. Pure
mechanical move, low risk if import contracts are preserved, but it's a
load-bearing core file touched by nearly everything, so worth doing
deliberately with the full suite rather than blind. Not done.
