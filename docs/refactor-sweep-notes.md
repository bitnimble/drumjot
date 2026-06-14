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
| minimap | not started | `minimap.tsx` (~530) has pure peak/tick canvas-prep logic that could move to a tested helper. Perf-adjacent (canvas paint), review before splitting. |
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
