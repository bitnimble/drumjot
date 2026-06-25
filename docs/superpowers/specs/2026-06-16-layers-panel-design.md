# Layers panel + first-class layers, design spec

Status: approved (brainstorm 2026-06-16), implementing autonomously.

## Goal

Make **layers first-class** in the score and add a right-sidebar **Layers
panel** that mirrors the score's track layout from one shared source of truth.
Today the score is *lane*-oriented: one row per lane, with notes from every
`||` layer merged into it (`mergedTrackFor`). We **invert** that, the default
becomes layer-segregated (all of layer 1's tracks, then layer 2's, …), and the
old merged look becomes an opt-in **"Visually merge layers"** View toggle.

## Terminology

lane = instrument (snare, kick…) · layer = a `||` voice · track = one rendered
row · group = a styled cluster of tracks within a layer (e.g. a cymbal audio
waveform + the crash & ride tracks).

## Requirements (from the user)

- Panel shows all tracks, grouped by layer.
- Drag-drop reorder tracks, incl. moving a track to another layer.
- The **same lane may exist in multiple layers** (layer 1 snare + layer 2
  snare = two tracks). A **single layer may NOT hold two tracks of the same
  lane** (DSL constraint).
- Tracks in a layer can be grouped (audio + instrument together); groups show a
  heading + indentation; tracks move in/out of groups via the panel, same as
  dragging trackheads in the gutter.
- Panel and score never disagree, one source of truth.
- The score is updated to render layers first-class.
- Whole layers reorder (layer 2 before layer 1); whole **groups** reorder too.
- Layers have colors → a muted/transparent background band behind the layer +
  its tracks in the score. **Layer 1 defaults to transparent**; layers 2+ get a
  `PICKER_PALETTE`-rotation color at creation, user-overridable (incl. back to
  none).
- **View-only** "Visually merge layers" (off by default): collapse matching
  lanes to single rows. No data change. Edits route per-note by each note's own
  track; a click-to-add lands on the firstmost layer that has a track in the
  merged row; moving a note across rows = delete on the source track + add on
  the destination.

## Data model (schema additions, all in `JotSchema` / Loro doc)

No back-compat: the schema is never saved to disk today, so this is a clean new
format. `tracks`, `layers`, `trackGroups`, `ordering` are **non-optional**;
whoever constructs a Jot builds them (helper below).

### Tracks are first-class entities

```ts
tracks: idMap(TrackSchema)
TrackSchema =
  | { id, kind:'instrument', lane }     // lane = the instrument; moved OFF notes
  | { id, kind:'audio', audioId }
  | { id, kind:'lyrics', lyricsId }
```

- **`note.trackId` replaces `note.layerId`.** A note → its track → lane, and
  (via `ordering` reverse-lookup) its group + layer. Moving a track across
  layers needs **no note re-homing**, notes keep their `trackId`; only
  `ordering` changes.
- **`lane` moves onto the instrument track.** `note.lane` is removed; replace
  reads with **`laneForNote(note)`**, a `computedFn` doing `trackId → track →
  lane`. (A later augmented `ReactiveJot` will fold such helpers onto the schema;
  not now.) `jot.instruments` stays keyed by lane (both snares share the snare
  instrument). The derived `StructNote` still carries `lane` for rendering.
- **Group/pattern container elements carry no layer/track ref.** Their layer is
  derived from their child notes' tracks (all children share a layer by
  construction). `layerId` is removed from the schema entirely.

### Layers, groups, ordering

```ts
layers:      idMap({ id, name?, color? })      // color? = '#rrggbb' or absent (transparent)
trackGroups: idMap({ id, name, color? })       // group heading text + optional tint
ordering = {
  layers: [                                    // layer order, top→bottom in score
    { layerId,
      slots: [                                 // groups + loose runs, in order
        { groupId: 'g1', tracks: [trackId, …] },   // a named group
        { groupId: null, tracks: [trackId, …] },   // loose (ungrouped) run
      ] } ] }
```

- Multiple `groupId:null` runs allowed (loose tracks interleave around groups).
- Reverse-lookup computeds: `trackId → layerId`, `trackId → groupId`. The
  structure store keeps bucketing per `(bar, layer)` for granular perf; a note's
  layer now comes via `trackId → track`'s placement (containers via first child).
- **`buildDefaultOrdering(tracks)`** helper: lays out a sane arrangement (one
  layer, default mixer kind-order, audio+lane pairing like
  `buildDebugBundleTrackOrder` does today). Converters (ParaDB / MIDI / RLRR /
  transcribe / debug-bundle) call it. Mid-session newcomers (transcribe adds a
  lane, a stem loads) = explicit presenter write (create track + splice into
  `ordering`), with a defensive read-time guard tolerating transient dangling
  refs.

## Rendering (score) + mute/solo

- Score iterates `ordering`: per layer → a **color band** background behind its
  rows (spanning score width, behind notes, per `docs/score-stacking.md`) → per
  slot → named group draws a heading + indent (with a drag handle) → each track
  draws its row keyed by **`trackId`** (was per-lane). `mergedTrackFor` survives
  only for the merge view.
- **Per-track mute/solo/volume**, keyed by `trackId`: `mutedTracks` /
  `soloedTracks: Set<trackId>`, `trackVolumes: Map<trackId, number>`. The
  **playback filter becomes per-track** (note → `trackId` → audible?). In merged
  view a collapsed row aggregates its tracks (mute = mute each underlying track).
- Gutter trackhead drag writes `ordering` (replacing the old
  `MixerStore.trackOrder`); the panel writes the **same** `ordering` via the
  same presenter methods.

## Layers panel (right sidebar)

- Lives in the existing sidebar stub (`src/sidebar/panels/layers_panel.tsx`).
- **Structural only**, arrange + rename + color + group. No M/S/volume (those
  stay in the gutter). A mute toggle could be added later.
- Mirrors the score: layer header (drag handle + color swatch + name + ⋯), its
  groups (drag handle + heading + ⋯, indented) and loose tracks, each track row
  (drag handle + swatch + name + kind tag).
- **Drag-drop:** reorder tracks within a layer, move a track to another layer,
  pull a track in/out of a group, reorder whole layers (drag the layer header),
  reorder whole groups (drag the group handle). Drop indicators.
- **⋯ menus:** layer = rename / pick color (incl. none) / delete layer / new
  group. group = rename / pick color / ungroup. New layers default name
  "Layer N" + palette-rotation color; new groups default "Group N".

## "Visually merge layers", View menu toggle (off by default)

- View-only state (no schema change). In `view_menu.tsx`, a `ToggleMenuItem`.
- ON: render **flat per-lane rows, no layer bands**. Collapse **all** same-lane
  tracks across every layer into one row. **Groups still shown where possible:**
  merge other layers' matching lanes into the **topmost layer's** groups.
- Editing routing on a merged row:
  - existing note → edits its own track (each note knows its track → layer);
  - click-to-add → firstmost layer that has a track in the merged row;
  - move a note to a different lane's row → delete on source track + add on a
    track of the destination lane (firstmost layer with it).

## Architecture placement (store/presenter rules)

- Schema entities live in the Loro doc (data). A new domain owns the
  ordering/layers/groups read-model + reverse-lookups + the merge-view derivation, a `LayersStore` (data: observables/computeds + reverse-lookup computeds) and
  `LayersPresenter` (the single writer: reorder/move/group/ungroup/rename/color,
  create/delete layer & group, all `ordering` + `tracks` mutations). The sidebar
  panel + score gutter both call the presenter. Mixer mute/solo/volume re-key to
  `trackId` (MixerStore data, MixerPresenter writes). Keep the three-layer split;
  no barrels; no DOM layout reads in hot paths; 120fps budget.

## Implementation approach: expand-contract

The change surface is large (70+ `.lane` sites, the structure store, mixer,
playback, converters). To keep every commit green on a live codebase, the new
model is added **alongside** the old one (expand), consumers migrate phase by
phase, and `lane`/`layerId` are removed **last** (contract). Same end state.
Crucially, almost all `.lane` reads are of the *derived* `StructNote.lane`
(kept, populated via the track) rather than the raw schema note, so the true
raw-read surface is small. All Jot construction funnels through `dslToInit`
(parse / MIDI / RLRR / fakes), so converter work is one function.

## Phasing (each phase keeps build + unit + e2e green, committed separately)

1. ✅ **Schema EXPAND.** `tracks`/`trackGroups`/`ordering` + layer `color` +
   note `trackId` added (lane/layerId kept); `ordering.ts` helpers
   (`laneForNote`, `trackLaneOf`, reverse-lookups, `TrackBuilder`); `dslToInit`
   seeds tracks + default loose ordering. No behaviour change. *(commit
   b487281)*
2. ✅ **Reactive read-model.** `LayersStore` (`src/editing/layers/`): the
   layer → slot(group|loose) → track `layout` + memoised reverse-lookups +
   default band colours (layer 1 transparent, rest palette). *(commits "reactive
   LayersStore", "default layer band colours")*
3. ✅ **Read-only Layers panel.** Wired into the composition root + a context;
   the sidebar panel renders the tree (band tint, group heading/indent, track
   labels). View-only mirror. *(commit "read-only Layers panel")*

4. ✅ **Inversion foundations (no score change yet).**
   `StructuralPresenter.barsForTrack(layerId, lane)`, per-track (not merged)
   render data, plus `LayersPresenter`, the single writer (layer colour/name,
   group name/colour, `reorderLayer`, `moveTrack` across/within layers with
   group-join + empty-slot prune), unit-tested against a live jot so the
   nested-Loro mutation path is verified. *(commits b82f5eb, 3fe3163)*
5. ✅ **Interactive panel: rename + band colour** via a per-layer ⋯ menu
   (reusing the dropdown + colour-picker primitives) writing through the
   presenter; read-model exposes `hasColorOverride`. *(commit b6cfa19)*

6. ✅ **Score renders layer-first** (rendering inversion). The score's
   instrument rows now come from `LayersStore.layout`: one tinted band per layer
   (transparent layer 1, palette default for the rest), holding its groups
   (heading + indent) and per-track rows keyed by `trackId` via `barsForTrack`,
   so the same lane in two layers shows two independent rows. Default ordering
   uses the familiar mixer kind-order (helpers moved to a neutral
   `instruments/mixer_order` module shared by the mixer + the DSL converter).
   Verified with browser screenshots (single-layer unchanged; two-layer bands +
   two-snare rows; lead-in caption). *(commit 08bd29b)*

7. ✅ **Per-track mute/solo/volume + playback filter.** Re-keyed to a
   `layerId/lane` track key; `PlaybackEvent` carries `layerId`; muting one
   layer's snare leaves the other's audible. *(commit e8b7391)*
8. ✅ **Panel drag-drop.** Native HTML5 DnD: reorder a track, move it across
   layers, in/out of a group, reorder whole layers; the score reflects every
   move (both read `ordering`). *(commit 299da83)*
9. ✅ **Groups.** Create (per-track button) / ungroup / rename / colour
   (group ⋯ menu); tracks join via DnD; **whole groups reorder** via the group
   header handle. *(commits 121ae0d, c02994d)*
10. ✅ **"Visually merge layers"** View toggle (off by default) + per-note edit
    routing. Collapses same-lane tracks across layers to one flat row (no bands),
    keeping the topmost layer's groups; the merged row aggregates mute/solo/volume
    across its layers and shows the union of notes. Insert lands on the clicked
    row's layer (per-track) or the firstmost layer (merged); also fixed a
    phase-6 bug where inserting on a non-first layer's row went to layer 1.
    *(commit cddb550)*

11. ✅ **Audio/lyrics tracks into `ordering`** (core). Audio + lyrics are now
    first-class `tracks` placed in `ordering` (layer 0 by default, via a
    `LayersPresenter` sync reaction on load/unload), so they appear in the panel
    and group with instrument tracks via DnD; the score renders all row kinds
    from `LayersStore.layout` (bands), not the legacy `trackOrder` above them.
    `trackOrder`/`onMoveTrack` removed from MixerView + JotEditor. Audio
    load/mute/solo/waveform/playback unchanged (e2e green); new e2e asserts a
    loaded audio track shows in the panel. *(commit bbe7b3a)*

   --- remaining (cleanup; the user-facing feature is complete) ---

A. **Audio colour-inheritance + full `trackOrder` retirement.** The audio
   waveform colour still inherits via the retained `MixerStore.trackOrder`
   (`groupInstrumentLanes` reads its `pair:<lane>` groupId), so debug-bundle
   audio keeps its inherited colour but a *manually* panel-grouped audio track
   doesn't yet recolour from its `ordering` group. To finish: add
   `MixerStore.groupInstrumentLanesForAudio(audioId)` computed from `ordering`,
   re-point `resolveAudioInheritedColor` + `AudioTrack.lane`/`color` at it, seed
   the debug-bundle audio↔lane pairing as `ordering` groups
   (`applyDebugBundleOrdering`, replacing `buildDebugBundleTrackOrder`), then
   delete `trackOrder`/`TrackKey`/`reorderTrackOrder`/`syncTrackOrder`/
   `buildDebugBundleTrackOrder`/`laneOrder`/`firstInstrumentIdx`. NOTE the
   debug-bundle e2e is **gated** (`E2E_DEBUG_BUNDLE`), so the pairing change is
   not automatically verifiable here, needs a human/bundle check.

B. **CONTRACT** (internal cleanup, no user-visible change). Remove `lane`/
   `layerId` from notes (keep `lane` only on pattern-body template notes), switch
   `flatten`/membership from `layerId` to `trackId` (edited/inserted notes must
   then mint a `trackId`, incl. for brand-new lanes, and pattern-instantiated
   notes resolve a track at flatten), make `tracks`/`ordering` required. The app
   is fully functional with `lane`/`layerId` retained, so this is deferrable.

### Status (autonomous session)

**Phases 1-11 done, green, committed (~19 feature commits, `b487281..bbe7b3a`).**
The full Layers feature is implemented and verified: schema + read-model + the
panel (rename/colour, DnD reorder/cross-layer/in-out-group, groups
create/ungroup/reorder), the score rendering layers first-class (bands, groups,
per-track rows), per-track mute/solo/volume + playback filter, the "Visually
merge layers" view with per-note edit routing, and audio/lyrics tracks folded
into `ordering` (groupable with instrument tracks; rendered from the bands).
Verified by driving a real browser with Playwright (screenshots of the layer
bands + merged view) and a broad e2e suite (layer_bands, layers_dnd,
multilayer_insert, audio_tracks). Build, unit suite (463), and functional e2e
are green at HEAD; the only red is the documented perf-zoom `120fps` contention
flake (passes 4/4 in isolation).

Remaining is cleanup only, no user-facing change: (A) migrate audio
colour-inheritance to `ordering` groups + fully retire the legacy `trackOrder`
(kept alive only for that colour path; debug-bundle e2e is gated so the pairing
change needs a human/bundle check); (B) the `lane`/`layerId` contraction.

### Notes for the next session

- `note.lane` is retained for pattern-body **template** notes (no layer
  context, so no `trackId`); `laneForNote` falls back to it. `lane` won't fully
  disappear at contraction, it stays on template notes.
- `ordering` is a top-level `movableList` (not a wrapper record) so it defaults
  to empty when unseeded. `jot.ordering` IS the ordered layer list.
- Track ids use their own counter (`tk0…`) inside `TrackBuilder`, NOT the shared
  element-id counter, using the shared one shifts element ids and, because
  `membership` string-sorts ids, perturbs first-seen lane order (caught by
  `legend.e2e.ts`).
- The **perf `zooming…120fps`** e2e is flaky on this host under load (~2/3
  pass; passes in isolation). Functional specs are deterministic. Don't chase it
  as a regression.

## Verification

After each phase: `bun run typecheck`/`test`/`lint:design` (tsc + bun unit + stylelint) and, since
every phase touches `src/**`, `bun run e2e` (full pass before claiming a phase
done; scoped `bun run e2e <spec>` while iterating). `bun run build` for a quick
compile smoke. Update e2e specs coupled to changed UI in the same phase. Commit
each phase with an explicit pathspec (never a bare `git commit`, the user edits
the tree concurrently).
