# Editing: multi-select, delete, snapping, and drag-move

## Goal

Add four editing capabilities to the jot editor, plus the supporting
infrastructure they share:

1. **Multi-select**, ctrl-click (toggle individual), shift-click
   (extend consecutive), and marquee. With ≥2 notes selected, show a
   subtle bounding box ("selection frame").
2. **Delete**, selected notes removed via the Delete key. A single
   `deleteSelection()` method, callable from both the keymap (now) and a
   future context menu (later).
3. **Snapping**, toggled from a new **Edit** toolbar dropdown ("Enable
   snapping"). When on, positions snap to the nearest line in the
   **union** of all currently-enabled grid-line families. Affects both
   adding (fix up existing insert) and moving.
4. **Move**, select then drag horizontally (within a lane) and/or
   vertically (to another instrument lane), in a single drag. Snapping
   applies during move when enabled.

Two cross-cutting pieces of infrastructure fall out of this work and are
in scope:

- A **command-registry + keymap** keyboard layer, so keys can be remapped
  later from settings.
- A **bulk-mutation facade** on the reactive-doc collections so that
  multi-element edits land as a single Loro commit (one sync delta, one
  future undo step) instead of N.

## Non-goals

- Context menu / mobile delete affordance (later; `deleteSelection()` is
  built to be its entry-agnostic target).
- A settings UI for remapping keys (the keymap is built to support it;
  the UI is later).
- Note **duration** editing / resize-drag (out of scope; insert keeps its
  hardcoded default duration).
- Undo/redo itself (the transaction primitive sets it up cleanly, but
  undo/redo is not implemented here).

---

## Architecture

### 1. Reactive-doc bulk mutations (CRDT layer)

**Problem.** Every facade write (`idMap.set`, `idMap.delete`, and each
register field write) calls `doc.commit()` immediately and synchronously
(`src/schema/reactive_doc.ts:196,305,310`). Loro accumulates uncommitted
ops into the *next* `commit()`, so the efficient pattern is "all ops then
one commit". Per-op commits fragment a bulk delete/move into 3N tiny Loro
changes, N× the sync overhead and N undo steps.

**Solution.** A `transact(fn)` helper **private to `reactive_doc.ts`**:
sets a module-level "defer commit" flag so the per-op `doc.commit()`
calls become no-ops, runs `fn` inside `runInAction` (to also coalesce
MobX reactions), then issues a single `doc.commit()` on the outermost
exit. Re-entrant-safe via a depth counter (nested `transact` commits only
once, at the outer boundary).

`transact` is **not exported** and is **never called by schema
consumers**. Instead the collection facades expose bulk operators that
use it internally:

- `idMap.delete(...ids: string[])`; variadic; batched → one commit.
- `idMap.setAll(entries: Iterable<[string, Record<string, unknown>]>)`; bulk set; batched → one commit.
- Existing single-op `set(id, value)` / `delete(id)` stay as the
  common-case path (single op, single commit, unchanged behaviour).

Presenters call `jot.elements.delete(...ids)` and
`jot.elements.setAll(updates)`; they never see `transact`.

### 2. Selection layer (refactor + extend)

`SelectionStore` currently holds mutation methods
(`beginSelection`/`selectNote`/…), violating the store-is-data-only rule.
Split it:

- **`SelectionStore`** (data only): `selectedNotes: Set<StructNote>`,
  `marquee: Box | undefined`, an `anchor: StructNote | undefined` (for
  shift-range), plus computeds:
  - `selectedNote`, single-select convenience (the one note, else
    undefined), preserving today's inline-label suppression for
    multi-note selections.
  - `selectionFrame`, bounding box over selected notes in (lane,
    absBeat) space, or undefined when <2 selected.
- **`SelectionPresenter`** (all mutations): `replace(note)`,
  `toggle(note)` (ctrl-click), `extendTo(note)` (shift-click consecutive,
  using `anchor`), `setFromMarquee(box)`, `clear()`, and the marquee
  lifecycle (`beginMarquee/moveMarquee/endMarquee`). Replaces the
  existing in-store mutation methods.

`extendTo` selects the contiguous run of notes between `anchor` and the
clicked note in document order within the relevant lane(s); exact
ordering rule documented at implementation time against `StructNote`
ordering.

### 3. Selection frame

A subtle overlay box enclosing all selected notes, rendered only when ≥2
notes are selected (`selectionFrame` computed). Projected from (lane,
absBeat) extents to px via the existing layout mapping (`pxPerBeat`,
`barNotePaddingBeats`, track geometry), **no DOM layout reads**.
Placement respects the score z-index ladder; consult
`docs/score-stacking.md` before adding the overlay so it escapes its bar
correctly.

### 4. Keyboard layer, command registry + keymap

- **`editing_commands.ts`**, a registry of named, enumerable commands:
  `{ id, label, run(ctx) }`. Initial commands: `deleteSelection`,
  `togglePlayPause` (folds in the existing spacebar handler).
- **`keymap.ts`**, default map of key-combo → command id
  (`Delete`/`Backspace` → `deleteSelection`, `Space` → `togglePlayPause`).
  Structured so a future settings UI can override it.
- **`useEditorKeymap` hook** (in `jot_editor.tsx`), a single keydown
  dispatcher: resolves combo → command id → handler, reusing the existing
  text-input/contentEditable/select guard. Replaces the ad-hoc spacebar
  listener.

### 5. Snapping

- **State**: `snappingEnabled: boolean` on `EditingStore`. Mutated by
  `EditingPresenter.setSnapping(on)`.
- **Pure helper `snapBeat.ts`**: given an absolute beat, the owning bar's
  geometry, and the set of **enabled grid-line families** (from
  `SettingsStore.gridLines`: `mainBeat`, `subBeat16`,
  `subBeatQuarterTriplet`, `subBeatTriplet`, `subBeat48`), generate the
  **union** of all enabled families' grid points within the bar and
  return the nearest. With 12ths and 16ths both on, a note can land on
  either. Snapping off → continuous positions (no quantization).
- Used by **both** insert and move. Fixes up insert, which today places
  at the exact clicked beat with no snapping.

### 6. Edit toolbar dropdown

New **`edit_menu.tsx`** mirroring `view_menu.tsx`: a `DropdownButton`
labeled "Edit" containing an "Enable snapping" `ToggleMenuItem` bound to
`editingPresenter.setSnapping` / `editingStore.snappingEnabled`. Wired
into `toolbar.tsx`.

### 7. Drag-move

- **Mutations on `EditingPresenter`** (the note-mutating presenter,
  reading `SelectionStore`):
  - `deleteSelection()` → `jot.elements.delete(...selectedIds)`, then
    clears selection.
  - `moveSelection(deltaBeat, targetLane)` → compute each selected note's
    new position: the **anchor note snaps** to the grid (when enabled),
    all others move by the **same delta** (relative spacing preserved).
    **Cross-bar moves re-home `barId`** with a corrected bar-relative
    beat when a note crosses a bar boundary. Lane reassigned to
    `targetLane` (the currently-rendered instrument track under the
    pointer's y). Commit via a single `jot.elements.setAll(updates)`.
- **Drag interaction** (pointer handlers on notes / bars row): a small px
  threshold distinguishes click-select from drag-move. In-flight drag
  offset is **transient interaction state**, a presenter observable for
  live preview, committed to the document only on pointer-up. Pixel↔(beat,
  lane) mapping is factored into a **shared coordinate helper** extracted
  from the existing `placeholderAt` logic in `instrument_track_view.tsx`,
  reused by insert, marquee, and drag-move.

---

## Data flow

```
pointer / key
   │
   ├─ key  → useEditorKeymap → command id → command.run(ctx)
   │                                          → editingPresenter.deleteSelection()
   │                                               → jot.elements.delete(...ids)  [1 commit]
   │
   └─ pointer (note/bars row)
        ├─ click/ctrl/shift → SelectionPresenter.{replace,toggle,extendTo}
        ├─ marquee drag     → SelectionPresenter.{begin,move,end}Marquee → setFromMarquee
        └─ move drag        → transient drag offset (presenter observable, live preview)
                              → on pointer-up → editingPresenter.moveSelection(Δbeat, lane)
                                   → snapBeat(anchor) + same-delta for rest, re-home barId
                                   → jot.elements.setAll(updates)               [1 commit]

SettingsStore.gridLines ──┐
EditingStore.snappingEnabled ─┴→ snapBeat()  (used by insert + move)
SelectionStore.selectedNotes → selectionFrame computed → overlay (≥2 notes)
```

## Store / presenter ownership (per project rules)

| Concern | Store (data) | Presenter (mutations) |
|---|---|---|
| Selection | `SelectionStore` (`selectedNotes`, `marquee`, `anchor`, `selectedNote`, `selectionFrame`) | `SelectionPresenter` (toggle/extend/marquee/clear) |
| Snapping flag | `EditingStore.snappingEnabled` | `EditingPresenter.setSnapping` |
| Note mutations | (document) | `EditingPresenter` (`insertNote`, `deleteSelection`, `moveSelection`) |
| Bulk commit | reactive-doc facade (`delete(...ids)`, `setAll`), `transact` private |, |

`SelectionPresenter` and `EditingPresenter` both read `SelectionStore`;
no store depends on a presenter; graph stays acyclic.

## Testing

- **Unit**: `snapBeat` (union of families, nearest, off=continuous,
  bar-edge behaviour); `transact` (single commit for N ops, re-entrant
  nesting, defer-flag reset on throw); `idMap.delete(...ids)` /
  `setAll` (one commit, correct final state); selection presenter
  (toggle/extend/marquee set math); `moveSelection` position math
  (anchor-snap + same-delta, cross-bar re-home).
- **e2e** (`src/editing/**/test/*.e2e.ts`): ctrl/shift/marquee select →
  selection frame appears for ≥2; Delete removes selection; Edit-menu
  snapping toggle changes insert placement; drag a note left/right snaps;
  drag a note onto another lane reassigns it; drag across a bar boundary
  re-homes it. Update any specs coupled to toolbar structure /
  `data-testid`s in the same change. Full `bun run e2e` green (26/26
  incl. perf) before claiming done.
- **Perf**: bulk delete/move stays within the 120 fps / 8.3 ms frame
  budget; verify the single-commit path and per-(bar,lane) computeds
  keep re-renders scoped (guard against a full re-render on multi-note
  edits).

## Risks / open details (resolve at implementation time)

- Exact contiguous-run ordering rule for shift-`extendTo` across
  multi-lane selections.
- Selection-frame placement against the score-stacking contract.
- Cross-bar re-home math at tempo/meter changes (bar geometry varies).
