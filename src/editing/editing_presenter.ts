import { makeAutoObservable } from 'mobx';
import type { Element, Jot, NoteElement } from 'src/schema/schema';
import { SettingsStore } from 'src/settings/settings_store';
import type { StructLayer, StructNote } from 'src/editing/structure/structure_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenter } from 'src/editing/selection/selection_presenter';
import { enabledDivisors, snapBeat } from './snap';
import { buildLaneMap } from './score/note_geometry';
import { EditingStore, type EditMode, type PlaceholderNote } from './editing_store';
import { JotEditorStore } from './jot_editor_store';

/** Per-drag bookkeeping captured at drag start; presenter-local (not store
 *  state, like an in-flight AbortController). Positions are recomputed top-down
 *  from these + the live cursor x / target lane, so the drag reads no DOM. */
type DragMoveCtx = {
  /** The grabbed note (drives snapping + the lane remap origin). */
  anchor: StructNote;
  /** Pointer x at drag start; horizontal motion is a pure delta from here. */
  startClientX: number;
  anchorOrigLane: string;
  /** Snaps a raw beat delta to the grid (identity when snapping is off). */
  snap: (rawDeltaBeat: number) => number;
  /** Timeline length in beats, the upper clamp for a dragged position. */
  total: number;
  /** Rendered lead-in beats before musical bar 0. Preview glyphs are positioned
   *  in the rendered (lead-in-inclusive) coordinate, while `buildBarLayout`
   *  works in musical beats, so this offset bridges the two. */
  leadInBeats: number;
  /** Every selected note's id, origin lane, and origin absolute beat. */
  notes: { id: string; lane: string; origAbs: number }[];
  /** Rendered top-to-bottom lane order (mixer order), for the group lane shift. */
  laneOrder: readonly string[];
  lastClientX: number;
  lastTargetLane: string;
};

/** One bar's place in the absolute-beat coordinate (cumulative across bars). */
type BarSlot = { id: string; start: number; beats: number };

/** Identity lane map (no cross-lane move). */
const SAME_LANE = (lane: string): string => lane;

/**
 * Duration (in quarter-note units) of a hand-inserted note. A 16th note;
 * a placeholder default until note-duration editing / grid snapping land.
 */
const INSERTED_NOTE_DURATION = 0.25;

/**
 * Mutations for the editing domain: switching {@link EditMode}, tracking the
 * insert-mode placeholder, and committing a placeholder as a real note in the
 * reactive document. The single writer of {@link EditingStore}; note inserts
 * go straight through `jotEditorStore.jot` (the sanctioned document-write path).
 */
export class EditingPresenter {
  /** Live drag-move bookkeeping, or undefined when no drag is in flight. */
  private dragCtx: DragMoveCtx | undefined = undefined;

  constructor(
    private readonly editingStore: EditingStore,
    private readonly jotEditorStore: JotEditorStore,
    private readonly settingsStore: SettingsStore,
    private readonly selectionStore: SelectionStore,
    private readonly selectionPresenter: SelectionPresenter
  ) {
    makeAutoObservable<
      this,
      | 'editingStore'
      | 'jotEditorStore'
      | 'settingsStore'
      | 'selectionStore'
      | 'selectionPresenter'
      | 'dragCtx'
    >(this, {
      editingStore: false,
      jotEditorStore: false,
      settingsStore: false,
      selectionStore: false,
      selectionPresenter: false,
      dragCtx: false,
    });
  }

  setMode(mode: EditMode): void {
    this.editingStore.mode = mode;
    // Leaving insert mode drops any stale preview.
    if (mode !== 'insert') this.editingStore.placeholder = undefined;
  }

  /** Enable/disable grid snapping (Edit menu). */
  setSnapping(on: boolean): void {
    this.editingStore.snappingEnabled = on;
  }

  movePlaceholder(placeholder: PlaceholderNote): void {
    // Snap the preview as it moves so the user sees where the note will land;
    // `insertNote` then commits the already-snapped beat verbatim.
    this.editingStore.placeholder = this.snapPlaceholder(placeholder);
  }

  /** Snap a placeholder's beat to the grid (union of enabled families) when
   *  snapping is on, recomputing its `absBeat`. Identity when snapping is off. */
  private snapPlaceholder(p: PlaceholderNote): PlaceholderNote {
    if (!this.editingStore.snappingEnabled) return p;
    const divisors = enabledDivisors(this.settingsStore.gridLines);
    const beat = snapBeat(p.beat, divisors, p.barBeats);
    if (beat === p.beat) return p;
    return { ...p, beat, absBeat: p.absBeat - p.beat + beat };
  }

  clearPlaceholder(): void {
    this.editingStore.placeholder = undefined;
  }

  /**
   * Delete every selected note in one Loro commit, then clear the selection.
   * Entry-point-agnostic: called from the keymap today, a context menu later.
   * No-op with an empty selection or no loaded document.
   */
  deleteSelection(): void {
    const jot = this.jotEditorStore.jot;
    const notes = this.selectionStore.selectedNotes;
    if (!jot || notes.size === 0) return;
    jot.elements.delete(...[...notes].map((n) => n.id));
    this.selectionPresenter.clear();
  }

  /**
   * Move the selected notes by `deltaBeat` (in absolute-beat space) with the
   * anchor snapping to the grid and every other note following by the same
   * snapped delta, preserving relative spacing. Notes re-home across bar
   * boundaries automatically (absolute-beat → owning bar). `laneMap` remaps
   * each note's lane (cross-lane drag); identity by default. One Loro commit.
   *
   * All math is in tempo-independent beat units, so meter changes, discrete
   * tempo changes, and (unimplemented) interpolated tempo never enter here; * a quarter note stays a quarter note; beat→time is a downstream concern.
   */
  moveSelection(anchor: StructNote, deltaBeat: number, laneMap: (lane: string) => string = SAME_LANE): void {
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.structural?.musicalLayers;
    if (!jot || !layers || this.selectionStore.selectedNotes.size === 0) return;

    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;
    const byId = layout.byId;

    const absOf = (id: string): number | undefined => {
      const el = jot.elements.get(id) as Element | undefined;
      if (!el || el.barId === undefined) return undefined;
      const slot = byId.get(el.barId);
      return slot ? slot.start + el.beat : undefined;
    };

    // Snap the ANCHOR's target position; the resulting delta drives the group.
    const anchorAbs = absOf(anchor.id);
    if (anchorAbs === undefined) return;
    let snappedDelta = deltaBeat;
    const rawTarget = anchorAbs + deltaBeat;
    if (this.editingStore.snappingEnabled) {
      const divisors = enabledDivisors(this.settingsStore.gridLines);
      snappedDelta = snapBeat(rawTarget, divisors, layout.total) - anchorAbs;
    }

    const structural = this.jotEditorStore.structural;
    const updates: [string, Record<string, unknown>][] = [];
    for (const note of this.selectionStore.selectedNotes) {
      const el = jot.elements.get(note.id) as Element | undefined;
      if (!el || el.kind !== 'note' || el.barId === undefined) continue;
      const cur = byId.get(el.barId);
      if (!cur) continue;
      const newAbs = Math.min(Math.max(cur.start + el.beat + snappedDelta, 0), layout.total);
      const dest = homeBar(layout.slots, newAbs);
      const newLane = laneMap(el.lane);
      // Re-home to the layer that owns the destination lane so a cross-lane
      // move lands in the row clicked (a single-layer jot keeps `layerId`
      // unset; an unowned/new lane keeps the note's current layer).
      const owner = newLane === el.lane ? undefined : structural?.ownerLayerFor(newLane);
      updates.push([
        note.id,
        {
          ...el,
          barId: dest.id,
          beat: newAbs - dest.start,
          lane: newLane,
          ...(owner !== undefined ? { layerId: owner } : {}),
        },
      ]);
    }
    if (updates.length > 0) jot.elements.setAll(updates);
  }

  /**
   * Build the anchor-snapping function for a live drag: maps a raw beat delta
   * (cursor pixels ÷ pxPerBeat) to the delta after snapping the anchor's
   * absolute target onto the grid. Captures the bar layout, grid divisors, and
   * snapping flag once at drag start so each pointer move snaps without
   * rebuilding. Identity when snapping is off or the anchor can't be located,
   * so the preview matches what {@link moveSelection} ultimately commits.
   */
  snapDeltaFn(anchor: StructNote): (rawDeltaBeat: number) => number {
    const identity = (d: number): number => d;
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.structural?.musicalLayers;
    if (!jot || !layers || !this.editingStore.snappingEnabled) return identity;
    const layout = buildBarLayout(layers);
    const el = jot.elements.get(anchor.id) as Element | undefined;
    if (!el || el.barId === undefined) return identity;
    const slot = layout.byId.get(el.barId);
    if (!slot) return identity;
    const anchorAbs = slot.start + el.beat;
    const divisors = enabledDivisors(this.settingsStore.gridLines);
    return (rawDeltaBeat) => snapBeat(anchorAbs + rawDeltaBeat, divisors, layout.total) - anchorAbs;
  }

  /**
   * Start a drag-move of the current selection (grabbing `anchor`, which joins
   * the selection if it wasn't in it). Captures each selected note's origin
   * position so the preview + commit are computed top-down from the live
   * cursor delta and target lane, no DOM measured. Sets the initial preview
   * (notes at rest) and flips `dragActive` so the real glyphs hide.
   */
  beginDragMove(anchor: StructNote, startClientX: number): void {
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.structural?.musicalLayers;
    if (!jot || !layers) return;
    if (!this.selectionStore.isSelected(anchor)) this.selectionPresenter.replace(anchor);

    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;
    const absOf = (id: string): number | undefined => {
      const el = jot.elements.get(id) as Element | undefined;
      if (!el || el.barId === undefined) return undefined;
      const slot = layout.byId.get(el.barId);
      return slot ? slot.start + el.beat : undefined;
    };
    const notes: DragMoveCtx['notes'] = [];
    for (const n of this.selectionStore.selectedNotes) {
      const origAbs = absOf(n.id);
      if (origAbs !== undefined) notes.push({ id: n.id, lane: n.lane, origAbs });
    }
    if (notes.length === 0) return;

    this.dragCtx = {
      anchor,
      startClientX,
      anchorOrigLane: anchor.lane,
      snap: this.snapDeltaFn(anchor),
      total: layout.total,
      leadInBeats: this.jotEditorStore.structural?.barsForLane(anchor.lane).leadInBarsBeats ?? 0,
      notes,
      laneOrder: this.jotEditorStore.structural?.lanes ?? [],
      lastClientX: startClientX,
      lastTargetLane: anchor.lane,
    };
    this.editingStore.dragActive = true;
    this.applyPreview();
  }

  /**
   * Update the in-flight drag from a lane row's pointer move: `targetLane` is
   * the row the cursor is over (so cross-lane targeting needs no hit-testing),
   * `clientX` drives the horizontal delta, and `laneOrder` is that row's
   * rendered lane order (mixer order) for the group's vertical shift.
   */
  updateDragMove(targetLane: string, clientX: number, laneOrder: readonly string[]): void {
    const ctx = this.dragCtx;
    if (!ctx) return;
    ctx.lastClientX = clientX;
    ctx.lastTargetLane = targetLane;
    ctx.laneOrder = laneOrder;
    this.applyPreview();
  }

  /** Recompute the preview glyph positions from the current cursor delta +
   *  target lane. Horizontal: `(clientX - startX) / pxPerBeat`, snapped. Lane:
   *  each note shifts by the same row delta the anchor moved over `laneOrder`. */
  private applyPreview(): void {
    const ctx = this.dragCtx;
    if (!ctx) return;
    const px = this.jotEditorStore.structural?.pxPerBeat ?? 0;
    const rawDelta = px > 0 ? (ctx.lastClientX - ctx.startClientX) / px : 0;
    const snapped = ctx.snap(rawDelta);
    const order = ctx.laneOrder;
    const fromIdx = order.indexOf(ctx.anchorOrigLane);
    const toIdx = order.indexOf(ctx.lastTargetLane);
    const rowDelta = fromIdx >= 0 && toIdx >= 0 ? toIdx - fromIdx : 0;
    this.editingStore.dragPreview = ctx.notes.map((n) => {
      const i = order.indexOf(n.lane);
      const lane = i >= 0 ? order[Math.min(Math.max(i + rowDelta, 0), order.length - 1)] : n.lane;
      const musicalAbs = Math.min(Math.max(n.origAbs + snapped, 0), ctx.total);
      // Glyphs render in the rendered (lead-in-inclusive) coordinate.
      return { id: n.id, lane, absBeat: ctx.leadInBeats + musicalAbs };
    });
  }

  /** Commit the drag-move: writes the previewed positions via
   *  {@link moveSelection} (same snap + lane remap as the preview, so what you
   *  saw is what lands), then clears the preview. No-op if no drag is active. */
  commitDragMove(): void {
    const ctx = this.dragCtx;
    this.dragCtx = undefined;
    this.editingStore.dragActive = false;
    this.editingStore.dragPreview = [];
    if (!ctx) return;
    const px = this.jotEditorStore.structural?.pxPerBeat ?? 0;
    const rawDelta = px > 0 ? (ctx.lastClientX - ctx.startClientX) / px : 0;
    const laneMap = buildLaneMap(ctx.laneOrder, ctx.anchorOrigLane, ctx.lastTargetLane);
    this.moveSelection(ctx.anchor, rawDelta, laneMap);
  }

  /** Abandon an in-flight drag-move (pointercancel), leaving the document
   *  untouched and clearing the preview. */
  cancelDragMove(): void {
    this.dragCtx = undefined;
    this.editingStore.dragActive = false;
    this.editingStore.dragPreview = [];
  }

  /**
   * Commit the current insert-mode placeholder as a real note at exactly its
   * displayed bar + beat (snapping, if on, has already been applied to the
   * placeholder). No-op when there's no placeholder or no loaded document.
   */
  insertNote(): void {
    const placeholder = this.editingStore.placeholder;
    const jot = this.jotEditorStore.jot;
    if (!placeholder || !jot) return;
    const id = crypto.randomUUID();
    // Single-layer jots declare no layers; the structure store treats that as
    // the synthetic primary layer, so we omit `layerId`. Multi-layer jots must
    // tag the note with the layer that OWNS the clicked lane (e.g. the kick's
    // Feet layer in a hands/feet split) so it lands in the row clicked, not
    // whichever layer happens to be first. Falls back to the primary layer for
    // a brand-new lane no layer carries yet.
    const layerId =
      this.jotEditorStore.structural?.ownerLayerFor(placeholder.lane) ?? primaryLayerId(jot);
    const note: NoteElement = {
      id,
      barId: placeholder.barId,
      beat: placeholder.beat,
      duration: INSERTED_NOTE_DURATION,
      kind: 'note',
      lane: placeholder.lane,
      modifiers: [],
      ...(layerId !== undefined ? { layerId } : {}),
    };
    jot.elements.set(id, note);
  }
}

/** Build the absolute-beat coordinate from the musical bars (real bars only,
 *  no view-only lead-in): each bar's cumulative `start`, plus a by-id lookup
 *  and the total length. Bar geometry is shared across layers, so layer 0's
 *  bar list defines it. */
function buildBarLayout(layers: readonly StructLayer[]): {
  slots: BarSlot[];
  byId: Map<string, BarSlot>;
  total: number;
} {
  const bars = layers[0]?.bars ?? [];
  const slots: BarSlot[] = [];
  const byId = new Map<string, BarSlot>();
  let start = 0;
  for (const bar of bars) {
    const slot = { id: bar.id, start, beats: bar.beats };
    slots.push(slot);
    byId.set(bar.id, slot);
    start += bar.beats;
  }
  return { slots, byId, total: start };
}

/** The bar containing absolute beat `abs`; the last bar for a position at or
 *  past the end (so a note clamped to the timeline end lands in-bar). */
function homeBar(slots: readonly BarSlot[], abs: number): BarSlot {
  for (const slot of slots) {
    if (abs < slot.start + slot.beats) return slot;
  }
  return slots[slots.length - 1];
}

/** First declared layer id (numeric-sorted), mirroring `StructureStore`'s
 *  primary-layer pick. `undefined` for single-layer jots (no declared layers). */
function primaryLayerId(jot: Jot): string | undefined {
  const ids = [...jot.layers.keys()];
  if (ids.length === 0) return undefined;
  return ids.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))[0];
}
