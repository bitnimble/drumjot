import { makeAutoObservable } from 'mobx';
import type { Element, MutableJot, NoteElement } from 'src/schema/schema';
import { laneForNote, layerIdOfTrack } from 'src/schema/ordering';
import { SettingsStore } from 'src/settings/settings_store';
import type { StructLayer, StructNote } from 'src/editing/structure/structure_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenter } from 'src/editing/selection/selection_presenter';
import { LayersPresenter } from 'src/editing/layers/layers_presenter';
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
  /** Lane the vertical (cross-lane) shift is measured FROM. A notehead drag
   *  seeds this with the grabbed note's lane; a frame drag leaves it undefined
   *  so {@link updateDragMove} adopts the first lane the cursor reports (the row
   *  under the press), keeping the start shift at zero either way. */
  startLane: string | undefined;
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
    private readonly selectionPresenter: SelectionPresenter,
    private readonly layersPresenter: LayersPresenter
  ) {
    makeAutoObservable<
      this,
      | 'editingStore'
      | 'jotEditorStore'
      | 'settingsStore'
      | 'selectionStore'
      | 'selectionPresenter'
      | 'layersPresenter'
      | 'dragCtx'
    >(this, {
      editingStore: false,
      jotEditorStore: false,
      settingsStore: false,
      selectionStore: false,
      selectionPresenter: false,
      layersPresenter: false,
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
   * Group the selected notes into one {@link GroupElement} per `||` layer, IN
   * PLACE: the group's `duration` equals its children's natural span (scale 1),
   * so every note flattens back to its exact position and no tuplet bracket
   * fires, only a group frame appears. The children are rebased into the group's
   * internal space (keeping their ids, tracks, and modifiers); the group is
   * anchored at the bar holding its earliest note. A multi-layer selection
   * yields one group per layer; partitions with fewer than two top-level notes
   * (and notes already inside a group) are skipped.
   */
  groupSelection(): void {
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.structural?.musicalLayers;
    if (!jot || !layers || this.selectionStore.selectedNotes.size === 0) return;
    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;

    // Partition the selected, groupable note elements by their `||` layer (a
    // group lives in one layer; its layer derives from its children's tracks).
    const byLayer = new Map<string, { el: NoteElement; abs: number }[]>();
    for (const note of this.selectionStore.selectedNotes) {
      const el = jot.elements.get(note.id) as Element | undefined;
      // Only top-level notes (a note already in a group isn't in `jot.elements`).
      if (!el || el.kind !== 'note' || el.barId === undefined) continue;
      const slot = layout.byId.get(el.barId);
      if (!slot) continue;
      const layerKey = el.trackId !== undefined ? (layerIdOfTrack(jot, el.trackId) ?? '') : '';
      let arr = byLayer.get(layerKey);
      if (!arr) {
        arr = [];
        byLayer.set(layerKey, arr);
      }
      arr.push({ el, abs: slot.start + el.beat });
    }

    for (const members of byLayer.values()) {
      if (members.length < 2) continue;
      const start = Math.min(...members.map((m) => m.abs));
      const end = Math.max(...members.map((m) => m.abs + m.el.duration));
      const anchor = homeBar(layout.slots, start);
      const groupId = crypto.randomUUID();
      const children: Record<string, NoteElement> = {};
      for (const { el, abs } of members) {
        // Rebase into the group's internal space; drop `barId` (children live in
        // the group's coordinate space, not a bar's). Keep the id so selection /
        // provenance survive the regroup.
        children[el.id] = childNoteOf(el, abs - start);
      }
      jot.elements.delete(...members.map((m) => m.el.id));
      jot.elements.set(groupId, {
        kind: 'group',
        id: groupId,
        barId: anchor.id,
        beat: start - anchor.start,
        duration: end - start,
        children,
      } as unknown as Element);
    }
  }

  /**
   * Ungroup every group touched by the selection: flatten each owning
   * {@link GroupElement} back into top-level notes at their effective positions
   * (re-homed to the bar each lands in), keeping note ids, then remove the group
   * shell. The inverse of {@link groupSelection} for an in-place group; for a
   * tuplet it bakes the scaled onsets into plain notes. No-op if no selected
   * note belongs to a group.
   */
  ungroupSelection(): void {
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.structural?.musicalLayers;
    if (!jot || !layers) return;
    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;

    // Resolve the owning groups from the CURRENT structure by id: the selection
    // holds the `StructNote` objects as they were when selected (rendering
    // matches them by id), so a note selected BEFORE it was grouped carries no
    // `groupId` on its stale object, the live flattened note does.
    const selectedIds = new Set([...this.selectionStore.selectedNotes].map((n) => n.id));
    const groupIds = new Set<string>();
    for (const layer of layers) {
      for (const bar of layer.bars) {
        for (const lane of Object.keys(bar.tracks)) {
          for (const note of bar.tracks[lane].notes) {
            if (note.groupId !== undefined && selectedIds.has(note.id)) groupIds.add(note.groupId);
          }
        }
      }
    }
    for (const groupId of groupIds) {
      const group = jot.elements.get(groupId) as Element | undefined;
      if (!group || group.kind !== 'group' || group.barId === undefined) continue;
      const groupSlot = layout.byId.get(group.barId);
      if (!groupSlot) continue;
      const groupStart = groupSlot.start + group.beat;
      const children = [...group.children.values()] as Element[];
      const internalLen = children.reduce((m, c) => Math.max(m, c.beat + c.duration), 0);
      const scale = internalLen > 1e-9 ? group.duration / internalLen : 1;
      const restored: [string, NoteElement][] = [];
      for (const child of children) {
        if (child.kind !== 'note') continue;
        const abs = Math.min(Math.max(groupStart + child.beat * scale, 0), layout.total);
        const dest = homeBar(layout.slots, abs);
        // Fresh id: the child's original id was deleted from the top-level map
        // when the group formed, and resurrecting a tombstoned Loro key yields
        // an empty container, so mint a new one (matching `insertNote`).
        const id = crypto.randomUUID();
        restored.push([id, { ...childNoteOf(child, abs - dest.start), id, barId: dest.id }]);
      }
      jot.elements.delete(groupId);
      if (restored.length > 0) jot.elements.setAll(restored);
    }
    // The restored notes carry fresh ids, so the selection (still holding the
    // pre-ungroup note objects) no longer matches any element, clear it rather
    // than leave a dangling selection that renders nothing.
    if (groupIds.size > 0) this.selectionPresenter.clear();
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
      const curLane = laneForNote(jot, el);
      const newLane = laneMap(curLane);
      // A cross-lane move re-homes the note to the track for the destination
      // lane: in the layer that owns that lane (the row clicked), else the
      // note's own current layer. The note's layer is never stored, so only
      // its `trackId` changes; a same-lane move leaves the track untouched.
      let laneUpdate: Record<string, unknown> = {};
      if (newLane !== curLane) {
        const curLayer = el.trackId !== undefined ? layerIdOfTrack(jot, el.trackId) : undefined;
        const targetLayer = structural?.ownerLayerFor(newLane) ?? curLayer;
        const newTrackId =
          targetLayer !== undefined
            ? this.layersPresenter.ensureInstrumentTrack(targetLayer, newLane)
            : undefined;
        laneUpdate = { lane: newLane, ...(newTrackId !== undefined ? { trackId: newTrackId } : {}) };
      }
      updates.push([
        note.id,
        {
          ...el,
          barId: dest.id,
          beat: newAbs - dest.start,
          ...laneUpdate,
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
  beginDragMove(anchor: StructNote, startClientX: number, startLane?: string): void {
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
      startLane,
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
    // A frame drag seeds `startLane` lazily from the first reported row, so the
    // cross-lane shift is measured from wherever inside the frame the user
    // grabbed (zero at the start) rather than the snap anchor's lane.
    if (ctx.startLane === undefined) ctx.startLane = targetLane;
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
    const fromIdx = order.indexOf(ctx.startLane ?? ctx.anchor.lane);
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
    const laneMap = buildLaneMap(ctx.laneOrder, ctx.startLane ?? ctx.anchor.lane, ctx.lastTargetLane);
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
    // Resolve the layer the note lands in: the clicked row's layer (per-track
    // view) wins so it lands where clicked; a merged row carries no `layerId`,
    // so fall back to the firstmost layer that owns the lane (the merge-view
    // rule), then the primary layer for a brand-new lane no layer carries yet.
    // The note's home is then the instrument track for (layer, lane), minted
    // if the layer has none yet; so its layer derives from `ordering`, never
    // stored on the note.
    const layerId =
      placeholder.layerId ??
      this.jotEditorStore.structural?.ownerLayerFor(placeholder.lane) ??
      primaryLayerId(jot);
    const trackId =
      layerId !== undefined
        ? this.layersPresenter.ensureInstrumentTrack(layerId, placeholder.lane)
        : undefined;
    const note: NoteElement = {
      id,
      barId: placeholder.barId,
      beat: placeholder.beat,
      duration: INSERTED_NOTE_DURATION,
      kind: 'note',
      lane: placeholder.lane,
      modifiers: [],
      ...(trackId !== undefined ? { trackId } : {}),
    };
    jot.elements.set(id, note);
  }
}

/** A note element's child-init: its fields with `beat` rebased and `barId`
 *  dropped (a group's children live in the group's coordinate space). Undefined
 *  optionals are omitted so the reactive-doc write stays clean. Reused by
 *  ungroup, which adds a `barId` back on. */
function childNoteOf(el: NoteElement, beat: number): NoteElement {
  const out: Record<string, unknown> = {
    kind: 'note',
    id: el.id,
    beat,
    duration: el.duration,
    lane: el.lane,
    modifiers: [...el.modifiers],
  };
  if (el.trackId !== undefined) out.trackId = el.trackId;
  if (el.sticking !== undefined) out.sticking = el.sticking;
  if (el.roll !== undefined) out.roll = el.roll;
  if (el.offsetMs !== undefined) out.offsetMs = el.offsetMs;
  if (el.velocity !== undefined) out.velocity = el.velocity;
  if (el.midiNote !== undefined) out.midiNote = el.midiNote;
  if (el.midiTick !== undefined) out.midiTick = el.midiTick;
  if (el.vol !== undefined) out.vol = el.vol;
  return out as unknown as NoteElement;
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
function primaryLayerId(jot: MutableJot): string | undefined {
  const ids = [...jot.layers.keys()];
  if (ids.length === 0) return undefined;
  return ids.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))[0];
}
