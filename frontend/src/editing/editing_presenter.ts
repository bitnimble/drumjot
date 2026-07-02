import { makeAutoObservable } from 'mobx';
import type { Element, MutableJot, NoteElement } from 'src/schema/schema';
import { transact } from 'src/schema/reactive_doc';
import { laneForNote, layerIdOfTrack } from 'src/schema/ordering';
import { SettingsStore } from 'src/settings/settings_store';
import type { StructLayer, StructNote } from 'src/editing/structure/structure_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenter } from 'src/editing/selection/selection_presenter';
import { LayersPresenter } from 'src/editing/layers/layers_presenter';
import { enabledDivisors, snapBeat } from './snap';
import { buildLaneMap, notesById } from './score/note_geometry';
import { EditingStore, type DragPreviewNote, type EditMode, type PlaceholderNote } from './editing_store';
import { JotEditorStore } from './jot_editor_store';
import type { ClipboardNote, ClipboardPayload } from './clipboard/clipboard_payload';
import type { Resettable } from './session_reset';

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
  /** True when the dragged selection spans more than one lane. A multi-lane
   *  group moves horizontally only, each note keeps its own lane, so a
   *  cross-lane drag never re-pitches the cluster. A single-lane group (incl. a
   *  lone note) still follows the cursor's lane. Frozen at drag start. */
  spansMultipleLanes: boolean;
  /** Rendered top-to-bottom lane order (mixer order), for the group lane shift. */
  laneOrder: readonly string[];
  lastClientX: number;
  lastTargetLane: string;
};

/** Per-paste-placement bookkeeping, the paste analogue of {@link DragMoveCtx}.
 *  Presenter-local. Where a drag moves EXISTING notes by a cursor delta, a paste
 *  places COPIED notes at an absolute cursor beat; both share the span rule +
 *  clamp via {@link placementPreview}, so a multi-lane paste preserves lanes
 *  exactly like a multi-lane drag. */
type PasteCtx = {
  /** Each pasted note: a synthetic preview id, its lane, the anchor-relative
   *  beat offset (`baseAbs`), and the source fields to write on commit. */
  members: { id: string; lane: string; baseAbs: number; note: ClipboardNote }[];
  /** True when the cluster spans more than one lane (then lanes are preserved;
   *  a single-lane cluster follows the cursor's lane). */
  spansMultipleLanes: boolean;
  /** Snaps the cursor's absolute anchor beat to the grid (identity when off). */
  snap: (rawAbs: number) => number;
  /** Timeline length in beats (upper clamp) + rendered lead-in offset. */
  total: number;
  leadInBeats: number;
  laneOrder: readonly string[];
  /** Latest cursor anchor beat (musical) + lane the cursor is over. */
  lastAnchorAbs: number;
  lastTargetLane: string;
  /** False until the first pointer move over the score gives a real position;
   *  a commit before then is a no-op (nothing placed). */
  hasCursor: boolean;
};

/** One bar's place in the absolute-beat coordinate (cumulative across bars). */
type BarSlot = { id: string; start: number; beats: number };

type Gesture =
  | { kind: 'idle' }
  | { kind: 'drag'; ctx: DragMoveCtx }
  | { kind: 'paste'; ctx: PasteCtx };

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
export class EditingPresenter implements Resettable {
  /** The in-flight gesture (drag / paste), or `idle`. Single source of truth;
   *  {@link setGesture} keeps the observable `dragActive`/`pasteActive` flags in
   *  lockstep. */
  private gesture: Gesture = { kind: 'idle' };

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
      | 'gesture'
    >(this, {
      editingStore: false,
      jotEditorStore: false,
      settingsStore: false,
      selectionStore: false,
      selectionPresenter: false,
      layersPresenter: false,
      gesture: false,
    });
  }

  /** The single writer of gesture state: keeps editingStore.dragActive /
   *  pasteActive in lockstep with `gesture` (so at most one is ever true) and
   *  clears the shared preview when idle or when a paste starts (a paste's
   *  preview appears on the first cursor move; a drag's is recomputed by
   *  applyPreview immediately after). */
  private setGesture(next: Gesture): void {
    this.gesture = next;
    this.editingStore.dragActive = next.kind === 'drag';
    this.editingStore.pasteActive = next.kind === 'paste';
    if (next.kind !== 'drag') this.editingStore.dragPreview = [];
  }

  /** Run a compound gesture's mutations as ONE Loro commit (= one undo step).
   *  Each facade write commits on its own, so a multi-write gesture (delete
   *  then set; provision a track then write notes) would otherwise fragment
   *  into several undo steps, and a single Ctrl+Z would leave the document
   *  half-applied (notes gone but group not, or an orphan track). Falls back to
   *  running `fn` directly when no doc is loaded. */
  private transactDoc(fn: () => void): void {
    const doc = this.jotEditorStore.loroDoc;
    if (doc) {
      transact(doc, fn);
    } else {
      fn();
    }
  }

  /**
   * Session reset: cancel any in-flight paste/drag placement (their ctx
   * captured the PREVIOUS song's bar layout, so a stale placement must not
   * survive into the new song, committing misplaced notes, or leave the
   * editor stuck in paste mode) and return the editing store to its
   * fresh-load state. Registered in {@link SessionReset} and fired before the
   * document swap on every wholesale load, so it's the single authority for
   * the editing-domain reset (this replaced a separate doc-swap reaction that
   * did the same cancel).
   */
  reset(): void {
    this.cancelPaste();
    this.cancelDragMove();
    this.editingStore.reset();
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
    const layers = this.jotEditorStore.jot?.musicalLayers;
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

    this.transactDoc(() => {
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
    });
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
    const layers = this.jotEditorStore.jot?.musicalLayers;
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
    this.transactDoc(() => {
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
    });
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
    const layers = this.jotEditorStore.jot?.musicalLayers;
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

    this.transactDoc(() => {
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
          const targetLayer = jot.ownerLayerFor(newLane) ?? curLayer;
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
    });
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
    const layers = this.jotEditorStore.jot?.musicalLayers;
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
    const layers = this.jotEditorStore.jot?.musicalLayers;
    if (!jot || !layers) return;
    // A drag supersedes any in-flight paste placement (they share `dragPreview`).
    this.cancelPaste();
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

    const ctx: DragMoveCtx = {
      anchor,
      startClientX,
      startLane,
      snap: this.snapDeltaFn(anchor),
      total: layout.total,
      leadInBeats: this.jotEditorStore.jot?.barsForLane(anchor.lane).leadInBarsBeats ?? 0,
      notes,
      spansMultipleLanes: new Set(notes.map((n) => n.lane)).size > 1,
      laneOrder: this.jotEditorStore.jot?.lanes ?? [],
      lastClientX: startClientX,
      lastTargetLane: anchor.lane,
    };
    this.setGesture({ kind: 'drag', ctx });
    this.applyPreview();
  }

  /**
   * Update the in-flight drag from a lane row's pointer move: `targetLane` is
   * the row the cursor is over (so cross-lane targeting needs no hit-testing),
   * `clientX` drives the horizontal delta, and `laneOrder` is that row's
   * rendered lane order (mixer order) for the group's vertical shift.
   */
  updateDragMove(targetLane: string, clientX: number, laneOrder: readonly string[]): void {
    if (this.gesture.kind !== 'drag') return;
    const ctx = this.gesture.ctx;
    // A frame drag seeds `startLane` lazily from the first reported row, so the
    // cross-lane shift is measured from wherever inside the frame the user
    // grabbed (zero at the start) rather than the snap anchor's lane.
    if (ctx.startLane === undefined) ctx.startLane = targetLane;
    ctx.lastClientX = clientX;
    ctx.lastTargetLane = targetLane;
    ctx.laneOrder = laneOrder;
    this.applyPreview();
  }

  /** Recompute the drag-move preview from the current cursor delta + target
   *  lane. Horizontal: `(clientX - startX) / pxPerBeat`, snapped to the grid;
   *  the snapped delta shifts every member off its rest position. Lane: the
   *  shared span rule (see {@link placementPreview}). */
  private applyPreview(): void {
    if (this.gesture.kind !== 'drag') return;
    const ctx = this.gesture.ctx;
    const px = this.jotEditorStore.layout?.pxPerBeat ?? 0;
    const rawDelta = px > 0 ? (ctx.lastClientX - ctx.startClientX) / px : 0;
    const shift = ctx.snap(rawDelta);
    this.editingStore.dragPreview = placementPreview(
      ctx.notes.map((n) => ({ id: n.id, lane: n.lane, baseAbs: n.origAbs })),
      shift,
      ctx.spansMultipleLanes,
      ctx.lastTargetLane,
      ctx.total,
      ctx.leadInBeats
    );
  }

  /** Commit the drag-move: writes the previewed positions via
   *  {@link moveSelection} (same snap + lane remap as the preview, so what you
   *  saw is what lands), then clears the preview. No-op if no drag is active. */
  commitDragMove(): void {
    const g = this.gesture;
    this.setGesture({ kind: 'idle' });
    if (g.kind !== 'drag') return;
    const ctx = g.ctx;
    const px = this.jotEditorStore.layout?.pxPerBeat ?? 0;
    const rawDelta = px > 0 ? (ctx.lastClientX - ctx.startClientX) / px : 0;
    // Span rule (mirrors `applyPreview`): a multi-lane group keeps every lane
    // (identity map); a single-lane group re-homes wholesale onto the cursor's
    // lane via the row-index shift.
    const laneMap = ctx.spansMultipleLanes
      ? SAME_LANE
      : buildLaneMap(ctx.laneOrder, ctx.startLane ?? ctx.anchor.lane, ctx.lastTargetLane);
    this.moveSelection(ctx.anchor, rawDelta, laneMap);
  }

  /** Abandon an in-flight drag-move (pointercancel), leaving the document
   *  untouched and clearing the preview. */
  cancelDragMove(): void {
    if (this.gesture.kind === 'drag') this.setGesture({ kind: 'idle' });
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
      this.jotEditorStore.jot?.ownerLayerFor(placeholder.lane) ??
      primaryLayerId(jot);
    this.transactDoc(() => {
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
    });
  }

  // ---------- Clipboard: copy + paste placement ----------

  /**
   * Serialize the current selection into a {@link ClipboardPayload} for the
   * clipboard, or `undefined` when nothing's selected. Positions are normalized
   * to the cluster's earliest note (`relBeat` 0); each note carries its lane +
   * musical fields, but NOT its identity / owning bar / track, those are
   * re-resolved from the drop position on paste. `copiedAt` lets a later copy
   * (here or in another tab) win the newer-wins pick. The {@link ClipboardPresenter}
   * owns the clipboard write; this is just the document → payload projection.
   */
  copySelectionPayload(): ClipboardPayload | undefined {
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.jot?.musicalLayers;
    const selected = this.selectionStore.selectedNotes;
    if (!jot || !layers || selected.size === 0) return undefined;
    const layout = buildBarLayout(layers);
    const rows: { lane: string; abs: number; el: NoteElement }[] = [];
    for (const n of selected) {
      const el = jot.elements.get(n.id) as Element | undefined;
      if (!el || el.kind !== 'note' || el.barId === undefined) continue;
      const slot = layout.byId.get(el.barId);
      if (!slot) continue;
      rows.push({ lane: laneForNote(jot, el), abs: slot.start + el.beat, el });
    }
    if (rows.length === 0) return undefined;
    const minAbs = Math.min(...rows.map((r) => r.abs));
    const notes: ClipboardNote[] = rows.map((r) => ({
      lane: r.lane,
      relBeat: r.abs - minAbs,
      duration: r.el.duration,
      modifiers: [...r.el.modifiers],
      ...(r.el.sticking !== undefined ? { sticking: r.el.sticking } : {}),
      ...(r.el.roll !== undefined ? { roll: r.el.roll } : {}),
      ...(r.el.offsetMs !== undefined ? { offsetMs: r.el.offsetMs } : {}),
      ...(r.el.velocity !== undefined ? { velocity: r.el.velocity } : {}),
      ...(r.el.midiNote !== undefined ? { midiNote: r.el.midiNote } : {}),
    }));
    return { copiedAt: Date.now(), notes };
  }

  /** Snap an absolute anchor beat to the grid (paste analogue of
   *  {@link snapDeltaFn}). Identity when snapping is off. */
  private snapAbsFn(total: number): (rawAbs: number) => number {
    if (!this.editingStore.snappingEnabled) return (a) => a;
    const divisors = enabledDivisors(this.settingsStore.gridLines);
    return (rawAbs) => snapBeat(rawAbs, divisors, total);
  }

  /**
   * Begin a paste placement of `payload`: the copied cluster becomes a live
   * preview that follows the cursor (a click commits it, Esc cancels). Nothing
   * is written until commit. The preview is empty until the first pointer move
   * over the score gives an anchor position. Supersedes any in-flight paste.
   */
  beginPaste(payload: ClipboardPayload): void {
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.jot?.musicalLayers;
    if (!jot || !layers || payload.notes.length === 0) return;
    // A paste supersedes any in-flight drag (they share `dragPreview`).
    this.cancelDragMove();
    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;
    const lanes = new Set(payload.notes.map((n) => n.lane));
    const anchorLane = payload.notes[0].lane;
    const ctx: PasteCtx = {
      members: payload.notes.map((n, i) => ({
        id: `paste:${i}`,
        lane: n.lane,
        baseAbs: n.relBeat,
        note: n,
      })),
      spansMultipleLanes: lanes.size > 1,
      snap: this.snapAbsFn(layout.total),
      total: layout.total,
      leadInBeats: this.jotEditorStore.jot?.barsForLane(anchorLane).leadInBarsBeats ?? 0,
      laneOrder: this.jotEditorStore.jot?.lanes ?? [],
      lastAnchorAbs: 0,
      lastTargetLane: anchorLane,
      hasCursor: false,
    };
    // Preview appears on the first pointer move (which seeds the anchor);
    // setGesture clears dragPreview.
    this.setGesture({ kind: 'paste', ctx });
  }

  /**
   * Update an in-flight paste from a bars-row pointer move: `anchorAbs` is the
   * cursor's absolute musical beat (the row's `placeholderAt` mapping) and
   * `targetLane` is the row the cursor is over. Mirrors {@link updateDragMove},
   * but the horizontal quantity is an absolute beat, not a delta.
   */
  updatePaste(anchorAbs: number, targetLane: string, laneOrder: readonly string[]): void {
    if (this.gesture.kind !== 'paste') return;
    const ctx = this.gesture.ctx;
    ctx.lastAnchorAbs = anchorAbs;
    ctx.lastTargetLane = targetLane;
    ctx.laneOrder = laneOrder;
    ctx.hasCursor = true;
    this.applyPastePreview();
  }

  /** Recompute the paste preview from the current cursor anchor + target lane,
   *  through the same {@link placementPreview} span rule as a drag. */
  private applyPastePreview(): void {
    if (this.gesture.kind !== 'paste') return;
    const ctx = this.gesture.ctx;
    const shift = ctx.snap(ctx.lastAnchorAbs);
    this.editingStore.dragPreview = placementPreview(
      ctx.members,
      shift,
      ctx.spansMultipleLanes,
      ctx.lastTargetLane,
      ctx.total,
      ctx.leadInBeats
    );
  }

  /**
   * Commit the paste: write every previewed note as a fresh note (new id,
   * resolved layer/track for its lane, owning bar from its absolute beat) in
   * one Loro commit (so it's one undo step), then select the pasted notes. The
   * lane + position match the preview exactly (same span rule + snap). No-op
   * before the first pointer move (no position chosen) or with no document.
   */
  commitPaste(): void {
    const g = this.gesture;
    this.setGesture({ kind: 'idle' });
    if (g.kind !== 'paste' || !g.ctx.hasCursor) return;
    const ctx = g.ctx;
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.jot?.musicalLayers;
    if (!jot || !layers) return;
    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;
    const shift = ctx.snap(ctx.lastAnchorAbs);
    const newIds: string[] = [];
    this.transactDoc(() => {
      const entries: [string, Record<string, unknown>][] = [];
      for (const m of ctx.members) {
        const lane = ctx.spansMultipleLanes ? m.lane : ctx.lastTargetLane;
        const abs = Math.min(Math.max(m.baseAbs + shift, 0), layout.total);
        const dest = homeBar(layout.slots, abs);
        // Resolve the note's home like insertNote: the firstmost layer owning the
        // lane, else the primary layer; mint its instrument track if needed.
        const layerId = jot.ownerLayerFor(lane) ?? primaryLayerId(jot);
        const trackId =
          layerId !== undefined ? this.layersPresenter.ensureInstrumentTrack(layerId, lane) : undefined;
        const id = crypto.randomUUID();
        newIds.push(id);
        const n = m.note;
        entries.push([
          id,
          {
            id,
            barId: dest.id,
            beat: abs - dest.start,
            duration: n.duration,
            kind: 'note',
            lane,
            modifiers: [...n.modifiers],
            ...(n.sticking !== undefined ? { sticking: n.sticking } : {}),
            ...(n.roll !== undefined ? { roll: n.roll } : {}),
            ...(n.offsetMs !== undefined ? { offsetMs: n.offsetMs } : {}),
            ...(n.velocity !== undefined ? { velocity: n.velocity } : {}),
            ...(n.midiNote !== undefined ? { midiNote: n.midiNote } : {}),
            ...(trackId !== undefined ? { trackId } : {}),
          },
        ]);
      }
      if (entries.length > 0) jot.elements.setAll(entries);
    });
    if (newIds.length === 0) return;
    // Select the freshly-pasted notes (resolved from the recomputed structure).
    const byId = notesById(this.jotEditorStore.jot?.musicalLayers ?? []);
    const pasted: StructNote[] = [];
    for (const id of newIds) {
      const n = byId.get(id);
      if (n) pasted.push(n);
    }
    this.selectionPresenter.setNotes(pasted);
  }

  /** Abandon an in-flight paste placement (Esc / a new load), writing nothing
   *  and clearing the preview. No-op when no paste is active. */
  cancelPaste(): void {
    if (this.gesture.kind !== 'paste') return;
    this.setGesture({ kind: 'idle' });
  }
}

/**
 * The shared placement core for drag-move AND paste: position each member at
 * `baseAbs + shift` (clamped to the timeline) and assign its lane by the span
 * rule, a cluster spanning >1 lane keeps every member's own lane (a
 * horizontal-only move), while a single-lane cluster moves wholesale onto
 * `targetLane` (the cursor's lane). Returns preview glyphs in the rendered
 * (lead-in-inclusive) coordinate. This is the single definition of "how a
 * multi-note group follows the cursor", reused so paste behaves identically to
 * a multi-lane drag.
 */
function placementPreview(
  members: readonly { id: string; lane: string; baseAbs: number }[],
  shift: number,
  spansMultipleLanes: boolean,
  targetLane: string,
  total: number,
  leadInBeats: number
): DragPreviewNote[] {
  return members.map((m) => {
    const lane = spansMultipleLanes ? m.lane : targetLane;
    const musicalAbs = Math.min(Math.max(m.baseAbs + shift, 0), total);
    return { id: m.id, lane, absBeat: leadInBeats + musicalAbs };
  });
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
