import { makeAutoObservable, reaction } from 'mobx';
import type { Element, MutableJot, NoteElement } from 'src/schema/schema';
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
  /** Live paste-placement bookkeeping, or undefined when no paste is in flight. */
  private pasteCtx: PasteCtx | undefined = undefined;
  /** Disposes the doc-swap reaction (teardown / leak tests). */
  private readonly disposeLoadReaction: () => void;

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
      | 'pasteCtx'
      | 'disposeLoadReaction'
    >(this, {
      editingStore: false,
      jotEditorStore: false,
      settingsStore: false,
      selectionStore: false,
      selectionPresenter: false,
      layersPresenter: false,
      dragCtx: false,
      pasteCtx: false,
      disposeLoadReaction: false,
    });
    // A song load swaps the backing document out from under any in-flight
    // paste/drag, whose ctx captured the PREVIOUS song's bar layout. Cancel
    // them on the swap so a stale placement can't survive into the new song
    // (committing misplaced notes) or leave the editor stuck in paste mode.
    this.disposeLoadReaction = reaction(
      () => this.jotEditorStore.loroDoc,
      () => {
        this.cancelPaste();
        this.cancelDragMove();
      }
    );
  }

  /** Tear down the doc-swap reaction (editor disposal / leak tests). The
   *  presenter is otherwise page-lifetime, so the live app never calls this. */
  dispose(): void {
    this.disposeLoadReaction();
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

    this.dragCtx = {
      anchor,
      startClientX,
      startLane,
      snap: this.snapDeltaFn(anchor),
      total: layout.total,
      leadInBeats: this.jotEditorStore.structural?.barsForLane(anchor.lane).leadInBarsBeats ?? 0,
      notes,
      spansMultipleLanes: new Set(notes.map((n) => n.lane)).size > 1,
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

  /** Recompute the drag-move preview from the current cursor delta + target
   *  lane. Horizontal: `(clientX - startX) / pxPerBeat`, snapped to the grid;
   *  the snapped delta shifts every member off its rest position. Lane: the
   *  shared span rule (see {@link placementPreview}). */
  private applyPreview(): void {
    const ctx = this.dragCtx;
    if (!ctx) return;
    const px = this.jotEditorStore.structural?.pxPerBeat ?? 0;
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
    const ctx = this.dragCtx;
    this.dragCtx = undefined;
    this.editingStore.dragActive = false;
    this.editingStore.dragPreview = [];
    if (!ctx) return;
    const px = this.jotEditorStore.structural?.pxPerBeat ?? 0;
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
    const layers = this.jotEditorStore.structural?.musicalLayers;
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
      ...(r.el.vol !== undefined ? { vol: r.el.vol } : {}),
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
    const layers = this.jotEditorStore.structural?.musicalLayers;
    if (!jot || !layers || payload.notes.length === 0) return;
    // A paste supersedes any in-flight drag (they share `dragPreview`).
    this.cancelDragMove();
    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;
    const lanes = new Set(payload.notes.map((n) => n.lane));
    const anchorLane = payload.notes[0].lane;
    this.pasteCtx = {
      members: payload.notes.map((n, i) => ({
        id: `paste:${i}`,
        lane: n.lane,
        baseAbs: n.relBeat,
        note: n,
      })),
      spansMultipleLanes: lanes.size > 1,
      snap: this.snapAbsFn(layout.total),
      total: layout.total,
      leadInBeats: this.jotEditorStore.structural?.barsForLane(anchorLane).leadInBarsBeats ?? 0,
      laneOrder: this.jotEditorStore.structural?.lanes ?? [],
      lastAnchorAbs: 0,
      lastTargetLane: anchorLane,
      hasCursor: false,
    };
    this.editingStore.pasteActive = true;
    // Preview appears on the first pointer move (which seeds the anchor).
    this.editingStore.dragPreview = [];
  }

  /**
   * Update an in-flight paste from a bars-row pointer move: `anchorAbs` is the
   * cursor's absolute musical beat (the row's `placeholderAt` mapping) and
   * `targetLane` is the row the cursor is over. Mirrors {@link updateDragMove},
   * but the horizontal quantity is an absolute beat, not a delta.
   */
  updatePaste(anchorAbs: number, targetLane: string, laneOrder: readonly string[]): void {
    const ctx = this.pasteCtx;
    if (!ctx) return;
    ctx.lastAnchorAbs = anchorAbs;
    ctx.lastTargetLane = targetLane;
    ctx.laneOrder = laneOrder;
    ctx.hasCursor = true;
    this.applyPastePreview();
  }

  /** Recompute the paste preview from the current cursor anchor + target lane,
   *  through the same {@link placementPreview} span rule as a drag. */
  private applyPastePreview(): void {
    const ctx = this.pasteCtx;
    if (!ctx) return;
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
    const ctx = this.pasteCtx;
    this.pasteCtx = undefined;
    this.editingStore.pasteActive = false;
    this.editingStore.dragPreview = [];
    if (!ctx || !ctx.hasCursor) return;
    const jot = this.jotEditorStore.jot;
    const layers = this.jotEditorStore.structural?.musicalLayers;
    if (!jot || !layers) return;
    const layout = buildBarLayout(layers);
    if (layout.slots.length === 0) return;
    const structural = this.jotEditorStore.structural;
    const shift = ctx.snap(ctx.lastAnchorAbs);
    const newIds: string[] = [];
    const entries: [string, Record<string, unknown>][] = [];
    for (const m of ctx.members) {
      const lane = ctx.spansMultipleLanes ? m.lane : ctx.lastTargetLane;
      const abs = Math.min(Math.max(m.baseAbs + shift, 0), layout.total);
      const dest = homeBar(layout.slots, abs);
      // Resolve the note's home like insertNote: the firstmost layer owning the
      // lane, else the primary layer; mint its instrument track if needed.
      const layerId = structural?.ownerLayerFor(lane) ?? primaryLayerId(jot);
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
          ...(n.vol !== undefined ? { vol: n.vol } : {}),
          ...(n.offsetMs !== undefined ? { offsetMs: n.offsetMs } : {}),
          ...(n.velocity !== undefined ? { velocity: n.velocity } : {}),
          ...(n.midiNote !== undefined ? { midiNote: n.midiNote } : {}),
          ...(trackId !== undefined ? { trackId } : {}),
        },
      ]);
    }
    if (entries.length === 0) return;
    jot.elements.setAll(entries);
    // Select the freshly-pasted notes (resolved from the recomputed structure).
    const byId = notesById(this.jotEditorStore.structural?.musicalLayers ?? []);
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
    if (!this.pasteCtx && !this.editingStore.pasteActive) return;
    this.pasteCtx = undefined;
    this.editingStore.pasteActive = false;
    this.editingStore.dragPreview = [];
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
