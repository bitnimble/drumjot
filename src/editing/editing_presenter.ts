import { makeAutoObservable } from 'mobx';
import type { Element, Jot, NoteElement } from 'src/schema/schema';
import { SettingsStore } from 'src/settings/settings_store';
import type { StructLayer, StructNote } from 'src/editing/structure/structure_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenter } from 'src/editing/selection/selection_presenter';
import { enabledDivisors, snapBeat } from './snap';
import { EditingStore, type EditMode, type PlaceholderNote } from './editing_store';
import { JotEditorStore } from './jot_editor_store';

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
  constructor(
    private readonly editingStore: EditingStore,
    private readonly jotEditorStore: JotEditorStore,
    private readonly settingsStore: SettingsStore,
    private readonly selectionStore: SelectionStore,
    private readonly selectionPresenter: SelectionPresenter
  ) {
    makeAutoObservable<
      this,
      'editingStore' | 'jotEditorStore' | 'settingsStore' | 'selectionStore' | 'selectionPresenter'
    >(this, {
      editingStore: false,
      jotEditorStore: false,
      settingsStore: false,
      selectionStore: false,
      selectionPresenter: false,
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

    const updates: [string, Record<string, unknown>][] = [];
    for (const note of this.selectionStore.selectedNotes) {
      const el = jot.elements.get(note.id) as Element | undefined;
      if (!el || el.kind !== 'note' || el.barId === undefined) continue;
      const cur = byId.get(el.barId);
      if (!cur) continue;
      const newAbs = Math.min(Math.max(cur.start + el.beat + snappedDelta, 0), layout.total);
      const dest = homeBar(layout.slots, newAbs);
      updates.push([
        note.id,
        { ...el, barId: dest.id, beat: newAbs - dest.start, lane: laneMap(el.lane) },
      ]);
    }
    if (updates.length > 0) jot.elements.setAll(updates);
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
    // tag the note with its owning layer or it won't render in any layer.
    const layerId = primaryLayerId(jot);
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
