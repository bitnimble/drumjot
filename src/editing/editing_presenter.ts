import { makeAutoObservable } from 'mobx';
import type { Jot, NoteElement } from 'src/schema/schema';
import { SettingsStore } from 'src/settings/settings_store';
import { enabledDivisors, snapBeat } from './snap';
import { EditingStore, type EditMode, type PlaceholderNote } from './editing_store';
import { JotEditorStore } from './jot_editor_store';

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
    private readonly settingsStore: SettingsStore
  ) {
    makeAutoObservable<this, 'editingStore' | 'jotEditorStore' | 'settingsStore'>(this, {
      editingStore: false,
      jotEditorStore: false,
      settingsStore: false,
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
   * Commit the current insert-mode placeholder as a real note at exactly its
   * displayed bar + beat (no snapping yet). No-op when there's no placeholder
   * or no loaded document.
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

/** First declared layer id (numeric-sorted), mirroring `StructureStore`'s
 *  primary-layer pick. `undefined` for single-layer jots (no declared layers). */
function primaryLayerId(jot: Jot): string | undefined {
  const ids = [...jot.layers.keys()];
  if (ids.length === 0) return undefined;
  return ids.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))[0];
}
