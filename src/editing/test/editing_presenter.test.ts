/**
 * Unit tests for EditingPresenter: mode switching and committing an
 * insert-mode placeholder as a real note in the reactive document. The
 * inserted note must reflow into the structural layers.
 */
import { describe, expect, it } from 'bun:test';
import { parse } from 'src/schema/dsl/parser/parser';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { EditingStore } from 'src/editing/editing_store';
import { EditingPresenter } from 'src/editing/editing_presenter';
import { SettingsStore } from 'src/settings/settings_store';

// One kick on each downbeat of two 4/4 bars.
const SRC =
  '{{ bpm: 120, time: "4/4", instrumentMapping: { k:{name:"Kick"} } }} | k . . . | k . . . |';

function setup() {
  const store = new JotEditorStore();
  store.loadSource(parse(SRC));
  const editingStore = new EditingStore();
  const settings = new SettingsStore();
  const presenter = new EditingPresenter(editingStore, store, settings);
  return { store, editingStore, settings, presenter };
}

describe('EditingPresenter', () => {
  it('defaults to select mode', () => {
    const { editingStore } = setup();
    expect(editingStore.mode).toBe('select');
  });

  it('switching out of insert mode clears the placeholder', () => {
    const { editingStore, presenter } = setup();
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: 'x', beat: 1, absBeat: 1, barBeats: 4 });
    expect(editingStore.placeholder).toBeDefined();
    presenter.setMode('select');
    expect(editingStore.placeholder).toBeUndefined();
  });

  it('insertNote is a no-op without a placeholder', () => {
    const { store, presenter } = setup();
    const before = store.structural!.musicalLayers[0].bars[0].tracks['k'].notes.length;
    presenter.insertNote();
    const after = store.structural!.musicalLayers[0].bars[0].tracks['k'].notes.length;
    expect(after).toBe(before);
  });

  it('commits the placeholder as a note at its exact bar + beat', () => {
    const { store, presenter } = setup();
    const bar0 = store.structural!.musicalLayers[0].bars[0];
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: bar0.id, beat: 2, absBeat: 2, barBeats: 4 });
    presenter.insertNote();

    const notes = store.structural!.musicalLayers[0].bars[0].tracks['k'].notes;
    // The original downbeat kick plus the newly inserted one at beat 2.
    expect(notes.map((n) => n.beat).sort((a, b) => a - b)).toEqual([0, 2]);
  });

  it('snaps the placeholder to the grid when snapping is enabled', () => {
    const { editingStore, settings, presenter } = setup();
    // Default grid has main beats + 16ths; nearest 16th to 1.1 is 1.0.
    settings.gridLines = { ...settings.gridLines, subBeat16: true };
    presenter.setSnapping(true);
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: 'x', beat: 1.1, absBeat: 1.1, barBeats: 4 });
    expect(editingStore.placeholder!.beat).toBeCloseTo(1.0, 9);
    expect(editingStore.placeholder!.absBeat).toBeCloseTo(1.0, 9);
  });

  it('leaves the placeholder unsnapped when snapping is disabled', () => {
    const { editingStore, presenter } = setup();
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: 'x', beat: 1.1, absBeat: 1.1, barBeats: 4 });
    expect(editingStore.placeholder!.beat).toBeCloseTo(1.1, 9);
  });
});
