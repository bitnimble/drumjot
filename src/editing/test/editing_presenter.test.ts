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

// One kick on each downbeat of two 4/4 bars.
const SRC =
  '{{ bpm: 120, time: "4/4", instrumentMapping: { k:{name:"Kick"} } }} | k . . . | k . . . |';

function setup() {
  const store = new JotEditorStore();
  store.loadSource(parse(SRC));
  const editingStore = new EditingStore();
  const presenter = new EditingPresenter(editingStore, store);
  return { store, editingStore, presenter };
}

describe('EditingPresenter', () => {
  it('defaults to select mode', () => {
    const { editingStore } = setup();
    expect(editingStore.mode).toBe('select');
  });

  it('switching out of insert mode clears the placeholder', () => {
    const { editingStore, presenter } = setup();
    presenter.setMode('insert');
    presenter.movePlaceholder({ lane: 'k', barId: 'x', beat: 1, absBeat: 1 });
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
    presenter.movePlaceholder({ lane: 'k', barId: bar0.id, beat: 2, absBeat: 2 });
    presenter.insertNote();

    const notes = store.structural!.musicalLayers[0].bars[0].tracks['k'].notes;
    // The original downbeat kick plus the newly inserted one at beat 2.
    expect(notes.map((n) => n.beat).sort((a, b) => a - b)).toEqual([0, 2]);
  });
});
