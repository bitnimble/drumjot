import { describe, expect, it } from 'bun:test';
import { runInAction } from 'mobx';
import { NotePropertiesPresenter } from 'src/editing/note_properties/note_properties_presenter';
import { NotePropertiesStore } from 'src/editing/note_properties/note_properties_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { LayersPresenter } from 'src/editing/layers/layers_presenter';
import type { StructNote } from 'src/editing/structure/structure_store';
import type { NoteElement } from 'src/schema/schema';
import { parse } from 'src/schema/dsl/parser/parser';

const META =
  '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }}';

type Harness = {
  editor: JotEditorStore;
  selection: SelectionStore;
  presenter: NotePropertiesPresenter;
  store: NotePropertiesStore;
};

function setup(dsl: string): Harness {
  const editor = new JotEditorStore();
  runInAction(() => editor.loadSource(parse(`${META} ${dsl}`)));
  const selection = new SelectionStore();
  const layersPresenter = new LayersPresenter(() => editor.jot);
  return {
    editor,
    selection,
    presenter: new NotePropertiesPresenter(editor, selection, layersPresenter),
    store: new NotePropertiesStore(selection, editor),
  };
}

function lane(editor: JotEditorStore, laneId: string): StructNote[] {
  const out: StructNote[] = [];
  for (const bar of editor.structural!.layers[0].bars) {
    const t = bar.tracks[laneId];
    if (t) out.push(...t.notes);
  }
  return out;
}

function select(selection: SelectionStore, ...notes: StructNote[]): void {
  runInAction(() => {
    selection.state = { type: 'notes', notes: new Set(notes) };
  });
}

function el(editor: JotEditorStore, id: string): NoteElement {
  return editor.jot!.elements.get(id) as unknown as NoteElement;
}

describe('NotePropertiesPresenter', () => {
  it('sets a shared volume across the selection', () => {
    const { editor, selection, presenter } = setup('| s s . . |');
    const notes = lane(editor, 's');
    select(selection, ...notes);
    presenter.setVolume(10); // -> velocity 127
    for (const n of notes) expect(el(editor, n.id).velocity).toBe(127);
  });

  it('steps each note volume independently (preserving the spread)', () => {
    const { editor, selection, presenter } = setup('| s{vol: pp} s{vol: ff} . . |');
    const [soft, loud] = lane(editor, 's'); // velocities 13 and 127 (ui 1 and 10)
    select(selection, soft, loud);
    presenter.stepVolume(1);
    expect(el(editor, soft.id).velocity).toBe(25); // ui 2
    expect(el(editor, loud.id).velocity).toBe(127); // ui 10 clamps
  });

  it('nudges beat and carries overflow into the next bar', () => {
    const { editor, selection, presenter, store } = setup('| . . . s | k . . . |');
    const snare = lane(editor, 's')[0]; // bar 1, display beat 4
    select(selection, snare);
    expect(store.bar).toBe(1);
    expect(store.beat).toBe(4);
    presenter.stepBeat(1); // 4.25
    presenter.stepBeat(1); // 4.5
    presenter.stepBeat(1); // 4.75
    presenter.stepBeat(1); // -> overflow to bar 2, beat 1
    expect(store.bar).toBe(2);
    expect(store.beat).toBe(1);
  });

  it('moves a note to another bar, keeping its beat', () => {
    const { editor, selection, presenter, store } = setup('| . s . . | . . . . |');
    const snare = lane(editor, 's')[0]; // bar 1, display beat 2
    select(selection, snare);
    presenter.setBar(2);
    expect(store.bar).toBe(2);
    expect(store.beat).toBe(2); // beat preserved
  });

  it('re-homes notes onto a new lane', () => {
    const { editor, selection, presenter } = setup('| h s k . |');
    const snare = lane(editor, 's')[0];
    select(selection, snare);
    presenter.setLane('k');
    expect(el(editor, snare.id).lane).toBe('k');
  });

  it('toggling Roll on drops the modifiers it conflicts with', () => {
    const { editor, selection, presenter } = setup('| s:fl . . . |');
    const snare = lane(editor, 's')[0];
    select(selection, snare);
    expect(el(editor, snare.id).modifiers).toContain('fl');
    presenter.toggleRoll();
    expect(el(editor, snare.id).roll).toBe(true);
    expect(el(editor, snare.id).modifiers).not.toContain('fl');
  });

  it('tri-states a modifier from mixed -> all-on -> all-off', () => {
    const { editor, selection, presenter } = setup('| s:r s . . |');
    const notes = lane(editor, 's'); // one rimshot, one plain -> mixed
    select(selection, ...notes);
    presenter.toggleModifier('r'); // mixed -> all on
    for (const n of notes) expect(el(editor, n.id).modifiers).toContain('r');
    presenter.toggleModifier('r'); // all on -> all off
    for (const n of notes) expect(el(editor, n.id).modifiers).not.toContain('r');
  });

  it('sets and clears sticking', () => {
    const { editor, selection, presenter } = setup('| s . . . |');
    const snare = lane(editor, 's')[0];
    select(selection, snare);
    presenter.setSticking('r');
    expect(el(editor, snare.id).sticking).toBe('r');
    presenter.setSticking('none');
    expect(el(editor, snare.id).sticking).toBeUndefined();
  });

  it('nudges micro-timing by 1ms per step and clears at zero', () => {
    const { editor, selection, presenter } = setup('| s . . . |');
    const snare = lane(editor, 's')[0];
    select(selection, snare);
    presenter.stepMicroTiming(1);
    expect(el(editor, snare.id).offsetMs).toBe(1);
    presenter.stepMicroTiming(-1);
    expect(el(editor, snare.id).offsetMs).toBeUndefined();
  });
});
