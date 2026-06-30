import { describe, expect, it } from 'bun:test';
import { runInAction } from 'mobx';
import { NotePropertiesStore, MIXED } from 'src/editing/note_properties/note_properties_store';
import { SelectionStore } from 'src/editing/selection/selection';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import type { StructNote } from 'src/editing/structure/structure_store';
import { parse } from 'src/schema/dsl/parser/parser';

const META =
  '{{ time: "4/4", instrumentMapping: { h:{name:"HiHat"}, s:{name:"Snare"}, k:{name:"Kick"} } }}';

function setup(dsl: string): { editor: JotEditorStore; selection: SelectionStore; store: NotePropertiesStore } {
  const editor = new JotEditorStore();
  runInAction(() => editor.loadSource(parse(`${META} ${dsl}`)));
  const selection = new SelectionStore();
  return { editor, selection, store: new NotePropertiesStore(selection, editor) };
}

/** All notes on a lane across the first bar, in beat order. */
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

describe('NotePropertiesStore aggregation', () => {
  it('reports nothing for an empty selection', () => {
    const { store } = setup('| h s k . |');
    expect(store.count).toBe(0);
    expect(store.noteIdLabel).toBeUndefined();
  });

  it('reads a single note: lane, 1-based bar/beat, default volume, no roll', () => {
    const { editor, selection, store } = setup('| h s k . |');
    select(selection, lane(editor, 's')[0]); // snare on beat 2 (0-based beat 1)
    expect(store.count).toBe(1);
    expect(store.lane).toBe('s');
    expect(store.bar).toBe(1);
    expect(store.beat).toBe(2);
    expect(store.volumeUi).toBe(6); // DEFAULT_VELOCITY 80 -> step 6
    expect(store.roll).toBe(false);
    expect(store.noteIdLabel).toBe(`id: ${lane(editor, 's')[0].id}`);
  });

  it('summarises the on articulations, Roll first', () => {
    const { editor, selection, store } = setup('| s~:r . . . |');
    select(selection, lane(editor, 's')[0]);
    expect(store.articulationSummary).toBe('Roll, Rimshot');
  });

  it('converts an authored dynamic to the volume scale + label', () => {
    const { editor, selection, store } = setup('| s{vol: ff} . . . |');
    select(selection, lane(editor, 's')[0]);
    expect(store.volumeUi).toBe(10);
    expect(store.volumeLabel).toBe('ff');
  });

  it('marks differing numeric fields as mixed', () => {
    const { editor, selection, store } = setup('| s . s . |'); // two snares, beats 1 and 3
    select(selection, ...lane(editor, 's'));
    expect(store.lane).toBe('s'); // same lane
    expect(store.bar).toBe(1); // same bar
    expect(store.beat).toBe(MIXED); // different beats
  });

  it('reports a multi-lane selection as a mixed lane', () => {
    const { editor, selection, store } = setup('| h s k . |');
    select(selection, lane(editor, 'h')[0], lane(editor, 'k')[0]);
    expect(store.lane).toBe(MIXED);
    expect(store.availableLanes.map((l) => l.lane).sort()).toEqual(['h', 'k', 's']);
  });

  it('enables only modifiers valid for the selected lane', () => {
    const { editor, selection, store } = setup('| h s k . |');
    select(selection, lane(editor, 'h')[0]); // hi-hat: open/closed valid, rimshot not
    const byMod = new Map(store.modifierRows.map((r) => [r.mod, r.enabled]));
    expect(byMod.get('o')).toBe(true); // open is a hi-hat articulation
    expect(byMod.get('r')).toBe(false); // rimshot is not
  });

  it('intersects modifier validity across a multi-lane selection', () => {
    const { editor, selection, store } = setup('| h s k . |');
    select(selection, lane(editor, 'h')[0], lane(editor, 's')[0]);
    const byMod = new Map(store.modifierRows.map((r) => [r.mod, r.enabled]));
    expect(byMod.get('o')).toBe(false); // open invalid for snare -> disabled for the pair
    expect(byMod.get('m')).toBe(true); // mute valid for both
  });
});
