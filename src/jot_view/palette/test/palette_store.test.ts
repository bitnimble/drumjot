import { describe, expect, it } from 'bun:test';
import { createReactiveJot, type Jot } from 'src/schema/schema';
import { StructureStore } from 'src/jot_view/structure/structure_store';
import { PaletteStore } from 'src/jot_view/palette/palette_store';

function fixture(palette: readonly string[]) {
  const { model } = createReactiveJot({
    title: '',
    bpm: 120,
    bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
    elements: {
      k: { kind: 'note', id: 'k', barId: 'b1', beat: 0, pitch: 'k', duration: 1, modifiers: [] },
      s: { kind: 'note', id: 's', barId: 'b1', beat: 1, pitch: 's', duration: 1, modifiers: [] },
      h: { kind: 'note', id: 'h', barId: 'b1', beat: 0, pitch: 'h', duration: 1, modifiers: [] },
    },
    instruments: {
      k: { kind: 'kick', name: 'Kick' },
      s: { kind: 'snare' },
      h: { kind: 'hihat' },
    },
  });
  const structure = new StructureStore(() => model);
  return { model, store: new PaletteStore(structure, () => palette, () => model) };
}

describe('PaletteStore', () => {
  it('assigns palette colours by jot-wide pitch order', () => {
    const { store } = fixture(['#aaa', '#bbb', '#ccc']);
    expect(store.jotPitches).toEqual(['k', 's', 'h']);
    expect(store.colorForPitch('k')).toBe('#aaa');
    expect(store.colorForPitch('s')).toBe('#bbb');
    expect(store.colorForPitch('h')).toBe('#ccc');
  });

  it('wraps colours when there are more pitches than palette slots', () => {
    const { store } = fixture(['#aaa', '#bbb']);
    expect(store.colorForPitch('h')).toBe('#aaa'); // index 2 wraps
  });

  it('builds a legend with colour + instrument name per pitch', () => {
    const { store } = fixture(['#aaa', '#bbb', '#ccc']);
    expect(store.legend).toContainEqual(['k', { color: '#aaa', name: 'Kick' }]);
    expect(store.legend).toContainEqual(['s', { color: '#bbb', name: undefined }]);
  });
});
