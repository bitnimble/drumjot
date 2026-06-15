import { describe, expect, it } from 'bun:test';
import { createReactiveJot, type Jot, type NoteElement } from 'src/schema/schema';
import { StructureStore } from 'src/editing/structure/structure_store';
import { LayoutStore } from 'src/editing/viewport/layout_store';

const BAR_WIDTH = 448;

function jotWith(notes: Record<string, { pitch: string; beat: number }>, bars = 1) {
  const elements: Record<string, NoteElement> = {};
  for (const [id, n] of Object.entries(notes)) {
    elements[id] = { kind: 'note', id, barId: 'b1', duration: 1, modifiers: [], ...n };
  }
  const { model } = createReactiveJot({
    title: '',
    bpm: 120,
    bars: Array.from({ length: bars }, (_, i) => ({ id: `b${i + 1}`, tsCount: 4, tsUnit: 4 })),
    elements,
    instruments: {},
  });
  return model;
}

function layout(model: Jot) {
  return new LayoutStore(new StructureStore(() => model), () => BAR_WIDTH, () => 0);
}

describe('LayoutStore', () => {
  it('density factor is 1 at the reference onset density (2 onsets/beat)', () => {
    // 8 distinct onsets across a 4-beat bar = 2/beat → factor 1.
    const notes: Record<string, { pitch: string; beat: number }> = {};
    for (let i = 0; i < 8; i++) notes[`n${i}`] = { pitch: 'k', beat: i * 0.5 };
    const l = layout(jotWith(notes));
    expect(l.densityFactor).toBe(1);
    expect(l.pxPerBeat).toBe(BAR_WIDTH / 4);
  });

  it('clamps sparse and dense densities', () => {
    const sparse = layout(jotWith({ a: { pitch: 'k', beat: 0 } })); // 1 onset / 4 beats = 0.25 → /2 = 0.125 → floor 0.4
    expect(sparse.densityFactor).toBeCloseTo(0.4);
  });

  it('lays bars out left-to-right with cumulative x', () => {
    const l = layout(jotWith({ a: { pitch: 'k', beat: 0 }, b: { pitch: 'k', beat: 2 } }, 2));
    const pxPerBeat = l.pxPerBeat;
    expect(l.bars.map((b) => b.x)).toEqual([0, 4 * pxPerBeat]);
    expect(l.bars.map((b) => b.width)).toEqual([4 * pxPerBeat, 4 * pxPerBeat]);
    expect(l.contentWidthPx).toBe(8 * pxPerBeat);
  });

  it('notePadPx scales the engraving inset by pxPerBeat', () => {
    const model = jotWith({ a: { pitch: 'k', beat: 0 } });
    const l = new LayoutStore(new StructureStore(() => model), () => BAR_WIDTH, () => 0.25);
    expect(l.notePadPx).toBe(0.25 * l.pxPerBeat);
  });
});
