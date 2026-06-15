import { describe, expect, it } from 'bun:test';
import { bar, group, type Jot as DslJot, note } from 'src/schema/dsl/dsl';
import { StructureStore } from 'src/editing/structure/structure_store';
import { dslToReactive } from 'src/schema/dsl/from_dsl';

function structureOf(jot: DslJot) {
  const { model } = dslToReactive(jot);
  return { model, structure: new StructureStore(() => model) };
}

const META = {
  bpm: 120,
  time: { count: 4, unit: 4 },
  instrumentMapping: {
    k: { kind: 'kick' as const, name: 'Kick' },
    s: { kind: 'snare' as const },
  },
};

describe('dslToReactive', () => {
  it('carries metadata + instruments', () => {
    const { model } = structureOf({ title: 'T', globalMetadata: META, voices: [{ bars: [] }] });
    expect(model.title).toBe('T');
    expect(model.bpm).toBe(120);
    expect(model.instruments.get('k')!.name).toBe('Kick');
  });

  it('distributes a bar of straight notes across its beats', () => {
    const jot: DslJot = {
      title: '',
      globalMetadata: META,
      voices: [{ bars: [bar(note('k'), note('s'), note('k'), note('s'))] }],
    };
    const b = structureOf(jot).structure.voices[0].bars[0];
    expect(b.index).toBe(1);
    expect(b.tracks['k'].notes.map((n) => n.beat)).toEqual([0, 2]);
    expect(b.tracks['s'].notes.map((n) => n.beat)).toEqual([1, 3]);
  });

  it('converts a group filling the bar into a tuplet, scaling onsets', () => {
    const jot: DslJot = {
      title: '',
      globalMetadata: META,
      voices: [{ bars: [bar(group([note('k'), note('k'), note('k')]))] }],
    };
    const b = structureOf(jot).structure.voices[0].bars[0];
    expect(b.tupletSpans).toEqual([{ count: 3, startBeat: 0, endBeat: 4 }]);
    const beats = b.tracks['k'].notes.map((n) => n.beat);
    expect(beats[0]).toBeCloseTo(0);
    expect(beats[1]).toBeCloseTo(4 / 3);
    expect(beats[2]).toBeCloseTo(8 / 3);
  });

  it('preserves note modifiers and sticking', () => {
    const jot: DslJot = {
      title: '',
      globalMetadata: META,
      voices: [{ bars: [bar(note('s', { modifiers: ['a'], sticking: 'r' }))] }],
    };
    const note0 = structureOf(jot).structure.voices[0].bars[0].tracks['s'].notes[0];
    expect(note0.modifiers).toEqual(['a']);
    expect(note0.sticking).toBe('r');
  });
});
