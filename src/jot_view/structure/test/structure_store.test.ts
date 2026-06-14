import { describe, expect, it } from 'bun:test';
import { autorun, runInAction } from 'mobx';
import { createReactiveJot, type Jot } from 'src/schema/schema';
import { StructureStore } from 'src/jot_view/structure/structure_store';

function store(jot: Jot) {
  return new StructureStore(() => jot);
}

describe('grouping', () => {
  it('groups notes into bars and per-pitch tracks, sorted by beat', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      notes: {
        n2: { id: 'n2', barId: 'b1', beat: 2, pitch: 'k', duration: 1, modifiers: [] },
        n1: { id: 'n1', barId: 'b1', beat: 0, pitch: 'k', duration: 1, modifiers: [] },
        h1: { id: 'h1', barId: 'b1', beat: 0, pitch: 'h', duration: 0.5, modifiers: [] },
      },
      instruments: {},
    });
    const s = store(model);
    expect(s.voices.length).toBe(1);
    const bar = s.voices[0].bars[0];
    expect(bar.beats).toBe(4);
    expect(bar.tracks['k'].notes.map((n) => n.beat)).toEqual([0, 2]);
    expect(bar.tracks['h'].notes.length).toBe(1);
  });

  it('flags off-grid (non-dyadic) onsets as not straight', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      notes: {
        on: { id: 'on', barId: 'b1', beat: 1, pitch: 'k', duration: 1, modifiers: [] },
        off: { id: 'off', barId: 'b1', beat: 1 / 3, pitch: 'k', duration: 1, modifiers: [] },
      },
      instruments: {},
    });
    const notes = store(model).voices[0].bars[0].tracks['k'].notes;
    expect(notes.find((n) => n.beat === 1)!.straight).toBe(true);
    expect(notes.find((n) => n.beat !== 1)!.straight).toBe(false);
  });

  it('is reactive: adding a note updates the derived structure', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      notes: {},
      instruments: {},
    });
    const s = store(model);
    const counts: number[] = [];
    const dispose = autorun(() => counts.push(s.voices[0]?.bars[0]?.tracks['k']?.notes.length ?? 0));
    runInAction(() => {
      model.notes.set('n1', { id: 'n1', barId: 'b1', beat: 0, pitch: 'k', duration: 1, modifiers: [] });
    });
    dispose();
    expect(counts).toEqual([0, 1]);
  });
});

describe('bar indexing', () => {
  it('numbers a plain bar as 1', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      notes: {},
      instruments: {},
    });
    expect(store(model).voices[0].bars[0].index).toBe(1);
  });

  it('numbers lead-in bars negative and the first drum bar 1', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      leadBars: 1,
      bars: [
        { id: 'b0', tsCount: 4, tsUnit: 4 },
        { id: 'b1', tsCount: 4, tsUnit: 4 },
      ],
      notes: {},
      instruments: {},
    });
    const bars = store(model).voices[0].bars;
    expect(bars.map((b) => b.index)).toEqual([-1, 1]);
  });

  it('numbers an anacrusis bar 0 and sizes it to its content', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [
        { id: 'b0', tsCount: 4, tsUnit: 4, anacrusis: true },
        { id: 'b1', tsCount: 4, tsUnit: 4 },
      ],
      notes: {
        a: { id: 'a', barId: 'b0', beat: 0, pitch: 'k', duration: 1, modifiers: [] },
      },
      instruments: {},
    });
    const bars = store(model).voices[0].bars;
    expect(bars.map((b) => b.index)).toEqual([0, 1]);
    expect(bars[0].beats).toBe(1); // content-sized (one quarter note)
  });
});

describe('pattern spans', () => {
  it('builds one span per pattern instance, named and coloured by pattern', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      notes: {
        n1: { id: 'n1', barId: 'b1', beat: 0, pitch: 'k', duration: 1, modifiers: [], patternId: 'p1' },
        n2: { id: 'n2', barId: 'b1', beat: 2, pitch: 's', duration: 1, modifiers: [], patternId: 'p1' },
        loose: { id: 'loose', barId: 'b1', beat: 3, pitch: 'h', duration: 1, modifiers: [] },
      },
      instruments: {},
      patterns: { groove: { name: 'groove' } },
      patternInstances: { p1: { patternName: 'groove' } },
    });
    const spans = store(model).voices[0].bars[0].patternSpans;
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe('groove');
    expect(spans[0].startBeat).toBe(0);
    expect(spans[0].endBeat).toBe(2);
    expect([...spans[0].pitches].sort()).toEqual(['k', 's']);
    expect(spans[0].colorIndex).toBe(0);
  });
});
