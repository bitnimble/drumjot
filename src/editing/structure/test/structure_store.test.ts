import { describe, expect, it } from 'bun:test';
import { autorun, runInAction } from 'mobx';
import { createReactiveJot, type Jot } from 'src/schema/schema';
import { StructureStore } from 'src/editing/structure/structure_store';

function store(jot: Jot) {
  return new StructureStore(() => jot);
}

describe('grouping', () => {
  it('groups top-level note elements into bars and per-pitch tracks, sorted by beat', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      elements: {
        n2: { kind: 'note', id: 'n2', barId: 'b1', beat: 2, duration: 1, pitch: 'k', modifiers: [] },
        n1: { kind: 'note', id: 'n1', barId: 'b1', beat: 0, duration: 1, pitch: 'k', modifiers: [] },
        h1: { kind: 'note', id: 'h1', barId: 'b1', beat: 0, duration: 0.5, pitch: 'h', modifiers: [] },
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
      elements: {
        on: { kind: 'note', id: 'on', barId: 'b1', beat: 1, duration: 1, pitch: 'k', modifiers: [] },
        off: { kind: 'note', id: 'off', barId: 'b1', beat: 1 / 3, duration: 1, pitch: 'k', modifiers: [] },
      },
      instruments: {},
    });
    const notes = store(model).voices[0].bars[0].tracks['k'].notes;
    expect(notes.find((n) => n.beat === 1)!.straight).toBe(true);
    expect(notes.find((n) => n.beat !== 1)!.straight).toBe(false);
  });

  it('is reactive: adding an element updates the derived structure', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      elements: {},
      instruments: {},
    });
    const s = store(model);
    const counts: number[] = [];
    const dispose = autorun(() => counts.push(s.voices[0]?.bars[0]?.tracks['k']?.notes.length ?? 0));
    runInAction(() => {
      model.elements.set('n1', {
        kind: 'note',
        id: 'n1',
        barId: 'b1',
        beat: 0,
        duration: 1,
        pitch: 'k',
        modifiers: [],
      });
    });
    dispose();
    expect(counts).toEqual([0, 1]);
  });
});

describe('bar indexing', () => {
  it('numbers a plain bar 1, lead-in negative, anacrusis 0', () => {
    const { model: plain } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      elements: {},
      instruments: {},
    });
    expect(store(plain).voices[0].bars[0].index).toBe(1);

    const { model: lead } = createReactiveJot({
      title: '',
      bpm: 120,
      leadBars: 1,
      bars: [
        { id: 'b0', tsCount: 4, tsUnit: 4 },
        { id: 'b1', tsCount: 4, tsUnit: 4 },
      ],
      elements: {},
      instruments: {},
    });
    expect(store(lead).voices[0].bars.map((b) => b.index)).toEqual([-1, 1]);

    const { model: ana } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [
        { id: 'b0', tsCount: 4, tsUnit: 4, anacrusis: true },
        { id: 'b1', tsCount: 4, tsUnit: 4 },
      ],
      elements: {
        a: { kind: 'note', id: 'a', barId: 'b0', beat: 0, duration: 1, pitch: 'k', modifiers: [] },
      },
      instruments: {},
    });
    const bars = store(ana).voices[0].bars;
    expect(bars.map((b) => b.index)).toEqual([0, 1]);
    expect(bars[0].beats).toBe(1); // content-sized
  });
});

describe('groups & tuplets', () => {
  it('scales a group that fills its duration with no tuplet bracket', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      elements: {
        g: {
          kind: 'group',
          id: 'g',
          barId: 'b1',
          beat: 0,
          duration: 1,
          children: {
            a: { kind: 'note', id: 'a', beat: 0, duration: 0.5, pitch: 'k', modifiers: [] },
            b: { kind: 'note', id: 'b', beat: 0.5, duration: 0.5, pitch: 'k', modifiers: [] },
          },
        },
      },
      instruments: {},
    });
    const bar = store(model).voices[0].bars[0];
    expect(bar.tupletSpans).toEqual([]);
    expect(bar.tracks['k'].notes.map((n) => n.beat)).toEqual([0, 0.5]);
  });

  it('brackets a group whose children overflow its duration as a tuplet, scaling onsets', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      elements: {
        // 3 eighths (natural span 1.5) compressed into 1 beat → triplet.
        g: {
          kind: 'group',
          id: 'g',
          barId: 'b1',
          beat: 0,
          duration: 1,
          children: {
            a: { kind: 'note', id: 'a', beat: 0, duration: 0.5, pitch: 'k', modifiers: [] },
            b: { kind: 'note', id: 'b', beat: 0.5, duration: 0.5, pitch: 'k', modifiers: [] },
            c: { kind: 'note', id: 'c', beat: 1, duration: 0.5, pitch: 'k', modifiers: [] },
          },
        },
      },
      instruments: {},
    });
    const bar = store(model).voices[0].bars[0];
    expect(bar.tupletSpans).toEqual([{ count: 3, startBeat: 0, endBeat: 1 }]);
    const beats = bar.tracks['k'].notes.map((n) => n.beat);
    expect(beats[0]).toBeCloseTo(0);
    expect(beats[1]).toBeCloseTo(1 / 3);
    expect(beats[2]).toBeCloseTo(2 / 3);
  });
});

describe('pattern spans', () => {
  it('instantiates a pattern element: span named/coloured + body notes expanded', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      patterns: {
        p1: {
          id: 'p1',
          name: 'groove',
          body: {
            pa: { kind: 'note', id: 'pa', beat: 0, duration: 1, pitch: 'k', modifiers: [] },
            pb: { kind: 'note', id: 'pb', beat: 1, duration: 1, pitch: 's', modifiers: [] },
          },
        },
      },
      elements: {
        pe: { kind: 'pattern', id: 'pe', barId: 'b1', beat: 0, duration: 2, patternId: 'p1' },
      },
      instruments: {},
    });
    const bar = store(model).voices[0].bars[0];
    expect(bar.patternSpans.length).toBe(1);
    expect(bar.patternSpans[0].name).toBe('groove');
    expect(bar.patternSpans[0].startBeat).toBe(0);
    expect(bar.patternSpans[0].endBeat).toBe(2);
    expect([...bar.patternSpans[0].pitches].sort()).toEqual(['k', 's']);
    expect(bar.patternSpans[0].colorIndex).toBe(0);
    expect(bar.tracks['k'].notes[0].beat).toBe(0);
    expect(bar.tracks['s'].notes[0].beat).toBe(1);
  });
});
