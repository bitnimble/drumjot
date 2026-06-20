import { describe, expect, it } from 'bun:test';
import { runInAction } from 'mobx';
import type { Limb, Modifier, Sticking } from 'src/schema/dsl/dsl';
import type { DrumInstrumentKind } from 'src/instruments/instruments';
import { idMap, type Infer, record } from 'src/schema/descriptors';
import { createReactiveDoc } from 'src/schema/reactive_doc';
import {
  createMutableJot,
  GroupElementSchema,
  type GroupElement,
  type Instrument,
  JotSchema,
  type NoteElement,
  NoteElementSchema,
} from 'src/schema/schema';

// ---------- Type-level fidelity: the note variant matches the DSL types ----------
// The note-variant record is precisely inferable; assert `Infer<…>` is
// bidirectionally assignable to a hand-written mirror that references the
// DSL enums, so a missing modifier letter or wrong optionality fails tsc.

type NoteInfer = Infer<typeof NoteElementSchema>;
type ExpectedNote = {
  kind: 'note';
  id: string;
  layerId?: string;
  barId?: string;
  beat: number;
  duration: number;
  lane: string;
  modifiers: Modifier[];
  sticking?: Sticking;
  roll?: boolean;
  offsetMs?: number;
  velocity?: number;
  midiNote?: number;
  midiTick?: number;
};
type ExpectedInstrument = {
  kind: DrumInstrumentKind;
  name?: string;
  limb?: Limb;
  midiNote?: number;
};
const _noteFwd: ExpectedNote = null as unknown as NoteInfer;
const _noteBwd: NoteInfer = null as unknown as ExpectedNote;
const _instFwd: ExpectedInstrument = null as unknown as Instrument;
const _instBwd: Instrument = null as unknown as ExpectedInstrument;
void [_noteFwd, _noteBwd, _instFwd, _instBwd];

// ---------- Structural ----------

describe('JotSchema shape', () => {
  it('uses the right container kind per field', () => {
    expect(JotSchema.fields.bars.kind).toBe('movableList');
    expect(JotSchema.fields.elements.kind).toBe('idMap');
    expect(JotSchema.fields.patterns.kind).toBe('idMap');
    expect(JotSchema.fields.title.kind).toBe('reg');
  });

  it('the note variant is a record of registers', () => {
    expect(NoteElementSchema.fields.beat.kind).toBe('reg');
    expect(NoteElementSchema.fields.lane.kind).toBe('reg');
    expect(NoteElementSchema.fields.kind.kind).toBe('reg'); // discriminant
  });

  it('the group variant nests its children as an idMap', () => {
    expect(GroupElementSchema.fields.children.kind).toBe('idMap');
  });
});

// ---------- Runtime round-trip of the note variant ----------

describe('NoteElementSchema round-trips through a reactive doc', () => {
  it('stores and reads back a full note element', () => {
    const Doc = record({ els: idMap(NoteElementSchema) });
    const { model } = createReactiveDoc(Doc);
    runInAction(() => {
      model.els.set('n1', {
        kind: 'note',
        id: 'n1',
        beat: 1.5,
        duration: 0.5,
        lane: 'h',
        modifiers: ['a', 'o'],
        sticking: 'r',
      });
    });
    const n = model.els.get('n1')!;
    expect(n.beat).toBe(1.5);
    expect(n.lane).toBe('h');
    expect(n.modifiers).toEqual(['a', 'o']);
    expect(n.sticking).toBe('r');
    expect(n.roll).toBeUndefined();
  });
});

describe('createMutableJot', () => {
  it('deep-initializes a whole Jot from a plain object', () => {
    const { model } = createMutableJot({
      title: 'Breakbeat',
      bpm: 174,
      bars: [
        { id: 'b1', tsCount: 4, tsUnit: 4 },
        { id: 'b2', tsCount: 4, tsUnit: 4, tempoBpm: 180 },
      ],
      elements: {
        n1: { kind: 'note', id: 'n1', barId: 'b1', beat: 0, duration: 1, lane: 'k', modifiers: [] },
        n2: { kind: 'note', id: 'n2', barId: 'b1', beat: 2, duration: 1, lane: 's', modifiers: ['a'] },
      },
      instruments: { k: { kind: 'kick', name: 'Kick' }, s: { kind: 'snare' } },
    });

    expect(model.title).toBe('Breakbeat');
    expect(model.bpm).toBe(174);
    expect(model.bars.length).toBe(2);
    expect(model.bars.at(1)!.tempoBpm).toBe(180);
    expect(model.elements.size).toBe(2);
    expect((model.elements.get('n2') as NoteElement).modifiers).toEqual(['a']);
    expect(model.instruments.get('k')!.name).toBe('Kick');
  });

  it('starts empty when no initial object is given', () => {
    const { model } = createMutableJot();
    expect(model.bars.length).toBe(0);
    expect(model.elements.size).toBe(0);
  });

  it('snapshots to a plain JotState that seeds an identical mutable jot', () => {
    const seed = {
      title: 'Breakbeat',
      bpm: 174,
      bars: [
        { id: 'b1', tsCount: 4, tsUnit: 4 },
        { id: 'b2', tsCount: 4, tsUnit: 4, tempoBpm: 180 },
      ],
      elements: {
        n1: { kind: 'note' as const, id: 'n1', barId: 'b1', beat: 0, duration: 1, lane: 'k', modifiers: [] },
      },
      instruments: { k: { kind: 'kick' as const, name: 'Kick' } },
    };
    const doc = createMutableJot(seed);

    const state = doc.snapshot();
    // Plain JS surfaces: `bars` is an array, `elements`/`instruments` are
    // records keyed by id, no ReactiveMap/ReactiveList in sight.
    expect(Array.isArray(state.bars)).toBe(true);
    expect(state.bars.length).toBe(2);
    expect((state.elements.n1 as { lane: string }).lane).toBe('k');
    expect(state.instruments.k.name).toBe('Kick');

    // The snapshot round-trips straight back into a fresh mutable jot.
    const { model: clone } = createMutableJot(state);
    expect(clone.title).toBe('Breakbeat');
    expect(clone.bars.length).toBe(2);
    expect(clone.bars.at(1)!.tempoBpm).toBe(180);
    expect((clone.elements.get('n1') as NoteElement).lane).toBe('k');
    expect(clone.instruments.get('k')!.name).toBe('Kick');
  });

  it('edits round-trip (a top-level note element lane is one write)', () => {
    const { model } = createMutableJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      elements: {
        n1: { kind: 'note', id: 'n1', barId: 'b1', beat: 0, duration: 1, lane: 'cr', modifiers: [] },
      },
      instruments: {},
    });
    const n = model.elements.get('n1') as NoteElement;
    runInAction(() => {
      n.lane = 'rd';
    });
    expect((model.elements.get('n1') as NoteElement).lane).toBe('rd');
  });

  it('deep-initializes layers, tempo events, a pattern def, and a nested group', () => {
    const { model } = createMutableJot({
      title: 'x',
      bpm: 120,
      layers: { v0: { id: 'v0', name: 'Hands' } },
      bars: [
        { id: 'b0', tsCount: 4, tsUnit: 4, anacrusis: true },
        { id: 'b1', tsCount: 4, tsUnit: 4 },
      ],
      elements: {
        // A triplet group: 3 children spanning 1.5 natural beats in 1 beat.
        g1: {
          kind: 'group',
          id: 'g1',
          layerId: 'v0',
          barId: 'b1',
          beat: 0,
          duration: 1,
          children: {
            c1: { kind: 'note', id: 'c1', beat: 0, duration: 0.5, lane: 'k', modifiers: [] },
            c2: { kind: 'note', id: 'c2', beat: 0.5, duration: 0.5, lane: 's', modifiers: [] },
          },
        },
      },
      instruments: {},
      tempoEvents: { t1: { id: 't1', barId: 'b1', beat: 0, bpm: 140 } },
      patterns: { p1: { id: 'p1', name: 'groove', body: {} } },
    });
    expect(model.layers.get('v0')!.name).toBe('Hands');
    expect(model.bars.at(0)!.anacrusis).toBe(true);
    const g = model.elements.get('g1') as GroupElement;
    expect(g.kind).toBe('group');
    expect(g.children.size).toBe(2);
    expect((g.children.get('c1') as NoteElement).lane).toBe('k');
    expect(model.tempoEvents.get('t1')!.bpm).toBe(140);
    expect(model.patterns.get('p1')!.name).toBe('groove');
  });
});
