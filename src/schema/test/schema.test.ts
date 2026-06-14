import { describe, expect, it } from 'bun:test';
import { runInAction } from 'mobx';
import type { Limb, Modifier, Sticking } from 'src/dsl/dsl';
import type { DrumInstrumentKind } from 'src/instruments/instruments';
import { idMap, record } from 'src/schema/descriptors';
import { createReactiveDoc } from 'src/schema/reactive_doc';
import {
  createReactiveJot,
  type Instrument,
  type Note,
  NoteSchema,
  JotSchema,
} from 'src/schema/schema';

// ---------- Type-level fidelity: Infer<schema> must equal the DSL types ----------
// These are compile-time assertions (tsc fails if the schema drifts from
// the domain, e.g. a missing modifier letter or a wrong optionality).

type ExpectedNote = {
  id: string;
  barId: string;
  beat: number;
  pitch: string;
  duration: number;
  modifiers: Modifier[];
  sticking?: Sticking;
  roll?: boolean;
  offsetMs?: number;
  velocity?: number;
  midiNote?: number;
};
type ExpectedInstrument = {
  kind: DrumInstrumentKind;
  name?: string;
  limb?: Limb;
  midiNote?: number;
};

// Bidirectional assignment: a drift (missing/extra field, missing enum
// member, wrong value type) makes one of the two directions fail to
// compile. Uses plain assignability, which, unlike a strict-identity
// conditional, doesn't false-alarm on zod's enum output vs an aliased
// DSL union.
const _noteFwd: ExpectedNote = null as unknown as Note;
const _noteBwd: Note = null as unknown as ExpectedNote;
const _instFwd: ExpectedInstrument = null as unknown as Instrument;
const _instBwd: Instrument = null as unknown as ExpectedInstrument;
void [_noteFwd, _noteBwd, _instFwd, _instBwd];

// ---------- Structural ----------

describe('JotSchema shape', () => {
  it('uses the right container kind per field', () => {
    expect(JotSchema.fields.bars.kind).toBe('movableList');
    expect(JotSchema.fields.notes.kind).toBe('idMap');
    expect(JotSchema.fields.instruments.kind).toBe('idMap');
    expect(JotSchema.fields.title.kind).toBe('reg');
    expect(JotSchema.fields.bpm.kind).toBe('reg');
  });

  it('notes are flat registers (no nested containers)', () => {
    expect(NoteSchema.fields.beat.kind).toBe('reg');
    expect(NoteSchema.fields.pitch.kind).toBe('reg');
    expect(NoteSchema.fields.modifiers.kind).toBe('reg');
    expect(NoteSchema.fields.sticking.kind).toBe('reg');
  });
});

// ---------- Runtime round-trip of the real NoteSchema ----------

describe('NoteSchema round-trips through a reactive doc', () => {
  it('stores and reads back a full note', () => {
    const NotesDoc = record({ notes: idMap(NoteSchema) });
    const { model } = createReactiveDoc(NotesDoc);
    runInAction(() => {
      model.notes.set('n1', {
        id: 'n1',
        barId: 'b1',
        beat: 1.5,
        pitch: 'h',
        duration: 0.5,
        modifiers: ['a', 'o'],
        sticking: 'r',
      });
    });
    const n = model.notes.get('n1')!;
    expect(n.beat).toBe(1.5);
    expect(n.pitch).toBe('h');
    expect(n.modifiers).toEqual(['a', 'o']);
    expect(n.sticking).toBe('r');
    expect(n.roll).toBeUndefined();
  });
});

describe('createReactiveJot', () => {
  it('deep-initializes a whole Jot from a plain object', () => {
    const { model } = createReactiveJot({
      title: 'Breakbeat',
      bpm: 174,
      bars: [
        { id: 'b1', tsCount: 4, tsUnit: 4 },
        { id: 'b2', tsCount: 4, tsUnit: 4, tempoBpm: 180 },
      ],
      notes: {
        n1: { id: 'n1', barId: 'b1', beat: 0, pitch: 'k', duration: 1, modifiers: [] },
        n2: { id: 'n2', barId: 'b1', beat: 2, pitch: 's', duration: 1, modifiers: ['a'] },
      },
      instruments: {
        k: { kind: 'kick', name: 'Kick' },
        s: { kind: 'snare' },
      },
    });

    expect(model.title).toBe('Breakbeat');
    expect(model.bpm).toBe(174);
    expect(model.bars.length).toBe(2);
    expect(model.bars.at(1)!.tempoBpm).toBe(180);
    expect(model.notes.size).toBe(2);
    expect(model.notes.get('n2')!.modifiers).toEqual(['a']);
    expect(model.instruments.get('k')!.name).toBe('Kick');
  });

  it('starts empty when no initial object is given', () => {
    const { model } = createReactiveJot();
    expect(model.bars.length).toBe(0);
    expect(model.notes.size).toBe(0);
  });

  it('edits round-trip (move a note A→B is a single pitch write)', () => {
    const { model } = createReactiveJot({
      title: '',
      bpm: 120,
      bars: [{ id: 'b1', tsCount: 4, tsUnit: 4 }],
      notes: { n1: { id: 'n1', barId: 'b1', beat: 0, pitch: 'cr', duration: 1, modifiers: [] } },
      instruments: {},
    });
    runInAction(() => {
      model.notes.get('n1')!.pitch = 'rd';
    });
    expect(model.notes.get('n1')!.pitch).toBe('rd');
  });
});
