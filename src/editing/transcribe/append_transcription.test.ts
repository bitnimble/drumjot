import { describe, expect, it } from 'bun:test';
import { bar, note, type Jot, type Instrument } from 'src/schema/dsl/dsl';
import { dslToMutable } from 'src/schema/dsl/from_dsl';
import { createBlankJot } from 'src/editing/new_jot';
import { appendTranscription } from './append_transcription';

const KIT: Record<string, Instrument> = {
  k: { kind: 'kick' },
  s: { kind: 'snare' },
  h: { kind: 'hihat' },
};

/** Build a tiny single-layer jot: one bar per entry, each holding one note of
 *  the given lane. */
function makeJot(lanesPerBar: string[], opts: { bpm?: number; tempoBpm?: number } = {}): Jot {
  return {
    title: 'test',
    globalMetadata: { instrumentMapping: KIT, time: { count: 4, unit: 4 }, bpm: opts.bpm ?? 120 },
    layers: [{ bars: lanesPerBar.map((lane) => bar(note(lane))) }],
    tempoEvents:
      opts.tempoBpm !== undefined ? [{ barIndex: 0, beat: 0, bpm: opts.tempoBpm }] : undefined,
  };
}

type Snap = ReturnType<ReturnType<typeof dslToMutable>['snapshot']>;
const notes = (s: Snap) => Object.values(s.elements).filter((e) => e.kind === 'note');
const lanes = (s: Snap) => notes(s).map((n) => (n as { lane: string }).lane).sort();
const layerCount = (s: Snap) => Object.keys(s.layers).length;

describe('appendTranscription', () => {
  it('replaces the content of an empty jot (no warning case)', () => {
    const handle = dslToMutable(createBlankJot());
    expect(notes(handle.snapshot()).length).toBe(0);

    const tx = makeJot(['k', 's'], { bpm: 140, tempoBpm: 140 });
    const result = appendTranscription(handle.doc, handle.model, tx, { idPrefix: 't' });

    expect(result.mode).toBe('replace');
    expect(result.hadNotes).toBe(false);
    const s = handle.snapshot();
    expect(s.bars.length).toBe(2);
    expect(lanes(s)).toEqual(['k', 's']);
    expect(layerCount(s)).toBe(1);
    expect(s.bpm).toBe(140);
    expect(Object.keys(s.tempoEvents).length).toBe(1);
  });

  it('appends a new layer, preserving existing notes (same grid)', () => {
    const handle = dslToMutable(makeJot(['k', 'k'], { bpm: 120 }));
    const tx = makeJot(['s', 's'], { bpm: 120 });
    const result = appendTranscription(handle.doc, handle.model, tx, { idPrefix: 't' });

    expect(result.mode).toBe('append');
    expect(result.hadNotes).toBe(true);
    const s = handle.snapshot();
    // Both kicks kept + both snares added.
    expect(lanes(s)).toEqual(['k', 'k', 's', 's']);
    expect(s.bars.length).toBe(2); // reused
    expect(layerCount(s)).toBe(2); // existing + transcription
  });

  it('extends the grid to the longer of the two (transcription longer)', () => {
    const handle = dslToMutable(makeJot(['k', 'k'])); // 2 bars
    const tx = makeJot(['s', 's', 's', 's']); // 4 bars
    appendTranscription(handle.doc, handle.model, tx, { idPrefix: 't' });

    const s = handle.snapshot();
    expect(s.bars.length).toBe(4);
    expect(lanes(s)).toEqual(['k', 'k', 's', 's', 's', 's']);
  });

  it('keeps existing notes past the transcription grid (existing longer)', () => {
    const handle = dslToMutable(makeJot(['k', 'k', 'k', 'k'])); // 4 bars
    const tx = makeJot(['s', 's']); // 2 bars
    appendTranscription(handle.doc, handle.model, tx, { idPrefix: 't' });

    const s = handle.snapshot();
    expect(s.bars.length).toBe(4); // none dropped
    expect(lanes(s)).toEqual(['k', 'k', 'k', 'k', 's', 's']);
  });

  it('replaces existing tempo events and reports how many were dropped', () => {
    const handle = dslToMutable(makeJot(['k'], { bpm: 100, tempoBpm: 100 }));
    const tx = makeJot(['s'], { bpm: 90, tempoBpm: 90 });
    const result = appendTranscription(handle.doc, handle.model, tx, { idPrefix: 't' });

    expect(result.replacedTempoCount).toBe(1);
    const s = handle.snapshot();
    expect(s.bpm).toBe(90);
    const events = Object.values(s.tempoEvents);
    expect(events.length).toBe(1);
    expect((events[0] as { bpm: number }).bpm).toBe(90);
  });

  it('labels the inserted layer', () => {
    const handle = dslToMutable(makeJot(['k']));
    const tx = makeJot(['s']);
    appendTranscription(handle.doc, handle.model, tx, { idPrefix: 't', layerName: 'guitar.wav' });

    const s = handle.snapshot();
    const named = Object.values(s.layers).filter((l) => (l as { name?: string }).name === 'guitar.wav');
    expect(named.length).toBe(1);
  });
});
