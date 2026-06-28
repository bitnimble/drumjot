import { describe, expect, it } from 'bun:test';
import { TranscriptionSchema } from 'src/midi/transcription_schema';

const valid = {
  format: 1,
  tempoMap: {
    initial_bpm: 120,
    events: [
      { tick: 1234, bpm: 150 },
      { tick: 5678, bpm: { start: 120, end: 180, end_tick: 9000 }, shape: 'linear' },
    ],
  },
  barDrift: [0, 0, 0.03, 0.03],
};

describe('TranscriptionSchema', () => {
  it('accepts a well-formed container (step + ramp events, barDrift)', () => {
    const r = TranscriptionSchema.safeParse(valid);
    expect(r.success).toBe(true);
    if (r.success) {
      expect(r.data.tempoMap.events).toHaveLength(2);
      expect(r.data.barDrift).toEqual([0, 0, 0.03, 0.03]);
    }
  });

  it('treats barDrift as optional (older bundles)', () => {
    const { barDrift, ...noDrift } = valid;
    void barDrift;
    expect(TranscriptionSchema.safeParse(noDrift).success).toBe(true);
  });

  it('rejects an unknown format so the loader falls back to MIDI', () => {
    expect(TranscriptionSchema.safeParse({ ...valid, format: 2 }).success).toBe(false);
  });

  it('rejects a ramp event missing end_tick', () => {
    const bad = {
      ...valid,
      tempoMap: { initial_bpm: 120, events: [{ tick: 0, bpm: { start: 120, end: 180 } }] },
    };
    expect(TranscriptionSchema.safeParse(bad).success).toBe(false);
  });

  it('rejects an out-of-vocabulary ramp shape', () => {
    const bad = {
      ...valid,
      tempoMap: {
        initial_bpm: 120,
        events: [{ tick: 0, bpm: { start: 120, end: 180, end_tick: 900 }, shape: 'sine' }],
      },
    };
    expect(TranscriptionSchema.safeParse(bad).success).toBe(false);
  });

  it('rejects the not-yet-supported exponential/logarithmic shapes (linear only today)', () => {
    for (const shape of ['exponential', 'logarithmic'] as const) {
      const r = TranscriptionSchema.safeParse({
        ...valid,
        tempoMap: {
          initial_bpm: 120,
          events: [{ tick: 0, bpm: { start: 120, end: 180, end_tick: 900 }, shape }],
        },
      });
      expect(r.success).toBe(false);
    }
  });
});
