import { describe, expect, it } from 'bun:test';
import {
  beatToSecWithinBar,
  buildBarTempos,
  msOffsetToBeats,
  secToBeatWithinBar,
  type BarTempos,
} from 'src/schema/dsl/tempo';
import type { Jot } from 'src/schema/dsl/dsl';

/** Minimal Jot carrying just the tempo inputs `buildBarTempos` reads. */
function tempoJot(bpm: number, tempoEvents: Jot['tempoEvents']): Jot {
  return { globalMetadata: { bpm }, layers: [], tempoEvents } as unknown as Jot;
}

/**
 * Ground-truth seconds to traverse `[0, beat]` of a linear-in-time tempo
 * ramp (bpm rises at a constant rate per second, so bpm² is linear in beat),
 * by brute-force midpoint integration of `60 / bpm(b)`.
 */
function numericRampSec(bpm0: number, bpm1: number, L: number, beat: number): number {
  const steps = 200_000;
  const h = beat / steps;
  let s = 0;
  for (let i = 0; i < steps; i++) {
    const b = (i + 0.5) * h;
    const bpm = Math.sqrt(bpm0 * bpm0 + (bpm1 * bpm1 - bpm0 * bpm0) * (b / L));
    s += (60 / bpm) * h;
  }
  return s;
}

describe('msOffsetToBeats', () => {
  it('converts ms to beats at the local tempo (120 BPM -> 0.5 s/beat)', () => {
    // 25 ms at 0.5 s/beat = 0.05 beats.
    expect(msOffsetToBeats(25, 0.5)).toBeCloseTo(0.05, 9);
  });

  it('scales with tempo (60 BPM -> 1 s/beat)', () => {
    expect(msOffsetToBeats(25, 1)).toBeCloseTo(0.025, 9);
  });

  it('preserves sign for negative (behind-the-beat) offsets', () => {
    expect(msOffsetToBeats(-30, 0.5)).toBeCloseTo(-0.06, 9);
  });

  it('returns 0 for a degenerate (non-positive) sec-per-beat', () => {
    expect(msOffsetToBeats(25, 0)).toBe(0);
    expect(msOffsetToBeats(25, -1)).toBe(0);
  });
});

describe('beatToSecWithinBar with a linear-in-time tempo ramp', () => {
  // A ramp segment carries its endpoint BPMs (`bpm` at startBeat, `endBpm`
  // at endBeat); bpm² varies linearly with beat (constant accel per second).
  const rampTempos: BarTempos = {
    durationSec: (120 * 8) / (60 + 120), // 120·L/(bpm0+bpm1)
    segments: [{ startBeat: 0, endBeat: 8, bpm: 60, endBpm: 120 }],
  };

  it('matches the numeric integral of 60/bpm(b) along the ramp', () => {
    for (const beat of [0, 1, 2, 4, 6, 8]) {
      expect(beatToSecWithinBar(rampTempos, beat)).toBeCloseTo(
        numericRampSec(60, 120, 8, beat),
        4
      );
    }
  });

  it('reaches the average-tempo duration at the ramp end (60->120 over 8 = 5.333s)', () => {
    // 8 beats at the average 90 BPM = 8 / 90 * 60 = 5.3333s.
    expect(beatToSecWithinBar(rampTempos, 8)).toBeCloseTo(5.3333333, 6);
  });

  it('treats endBpm===bpm as a plain constant segment', () => {
    const flat: BarTempos = {
      durationSec: 4,
      segments: [{ startBeat: 0, endBeat: 8, bpm: 120, endBpm: 120 }],
    };
    // 4 beats into a constant 120 BPM (0.5 s/beat) = 2s.
    expect(beatToSecWithinBar(flat, 4)).toBeCloseTo(2, 9);
  });
});

describe('secToBeatWithinBar (inverse of beatToSecWithinBar)', () => {
  const rampTempos: BarTempos = {
    durationSec: (120 * 8) / (60 + 120),
    segments: [{ startBeat: 0, endBeat: 8, bpm: 60, endBpm: 120 }],
  };

  it('round-trips beat -> sec -> beat along a ramp', () => {
    for (const beat of [0, 1.5, 3, 5, 8]) {
      const sec = beatToSecWithinBar(rampTempos, beat);
      expect(secToBeatWithinBar(rampTempos, sec)).toBeCloseTo(beat, 6);
    }
  });

  it('round-trips through a plain constant segment', () => {
    const flat: BarTempos = {
      durationSec: 4,
      segments: [{ startBeat: 0, endBeat: 8, bpm: 120 }],
    };
    expect(secToBeatWithinBar(flat, 1.5)).toBeCloseTo(3, 9); // 1.5s at 0.5 s/beat
  });
});

describe('buildBarTempos with a gradual tempo ramp', () => {
  // bpm at the midpoint (global beat 4) of a 60->120 ramp over 8 beats:
  // sqrt(60² + (120²−60²)·4/8) = sqrt(9000).
  const MID = Math.sqrt(9000); // ≈ 94.868

  it('splits a multi-bar ramp into per-bar ramp segments', () => {
    const jot = tempoJot(60, [{ barIndex: 0, beat: 0, bpm: { start: 60, end: 120, duration: 8 } }]);
    const tempos = buildBarTempos(jot, [{ beats: 4 }, { beats: 4 }, { beats: 4 }]);

    // Bar 0: ramp 60 -> mid over its 4 beats.
    expect(tempos[0].segments).toHaveLength(1);
    expect(tempos[0].segments[0].bpm).toBeCloseTo(60, 6);
    expect(tempos[0].segments[0].endBpm).toBeCloseTo(MID, 6);
    // Bar 1: ramp mid -> 120.
    expect(tempos[1].segments[0].bpm).toBeCloseTo(MID, 6);
    expect(tempos[1].segments[0].endBpm).toBeCloseTo(120, 6);
    // Bar 2: constant 120 (post-ramp), no ramp.
    expect(tempos[2].segments[0].bpm).toBeCloseTo(120, 6);
    expect(tempos[2].segments[0].endBpm ?? 120).toBeCloseTo(120, 6);
  });

  it('makes the ramp bars sum to the average-tempo duration', () => {
    const jot = tempoJot(60, [{ barIndex: 0, beat: 0, bpm: { start: 60, end: 120, duration: 8 } }]);
    const tempos = buildBarTempos(jot, [{ beats: 4 }, { beats: 4 }]);
    // 8 beats at average 90 BPM = 5.3333s across the two bars.
    expect(tempos[0].durationSec + tempos[1].durationSec).toBeCloseTo(5.3333333, 5);
  });

  it('handles a sub-bar ramp: ramp then constant within the same bar', () => {
    // 4/4 bar, ramp 120 -> 180 over the first 2 beats, constant 180 after.
    const jot = tempoJot(120, [
      { barIndex: 0, beat: 0, bpm: { start: 120, end: 180, duration: 2 } },
    ]);
    const tempos = buildBarTempos(jot, [{ beats: 4 }]);
    expect(tempos[0].segments).toHaveLength(2);
    expect(tempos[0].segments[0]).toMatchObject({ startBeat: 0, endBeat: 2, bpm: 120 });
    expect(tempos[0].segments[0].endBpm).toBeCloseTo(180, 6);
    expect(tempos[0].segments[1]).toMatchObject({ startBeat: 2, endBeat: 4, bpm: 180 });
    expect(tempos[0].segments[1].endBpm ?? 180).toBeCloseTo(180, 6);
  });
});
