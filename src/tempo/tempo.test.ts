import { describe, expect, it } from 'bun:test';
import { msOffsetToBeats } from 'src/tempo';

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
