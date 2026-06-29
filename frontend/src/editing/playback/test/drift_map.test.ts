import { describe, expect, it } from 'bun:test';
import { buildDriftMap } from 'src/editing/playback/drift_map';
import { BarTiming } from 'src/editing/playback/timeline';

function bars(specs: [start: number, dur: number, drift: number][]): BarTiming[] {
  return specs.map(([startSec, durationSec, driftSec]) => ({ startSec, durationSec, driftSec }));
}

describe('buildDriftMap', () => {
  describe('no drift (the common, metronomic case)', () => {
    const map = buildDriftMap(bars([[0, 2, 0], [2, 2, 0], [4, 2, 0]]), -1);

    it('collapses to the flat jot - songLeadIn mapping', () => {
      expect(map.hasDrift).toBe(false);
      expect(map.jotToMedia(0)).toBe(1); // 0 - (-1)
      expect(map.jotToMedia(3)).toBe(4);
      expect(map.mediaToJot(4)).toBe(3);
    });

    it('is exact even outside the bar range (lead-in scrub)', () => {
      expect(map.jotToMedia(-1)).toBe(0); // audio t=0 at jot songLeadIn
      expect(map.mediaToJot(0)).toBe(-1);
    });
  });

  describe('with a sustained +0.1s drift from bar 1 onward', () => {
    // 3 bars of 2s; the real downbeat of bars 1 and 2 sits 0.1s late. songLeadIn
    // 0 so media == drift-adjusted jot.
    const map = buildDriftMap(bars([[0, 2, 0], [2, 2, 0.1], [4, 2, 0.1]]), 0);

    it('reports drift', () => {
      expect(map.hasDrift).toBe(true);
    });

    it('maps a drifted downbeat to its real recorded time', () => {
      // bar 1 downbeat: grid 2s, real 2.1s.
      expect(map.jotToMedia(2)).toBeCloseTo(2.1, 9);
      // bar 2 downbeat: grid 4s, real 4.1s (continuous with bar 1's media end).
      expect(map.jotToMedia(4)).toBeCloseTo(4.1, 9);
    });

    it('stretches the bar before the drift step (bar 0 → 2.1s of media)', () => {
      // Halfway through bar 0's jot span maps halfway through its 2.1s media span.
      expect(map.jotToMedia(1)).toBeCloseTo(1.05, 9);
    });

    it('round-trips jot → media → jot across the drift step', () => {
      for (const j of [0.5, 1, 2, 2.5, 3.9, 4, 5.5]) {
        expect(map.mediaToJot(map.jotToMedia(j))).toBeCloseTo(j, 9);
      }
    });

    it('is continuous at bar boundaries (no gap/overlap in media)', () => {
      // bar 0's media end == bar 1's media start.
      const justBefore = map.jotToMedia(2 - 1e-7);
      const at = map.jotToMedia(2);
      expect(Math.abs(at - justBefore)).toBeLessThan(1e-4);
    });
  });

  it('honours songLeadIn together with drift', () => {
    // 1s pre-roll (songLeadIn -1) + a 0.05s drift on bar 1.
    const map = buildDriftMap(bars([[0, 2, 0], [2, 2, 0.05]]), -1);
    // bar 1 real downbeat: grid 2s + drift 0.05s, then + 1s pre-roll → media 3.05.
    expect(map.jotToMedia(2)).toBeCloseTo(3.05, 9);
    expect(map.mediaToJot(3.05)).toBeCloseTo(2, 9);
  });

  it('handles an empty timeline as the flat mapping', () => {
    const map = buildDriftMap([], -0.5);
    expect(map.hasDrift).toBe(false);
    expect(map.jotToMedia(2)).toBe(2.5);
  });
});
