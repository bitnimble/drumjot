import { describe, expect, it } from 'bun:test';
import { JotTimeline } from 'src/editing/playback/timeline';
import { formatPlayheadTime, playheadBarBeat } from '../playhead_label';

describe('formatPlayheadTime', () => {
  it('formats whole seconds as M:SS.cc', () => {
    expect(formatPlayheadTime(0)).toBe('0:00.00');
    expect(formatPlayheadTime(5)).toBe('0:05.00');
    expect(formatPlayheadTime(65)).toBe('1:05.00');
  });

  it('includes centiseconds, truncated', () => {
    expect(formatPlayheadTime(1.239)).toBe('0:01.23');
    expect(formatPlayheadTime(72.5)).toBe('1:12.50');
  });

  it('prefixes a minus sign for negative (lead-in) times', () => {
    expect(formatPlayheadTime(-2.5)).toBe('-0:02.50');
  });
});

// Minimal stand-in for the fields `playheadBarBeat` reads: per-bar audio
// timings + the rendered structure's first layer bars (index + beat count).
// Two 4/4 bars, each 2s long, lead-in handled by the caller.
function fakeTimeline(): JotTimeline {
  return {
    bars: [
      { startSec: 0, durationSec: 2 },
      { startSec: 2, durationSec: 2 },
    ],
    rendered: {
      layers: [
        {
          bars: [
            { index: 1, tsCount: 4 },
            { index: 2, tsCount: 4 },
          ],
        },
      ],
    },
  } as unknown as JotTimeline;
}

describe('playheadBarBeat', () => {
  it('returns null for an empty timeline', () => {
    expect(playheadBarBeat({ bars: [] } as unknown as JotTimeline, 0)).toBeNull();
  });

  it('reports the downbeat of bar 1 at t=0', () => {
    expect(playheadBarBeat(fakeTimeline(), 0)).toBe('Bar 1, 1.00b');
  });

  it('reports the midpoint of bar 1 (4/4, 2s) as beat 3', () => {
    // 1s into a 2s 4/4 bar → 1 + 0.5*4 = beat 3.00.
    expect(playheadBarBeat(fakeTimeline(), 1)).toBe('Bar 1, 3.00b');
  });

  it('rolls into bar 2 at its start', () => {
    expect(playheadBarBeat(fakeTimeline(), 2)).toBe('Bar 2, 1.00b');
  });

  it('truncates rather than rounding up past the bar end', () => {
    // Just before bar 1 ends (1.999s) must stay in bar 1, not round to bar 2.
    const r = playheadBarBeat(fakeTimeline(), 1.999);
    expect(r?.startsWith('Bar 1,')).toBe(true);
  });

  it('pins to the last bar past the end of the score', () => {
    expect(playheadBarBeat(fakeTimeline(), 100)).toBe('Bar 2, 4.99b');
  });
});
