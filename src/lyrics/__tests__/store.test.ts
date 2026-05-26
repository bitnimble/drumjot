import { describe, expect, test } from 'bun:test';
import { JotTimeline } from 'src/playback';
import { LyricsStore, audioSecToBeat } from '../store';

/**
 * Build a synthetic `JotTimeline` from a list of (`startSec`,
 * `durationSec`) tuples and a parallel `beats` list. The rendered jot
 * reference is intentionally `undefined`; `audioSecToBeat` reads only
 * the `bars` array, so the test doesn't need a real RenderedJot.
 */
function fakeTimeline(
  rows: readonly { startSec: number; durationSec: number }[],
): JotTimeline {
  return {
    totalDurationSec: rows.reduce((acc, r) => Math.max(acc, r.startSec + r.durationSec), 0),
    bars: rows.map((r) => ({ startSec: r.startSec, durationSec: r.durationSec })),
    rendered: undefined,
  };
}

describe('audioSecToBeat', () => {
  // 4 bars, each 2s long covering 4 beats; steady 120 bpm in 4/4.
  const steadyTimeline = fakeTimeline([
    { startSec: 0, durationSec: 2 },
    { startSec: 2, durationSec: 2 },
    { startSec: 4, durationSec: 2 },
    { startSec: 6, durationSec: 2 },
  ]);
  const steadyBeats = [4, 4, 4, 4];

  test('beat 0 = bar 0 startSec', () => {
    expect(audioSecToBeat(0, steadyTimeline, 0, steadyBeats)).toBe(0);
  });

  test('linear within a bar', () => {
    // 1s into bar 0 (half-way) -> 2 beats in.
    expect(audioSecToBeat(1, steadyTimeline, 0, steadyBeats)).toBe(2);
    // Halfway through bar 2 (5s) -> 8 + 2 = 10 beats.
    expect(audioSecToBeat(5, steadyTimeline, 0, steadyBeats)).toBe(10);
  });

  test('handles a per-bar tempo change via different durations / beats', () => {
    const timeline = fakeTimeline([
      { startSec: 0, durationSec: 2 },
      { startSec: 2, durationSec: 1 }, // 2x tempo
    ]);
    const beats = [4, 4];
    // End of bar 0 → 4 beats.
    expect(audioSecToBeat(2, timeline, 0, beats)).toBe(4);
    // Halfway through the faster bar (audio 2.5s = half of bar 1) → 4 + 2.
    expect(audioSecToBeat(2.5, timeline, 0, beats)).toBe(6);
  });

  test('lead-in (negative startSec) maps audio 0 to a positive beat', () => {
    // 1 lead-in bar (-1s..0s, 4 beats) + 1 drum bar (0..2s, 4 beats).
    const timeline = fakeTimeline([
      { startSec: -1, durationSec: 1 },
      { startSec: 0, durationSec: 2 },
    ]);
    const beats = [4, 4];
    // drumsT0Sec = 1: audio 0 = jot time -1, the lead-in's start.
    expect(audioSecToBeat(0, timeline, 1, beats)).toBe(0);
    // audio 1s = jot time 0 = end of lead-in / start of bar 1 = 4 beats.
    expect(audioSecToBeat(1, timeline, 1, beats)).toBe(4);
  });

  test('returns undefined for times outside the timeline', () => {
    expect(audioSecToBeat(-1, steadyTimeline, 0, steadyBeats)).toBeUndefined();
    expect(audioSecToBeat(8, steadyTimeline, 0, steadyBeats)).toBeUndefined();
  });

  test('returns undefined when bars and beats arrays disagree in length', () => {
    expect(audioSecToBeat(1, steadyTimeline, 0, [4, 4])).toBeUndefined();
  });

  test('empty timeline returns undefined', () => {
    expect(audioSecToBeat(0, fakeTimeline([]), 0, [])).toBeUndefined();
  });
});

describe('LyricsStore', () => {
  test('load + clear lifecycle', () => {
    const s = new LyricsStore();
    expect(s.hasLyrics).toBe(false);
    s.load(
      [{ startSec: 0, text: 'hi' }],
      { source: 'lrclib', sourceLabel: 'LRCLIB · Test - Artist' },
    );
    expect(s.hasLyrics).toBe(true);
    expect(s.sourceLabel).toBe('LRCLIB · Test - Artist');
    expect(s.offsetSec).toBe(0);
    s.setOffsetSec(2.5);
    expect(s.offsetSec).toBe(2.5);
    s.clear();
    expect(s.hasLyrics).toBe(false);
    expect(s.sourceLabel).toBeUndefined();
    expect(s.offsetSec).toBe(0);
  });

  test('a new load resets the offset', () => {
    const s = new LyricsStore();
    s.load([{ startSec: 0, text: 'a' }], { source: 'file', sourceLabel: 'File · a.lrc' });
    s.setOffsetSec(3);
    s.load([{ startSec: 0, text: 'b' }], { source: 'file', sourceLabel: 'File · b.lrc' });
    expect(s.offsetSec).toBe(0);
  });

  test('setOffsetSec clamps to ±60s', () => {
    const s = new LyricsStore();
    s.setOffsetSec(999);
    expect(s.offsetSec).toBe(60);
    s.setOffsetSec(-999);
    expect(s.offsetSec).toBe(-60);
    s.setOffsetSec(Number.NaN);
    // NaN is rejected; previous value is preserved.
    expect(s.offsetSec).toBe(-60);
  });

  test('activeLineIndexAt honours the current offset', () => {
    const s = new LyricsStore();
    s.load(
      [
        { startSec: 0, text: 'first' },
        { startSec: 10, text: 'second' },
      ],
      { source: 'lrclib', sourceLabel: 'LRCLIB · x - y' },
    );
    expect(s.activeLineIndexAt(11)).toBe(1);
    s.setOffsetSec(2);
    // Line 1 now starts at audio t=12, so t=11 still belongs to line 0.
    expect(s.activeLineIndexAt(11)).toBe(0);
    expect(s.activeLineIndexAt(12)).toBe(1);
  });
});
