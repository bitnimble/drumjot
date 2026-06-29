import { describe, expect, test } from 'bun:test';
import { JotTimeline } from 'src/editing/playback/timeline';
import { LyricsStore, audioSecToBeat } from '../store';

/**
 * Build a synthetic `JotTimeline` from a list of (`startSec`,
 * `durationSec`) tuples and a parallel `beats` list. The rendered jot
 * reference is intentionally `undefined`; `audioSecToBeat` reads only
 * the `bars` array, so the test doesn't need a real laid-out jot.
 */
function fakeTimeline(
  rows: readonly { startSec: number; durationSec: number }[],
): JotTimeline {
  return {
    totalDurationSec: rows.reduce((acc, r) => Math.max(acc, r.startSec + r.durationSec), 0),
    bars: rows.map((r) => ({ startSec: r.startSec, durationSec: r.durationSec })),
    tempos: [],
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
    // songLeadIn = -1: audio 0 = jot time -1, the lead-in's start.
    expect(audioSecToBeat(0, timeline, -1, beats)).toBe(0);
    // audio 1s = jot time 0 = end of lead-in / start of bar 1 = 4 beats.
    expect(audioSecToBeat(1, timeline, -1, beats)).toBe(4);
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
  test('add returns a fresh id each call', () => {
    const s = new LyricsStore();
    expect(s.hasAnyLyrics).toBe(false);
    const a = s.add([{ startSec: 0, text: 'one' }], {
      source: 'lrclib',
      sourceLabel: 'LRCLIB · A - X',
    });
    const b = s.add([{ startSec: 0, text: 'two' }], {
      source: 'lrclib',
      sourceLabel: 'LRCLIB · B - Y',
    });
    expect(a).not.toBe(b);
    expect(s.hasAnyLyrics).toBe(true);
    expect(s.trackIds).toEqual([a, b]);
  });

  test('add disambiguates a duplicate sourceLabel with (2), (3) ...', () => {
    const s = new LyricsStore();
    const a = s.add([], { source: 'file', sourceLabel: 'File · song.lrc' });
    const b = s.add([], { source: 'file', sourceLabel: 'File · song.lrc' });
    const c = s.add([], { source: 'file', sourceLabel: 'File · song.lrc' });
    expect(s.get(a)?.sourceLabel).toBe('File · song.lrc');
    expect(s.get(b)?.sourceLabel).toBe('File · song.lrc (2)');
    expect(s.get(c)?.sourceLabel).toBe('File · song.lrc (3)');
  });

  test('add with a unique label leaves it unchanged', () => {
    const s = new LyricsStore();
    s.add([], { source: 'file', sourceLabel: 'File · a.lrc' });
    const b = s.add([], { source: 'file', sourceLabel: 'File · b.lrc' });
    expect(s.get(b)?.sourceLabel).toBe('File · b.lrc');
  });

  test('add does not mutate other tracks', () => {
    const s = new LyricsStore();
    const a = s.add([{ startSec: 0, text: 'first' }], {
      source: 'lrclib',
      sourceLabel: 'A',
    });
    s.setOffsetSec(a, 1.25);
    const before = { ...s.get(a)! };
    s.add([{ startSec: 0, text: 'second' }], { source: 'file', sourceLabel: 'B' });
    expect(s.get(a)).toEqual(before);
  });

  test('remove drops one track; others keep their offsets and lines', () => {
    const s = new LyricsStore();
    const a = s.add([{ startSec: 0, text: 'a' }], { source: 'file', sourceLabel: 'A' });
    const b = s.add([{ startSec: 0, text: 'b' }], { source: 'file', sourceLabel: 'B' });
    s.setOffsetSec(b, 2);
    s.remove(a);
    expect(s.trackIds).toEqual([b]);
    expect(s.get(b)?.offsetSec).toBe(2);
    expect(s.get(b)?.lines[0].text).toBe('b');
  });

  test('clear drops every track', () => {
    const s = new LyricsStore();
    s.add([], { source: 'file', sourceLabel: 'A' });
    s.add([], { source: 'file', sourceLabel: 'B' });
    s.clear();
    expect(s.hasAnyLyrics).toBe(false);
    expect(s.trackIds).toEqual([]);
  });

  test('setOffsetSec only mutates the targeted track and clamps to ±60s', () => {
    const s = new LyricsStore();
    const a = s.add([], { source: 'file', sourceLabel: 'A' });
    const b = s.add([], { source: 'file', sourceLabel: 'B' });
    s.setOffsetSec(a, 999);
    expect(s.get(a)?.offsetSec).toBe(60);
    expect(s.get(b)?.offsetSec).toBe(0);
    s.setOffsetSec(a, -999);
    expect(s.get(a)?.offsetSec).toBe(-60);
    s.setOffsetSec(a, Number.NaN);
    // NaN rejected; previous value preserved.
    expect(s.get(a)?.offsetSec).toBe(-60);
  });

  test('setOffsetSec on unknown id is a no-op', () => {
    const s = new LyricsStore();
    s.setOffsetSec('lyrics-99999', 5);
    expect(s.trackIds).toEqual([]);
  });

  test('replace preserves offsetSec, source, and label by default', () => {
    const s = new LyricsStore();
    const a = s.add([{ startSec: 0, text: 'line' }], {
      source: 'lrclib',
      sourceLabel: 'LRCLIB · X - Y',
    });
    s.setOffsetSec(a, 3.5);
    s.replace(a, [{ startSec: 0, text: 'new' }]);
    const t = s.get(a)!;
    expect(t.lines[0].text).toBe('new');
    expect(t.offsetSec).toBe(3.5);
    expect(t.source).toBe('lrclib');
    expect(t.sourceLabel).toBe('LRCLIB · X - Y');
  });

  test('replace can override sourceLabel without touching offset', () => {
    const s = new LyricsStore();
    const a = s.add([], { source: 'plaintext', sourceLabel: 'Plain text' });
    s.setOffsetSec(a, -1);
    s.replace(a, [{ startSec: 0, text: 'l' }], { sourceLabel: 'Whisper aligned' });
    const t = s.get(a)!;
    expect(t.sourceLabel).toBe('Whisper aligned');
    expect(t.offsetSec).toBe(-1);
    expect(t.source).toBe('plaintext');
  });

  test('replace on unknown id is a no-op', () => {
    const s = new LyricsStore();
    s.replace('lyrics-99999', [{ startSec: 0, text: 'x' }]);
    expect(s.trackIds).toEqual([]);
  });
});
