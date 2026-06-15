import { describe, expect, it } from 'bun:test';
import type { LyricLine } from 'src/lyrics/lrc';
import type { JotTimeline } from 'src/editing/playback/timeline';
import { MIN_BEAT_WIDTH, positionLyricLines } from '../lyric_layout';

// Two 4-beat bars, each 2 s long: jot-time [0,2) → beats [0,4),
// [2,4) → beats [4,8). So audioSecToBeat is `sec * 2` over [0,4).
const TIMELINE: JotTimeline = {
  totalDurationSec: 4,
  bars: [
    { startSec: 0, durationSec: 2 },
    { startSec: 2, durationSec: 2 },
  ],
  rendered: undefined,
};
const STRUCT_BEATS = [4, 4];
const LAYER_BEATS = 8;

// Project just the comparable fields of each word cell (the `source`
// back-reference is identity-only, asserted separately where it matters).
const cells = (line: { wordPositions: { sourceIdx: number; text: string; beatOffset: number; beatWidth: number }[] | undefined }) =>
  line.wordPositions?.map(({ sourceIdx, text, beatOffset, beatWidth }) => ({
    sourceIdx,
    text,
    beatOffset,
    beatWidth,
  }));

function position(lines: LyricLine[], offsetSec = 0, songLeadIn = 0) {
  return positionLyricLines(lines, TIMELINE, songLeadIn, STRUCT_BEATS, offsetSec, LAYER_BEATS);
}

describe('positionLyricLines (word-less / LRCLIB-style)', () => {
  it('bounds a lone line by the whole layer', () => {
    const out = position([{ startSec: 0, text: 'hello' }]);
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ i: 0, text: 'hello', startBeat: 0, endBeat: LAYER_BEATS });
    expect(out[0].wordPositions).toBeUndefined();
  });

  it('bounds each line by the next line start', () => {
    const out = position([
      { startSec: 0, text: 'a' },
      { startSec: 2, text: 'b' },
    ]);
    expect(out.map((l) => [l.startBeat, l.endBeat])).toEqual([
      [0, 4],
      [4, 8],
    ]);
  });

  it('shifts positions by offsetSec', () => {
    const out = position([{ startSec: 0, text: 'x' }], 2);
    // 0s + 2s offset => jot-time 2s => beat 4.
    expect(out[0].startBeat).toBe(4);
  });

  it('accounts for the audio lead-in (songLeadIn)', () => {
    const out = position([{ startSec: 2, text: 'd' }], 0, -2);
    // 2s audio + (-2s songLeadIn) => jot-time 0 => beat 0.
    expect(out[0].startBeat).toBe(0);
  });
});

describe('positionLyricLines (blank / out-of-range)', () => {
  it('drops a blank line with no words', () => {
    expect(position([{ startSec: 0, text: '   ' }])).toEqual([]);
  });

  it('drops a line whose start falls outside the timeline', () => {
    // 10s is past the 4s timeline => audioSecToBeat undefined => skipped.
    expect(position([{ startSec: 10, text: 'late' }])).toEqual([]);
  });
});

describe('positionLyricLines (word-aligned)', () => {
  it('lays out each word cell relative to the line start', () => {
    const out = position([
      {
        startSec: 0,
        text: 'hi there',
        words: [
          { startSec: 0, endSec: 1, text: 'hi' },
          { startSec: 1, endSec: 2, text: 'there' },
        ],
      },
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].startBeat).toBe(0);
    expect(out[0].endBeat).toBe(4);
    expect(cells(out[0])).toEqual([
      { sourceIdx: 0, text: 'hi', beatOffset: 0, beatWidth: 2 },
      { sourceIdx: 1, text: 'there', beatOffset: 2, beatWidth: 2 },
    ]);
  });

  it('drops edge words out of range but preserves sourceIdx', () => {
    const out = position([
      {
        startSec: 0,
        text: 'pre in',
        words: [
          { startSec: -1, endSec: 0, text: 'pre' }, // before the timeline
          { startSec: 0, endSec: 1, text: 'in' },
        ],
      },
    ]);
    expect(cells(out[0])).toEqual([{ sourceIdx: 1, text: 'in', beatOffset: 0, beatWidth: 2 }]);
  });

  it('floors a non-positive word duration to MIN_BEAT_WIDTH', () => {
    const out = position([
      { startSec: 0, text: 'pt', words: [{ startSec: 0, endSec: 0, text: 'pt' }] },
    ]);
    expect(cells(out[0])![0].beatWidth).toBe(MIN_BEAT_WIDTH);
  });

  it('keeps the source word by reference for the debug tooltip', () => {
    const word = { startSec: 0, endSec: 1, text: 'hi' };
    const out = position([{ startSec: 0, text: 'hi', words: [word] }]);
    expect(out[0].wordPositions![0].source).toBe(word);
  });
});
