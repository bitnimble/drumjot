import { describe, expect, it } from 'bun:test';
import {
  computeBarLayouts,
  noteMarksEqual,
  type MinimapBarTiming,
  type NoteMark,
} from './minimap_layout';

const bar = (startSec: number, durationSec: number): MinimapBarTiming => ({ startSec, durationSec });

describe('computeBarLayouts', () => {
  it('returns an empty result when width is non-positive', () => {
    const r = computeBarLayouts([bar(0, 1)], [{ beats: 4 }], 0);
    expect(r.hasContent).toBe(false);
    expect(r.bars).toEqual([]);
  });

  it('returns an empty result when either side has no bars', () => {
    expect(computeBarLayouts([], [{ beats: 4 }], 100).hasContent).toBe(false);
    expect(computeBarLayouts([bar(0, 1)], [], 100).hasContent).toBe(false);
  });

  it('returns an empty result when total duration is non-positive', () => {
    // Single zero-duration bar => total 0.
    const r = computeBarLayouts([bar(5, 0)], [{ beats: 4 }], 100);
    expect(r.hasContent).toBe(false);
  });

  it('maps two equal bars to halves of the width', () => {
    const r = computeBarLayouts([bar(0, 1), bar(1, 1)], [{ beats: 4 }, { beats: 4 }], 200);
    expect(r.hasContent).toBe(true);
    expect(r.totalDuration).toBe(2);
    expect(r.firstStartSec).toBe(0);
    expect(r.bars).toEqual([
      { x: 0, width: 100, beats: 4 },
      { x: 100, width: 100, beats: 4 },
    ]);
  });

  it('is offset-invariant (subtracts the first start)', () => {
    // Same shape as above but shifted to start at t=10.
    const r = computeBarLayouts([bar(10, 1), bar(11, 3)], [{ beats: 4 }, { beats: 12 }], 400);
    expect(r.firstStartSec).toBe(10);
    expect(r.totalDuration).toBe(4);
    expect(r.bars).toEqual([
      { x: 0, width: 100, beats: 4 },
      { x: 100, width: 300, beats: 12 },
    ]);
  });

  it('defaults missing structural beats to 0', () => {
    const r = computeBarLayouts([bar(0, 1), bar(1, 1)], [{ beats: 4 }], 200);
    expect(r.bars[1].beats).toBe(0);
  });
});

describe('noteMarksEqual', () => {
  const marks = (...xs: number[]): NoteMark[] => xs.map((x) => ({ x, color: '#fff' }));

  it('is true for the same reference', () => {
    const a = marks(1, 2);
    expect(noteMarksEqual(a, a)).toBe(true);
  });

  it('is true for content-identical lists', () => {
    expect(noteMarksEqual(marks(1, 2, 3), marks(1, 2, 3))).toBe(true);
  });

  it('is false on differing length', () => {
    expect(noteMarksEqual(marks(1, 2), marks(1, 2, 3))).toBe(false);
  });

  it('is false on a differing x', () => {
    expect(noteMarksEqual(marks(1, 2), marks(1, 9))).toBe(false);
  });

  it('is false on a differing color', () => {
    expect(noteMarksEqual([{ x: 1, color: '#fff' }], [{ x: 1, color: '#000' }])).toBe(false);
  });
});
