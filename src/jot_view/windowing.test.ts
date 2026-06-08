import { describe, expect, it } from 'bun:test';
import { intersectsBeatRange } from './windowing';

describe('intersectsBeatRange', () => {
  const range = { startBeat: 10, endBeat: 20 };

  it('renders everything when windowing is disabled (null range)', () => {
    expect(intersectsBeatRange(null, 0, 4)).toBe(true);
    expect(intersectsBeatRange(null, 1000, 4)).toBe(true);
  });

  it('includes a bar fully inside the window', () => {
    expect(intersectsBeatRange(range, 12, 4)).toBe(true);
  });

  it('excludes a bar entirely left of the window', () => {
    // [4, 8] ends before startBeat 10.
    expect(intersectsBeatRange(range, 4, 4)).toBe(false);
  });

  it('excludes a bar entirely right of the window', () => {
    // [24, 28] starts after endBeat 20.
    expect(intersectsBeatRange(range, 24, 4)).toBe(false);
  });

  it('includes a bar straddling the left edge', () => {
    // [8, 12] crosses startBeat 10.
    expect(intersectsBeatRange(range, 8, 4)).toBe(true);
  });

  it('includes a bar straddling the right edge', () => {
    // [18, 22] crosses endBeat 20.
    expect(intersectsBeatRange(range, 18, 4)).toBe(true);
  });

  it('includes a bar flush against either edge (inclusive endpoints)', () => {
    // ends exactly at startBeat.
    expect(intersectsBeatRange(range, 6, 4)).toBe(true);
    // starts exactly at endBeat.
    expect(intersectsBeatRange(range, 20, 4)).toBe(true);
  });

  it('includes a wide bar that contains the whole window', () => {
    expect(intersectsBeatRange(range, 0, 100)).toBe(true);
  });
});
