import { describe, expect, it } from 'bun:test';
import { buildLaneMap } from 'src/editing/score/note_geometry';

describe('buildLaneMap', () => {
  const order = ['h', 's', 'k', 't'];

  it('is identity when source and target lanes match', () => {
    const map = buildLaneMap(order, 's', 's');
    expect(order.map(map)).toEqual(order);
  });

  it('shifts every lane by the anchor row delta', () => {
    // anchor h -> k is +2 rows; each lane shifts down 2 (clamped at the end).
    const map = buildLaneMap(order, 'h', 'k');
    expect(map('h')).toBe('k');
    expect(map('s')).toBe('t');
    expect(map('k')).toBe('t'); // clamped at last row
  });

  it('shifts upward for a negative row delta', () => {
    const map = buildLaneMap(order, 't', 's'); // -2 rows
    expect(map('t')).toBe('s');
    expect(map('k')).toBe('h');
    expect(map('h')).toBe('h'); // clamped at first row
  });

  it('is identity when a lane is not in the order', () => {
    const map = buildLaneMap(order, 'x', 's');
    expect(map('h')).toBe('h');
  });
});
