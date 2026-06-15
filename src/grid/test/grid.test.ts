import { describe, expect, it } from 'bun:test';
import { Jot } from 'src/schema/dsl/dsl';
import { DEFAULT_GRID_DIVISION, gridDivisionFor, slotsPerQuarter } from 'src/grid/grid';

function jotWith(gridDivision?: number): Jot {
  return {
    title: '',
    globalMetadata: { bpm: 120, time: { count: 4, unit: 4 }, ...(gridDivision !== undefined ? { gridDivision } : {}) },
    voices: [{ bars: [] }],
  };
}

describe('gridDivisionFor', () => {
  it('falls back to the default when unset', () => {
    expect(gridDivisionFor(jotWith())).toBe(DEFAULT_GRID_DIVISION);
    expect(DEFAULT_GRID_DIVISION).toBe(48);
  });

  it('reads the per-jot override', () => {
    expect(gridDivisionFor(jotWith(96))).toBe(96);
  });
});

describe('slotsPerQuarter', () => {
  it('derives 12 slots per quarter from the default 1/48 grid', () => {
    expect(slotsPerQuarter(jotWith())).toBe(12);
  });

  it('scales with the grid division', () => {
    expect(slotsPerQuarter(jotWith(96))).toBe(24);
  });
});
