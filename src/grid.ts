import type { Metadata } from 'src/dsl';

/**
 * Grid density (1/N-of-a-whole-note) assumed when a jot carries no
 * explicit `globalMetadata.gridDivision`. 48 = 12 slots per quarter
 * note, the value `from_midi` has historically used; LCM(4, 3) per
 * quarter so both straight 16ths and triplet 8ths are representable.
 */
export const DEFAULT_GRID_DIVISION = 48;

/**
 * Anything carrying `globalMetadata`, a raw {@link import('src/dsl').Jot}
 * or a `RenderedJot`. Kept structural so both work without importing the
 * rendered wrapper here.
 */
type HasGlobalMetadata = { globalMetadata: Metadata };

/** The grid density a jot was produced at; falls back to the default. */
export function gridDivisionFor(jot: HasGlobalMetadata): number {
  return jot.globalMetadata.gridDivision ?? DEFAULT_GRID_DIVISION;
}

/** Slots per quarter-note beat for a jot's grid (e.g. 12 for the 1/48 default). */
export function slotsPerQuarter(jot: HasGlobalMetadata): number {
  return gridDivisionFor(jot) / 4;
}
