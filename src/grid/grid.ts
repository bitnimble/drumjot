/**
 * Grid density (1/N-of-a-whole-note) assumed when a jot carries no
 * explicit `gridDivision`. 48 = 12 slots per quarter
 * note, the value `from_midi` has historically used; LCM(4, 3) per
 * quarter so both straight 16ths and triplet 8ths are representable.
 */
export const DEFAULT_GRID_DIVISION = 48;

/**
 * Anything carrying a `gridDivision` register, the reactive `MutableJot`
 * (which lifts it to a top-level field) satisfies this directly. Kept
 * structural so callers can pass the reactive document without importing a
 * heavier wrapper here.
 */
type HasGridDivision = { gridDivision?: number };

/** The grid density a jot was produced at; falls back to the default. */
export function gridDivisionFor(jot: HasGridDivision): number {
  return jot.gridDivision ?? DEFAULT_GRID_DIVISION;
}

/** Slots per quarter-note beat for a jot's grid (e.g. 12 for the 1/48 default). */
export function slotsPerQuarter(jot: HasGridDivision): number {
  return gridDivisionFor(jot) / 4;
}
