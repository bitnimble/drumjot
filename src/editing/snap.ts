import type { GridLineSettings } from 'src/settings/settings_store';

/**
 * Grid-snapping in **beat space** (quarter-note units). Each grid-line family
 * subdivides a beat into a fixed number of equal parts; snapping with several
 * families enabled targets the **union** of their lines, so 16ths and 8th-
 * triplets both being visible means a note can land on either.
 *
 * Because storage and movement are all in tempo-independent beat units, the
 * same snap function serves every bar and tempo: a 16th is 0.25 beats whether
 * the bar is fast or slow.
 */

/** Subdivisions-per-beat for each grid-line family (see {@link GridLineSettings}). */
export const GRID_FAMILY_PER_BEAT: Record<keyof GridLineSettings, number> = {
  mainBeat: 1,
  subBeat16: 4,
  subBeatQuarterTriplet: 1.5,
  subBeatTriplet: 3,
  subBeat48: 12,
};

/** The per-beat divisors for the currently-enabled grid families. */
export function enabledDivisors(grid: GridLineSettings): number[] {
  return (Object.keys(GRID_FAMILY_PER_BEAT) as (keyof GridLineSettings)[])
    .filter((k) => grid[k])
    .map((k) => GRID_FAMILY_PER_BEAT[k]);
}

/**
 * Snap `beat` to the nearest line across all `divisors` (each a
 * subdivisions-per-beat count), clamped to `[0, maxBeat]`. With no divisors
 * the beat is returned unchanged (snapping has no grid to target). Computed
 * per-family without enumerating the whole bar: each family's nearest line is
 * `round(beat * perBeat) / perBeat`, and the closest across families wins.
 */
export function snapBeat(beat: number, divisors: readonly number[], maxBeat = Infinity): number {
  if (divisors.length === 0) return beat;
  let best = beat;
  let bestDist = Infinity;
  for (const perBeat of divisors) {
    const cand = Math.min(Math.max(Math.round(beat * perBeat) / perBeat, 0), maxBeat);
    const dist = Math.abs(beat - cand);
    if (dist < bestDist) {
      bestDist = dist;
      best = cand;
    }
  }
  return best;
}
