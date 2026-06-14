/**
 * Pure DSL element metrics: layout beat-weights and the straightness
 * (dyadic-grid) test. No layout / pixel / MobX concerns; consumed by the
 * reactive converter (`src/schema/from_dsl.ts`) and the structure store.
 *
 * Pattern/repeat expansion used to live here too, but the reactive
 * document builds its element tree directly (patterns become `pattern`
 * leaves referencing a `PatternDef` by id, repeats are unrolled by the
 * converter), so the old DSL tree-rewriting is gone.
 */
import { Element } from 'src/dsl/dsl';

// ---------- Straightness ----------

/**
 * True when `x` is a dyadic rational, an integer multiple of 1/2^m for
 * some small m. Used two ways:
 *
 *  - on a note's beat position within the bar: a "standard straight
 *    note" sits on the binary grid (whole/half/quarter/eighth/16th/...);
 *  - on a group's cumulative weight fractions: a straight subdivision
 *    splits at dyadic fractions, a tuplet (triplet/quintuplet/swing)
 *    does not.
 *
 * m runs up to 6 (down to 1/64-of-a-quarter), which comfortably covers
 * everything the renderer needs while still rejecting thirds, fifths,
 * sevenths, etc. The tolerance is on the scaled value so float drift
 * from weight division (e.g. 1/3 = 0.3333…) doesn't read as straight.
 */
export function isDyadic(x: number): boolean {
  if (!Number.isFinite(x)) return false;
  const a = Math.abs(x);
  for (let m = 0; m <= 6; m++) {
    const scaled = a * (1 << m);
    if (Math.abs(scaled - Math.round(scaled)) < 1e-4) return true;
  }
  return false;
}

// ---------- Element weights ----------

/**
 * Effective layout-weight of an element after expansion (one slot per element).
 * Note: expansion unrolls repeats into sibling copies, so by the time the
 * layout reads weights every element's `repeat` is implicitly 1.
 */
export function elementWeight(el: Element): number {
  switch (el.kind) {
    case 'note':
    case 'rest':
    case 'simul':
    case 'group':
    case 'patternRef':
      return el.weight ?? 1;
  }
}

export function sumWeights(els: Element[]): number {
  return els.reduce((a, e) => a + elementWeight(e), 0);
}
