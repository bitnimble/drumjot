/**
 * Pure DSL element algorithms: pattern/repeat expansion (the tree
 * rewriting that turns `patternRef` + `*N` into a flat element list),
 * element beat weights, straightness (dyadic-grid) test, and the
 * element kind type-guards. No layout / pixel / MobX concerns; consumed
 * by the layout engine in `resolved_jot.ts` and re-exported from `jot.ts`.
 */
import {
  Element,
  Group,
  Note,
  Pattern,
  PatternRef,
  PatternSubstitution,
  Rest,
  Simultaneity,
} from 'src/dsl/dsl';

// ---------- Straightness ----------

/**
 * True when `x` is a dyadic rational — an integer multiple of 1/2^m for
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

// ---------- Pattern expansion ----------

/**
 * Expand pattern references and unroll `*N` repeats at every nesting level.
 * After expansion the element tree contains only notes, rests, simultaneities
 * and groups; pattern refs are inlined and no element has `repeat > 1`.
 */
export function expandElements(
  els: Element[],
  patterns: Record<string, Pattern>
): Element[] {
  const out: Element[] = [];
  for (const el of els) {
    for (const e of expandElement(el, patterns)) out.push(e);
  }
  return out;
}

function expandElement(el: Element, patterns: Record<string, Pattern>): Element[] {
  switch (el.kind) {
    case 'note':
    case 'rest':
      return unroll(el);
    case 'simul':
      return [{ ...el, elements: expandElements(el.elements, patterns) }];
    case 'group':
      return unroll({ ...el, elements: expandElements(el.elements, patterns) });
    case 'patternRef': {
      const pattern = patterns[el.name];
      // Tag the expanded group with `patternSource` so the renderer can draw
      // an outline + label for every usage (including unrolled `*N` copies).
      if (!pattern) {
        return unroll({
          kind: 'group',
          elements: [],
          weight: el.weight,
          repeat: el.repeat,
          patternSource: { name: el.name },
        });
      }
      let elements = expandElements(pattern.elements, patterns);
      if (el.substitutions) {
        elements = applySubstitutions(elements, el.substitutions);
      }
      return unroll({
        kind: 'group',
        elements,
        weight: el.weight,
        repeat: el.repeat,
        patternSource: { name: el.name },
      });
    }
  }
}

function unroll<T extends Element & { repeat?: number }>(el: T): Element[] {
  const repeat = el.repeat ?? 1;
  if (repeat <= 1) {
    const { repeat: _r, ...rest } = el;
    return [rest as Element];
  }
  const copies: Element[] = [];
  for (let i = 0; i < repeat; i++) {
    const { repeat: _r, ...rest } = el;
    copies.push(rest as Element);
  }
  return copies;
}

function applySubstitutions(
  elements: Element[],
  subs: PatternSubstitution[]
): Element[] {
  let result = elements;
  for (const sub of subs) {
    result = applySubstitution(result, sub.path, sub.replacement);
  }
  return result;
}

function applySubstitution(
  elements: Element[],
  path: Array<number | [number, number]>,
  replacement: Element
): Element[] {
  if (path.length === 0) return elements;
  const [head, ...rest] = path;
  const copy = elements.slice();

  if (rest.length === 0) {
    if (typeof head === 'number') {
      const idx = head - 1;
      if (idx >= 0 && idx < copy.length) copy[idx] = replacement;
    } else {
      const [start, end] = head;
      const s = Math.max(0, start - 1);
      const e = Math.min(copy.length, end);
      copy.splice(s, e - s, replacement);
    }
    return copy;
  }

  // Descend into a group at position `head`.
  const idx = (typeof head === 'number' ? head : head[0]) - 1;
  const target = copy[idx];
  if (target && target.kind === 'group') {
    copy[idx] = { ...target, elements: applySubstitution(target.elements, rest, replacement) };
  }
  return copy;
}

// ---------- Misc helpers ----------

/** Convenience type guards. */
export const isNote = (el: Element): el is Note => el.kind === 'note';
export const isRest = (el: Element): el is Rest => el.kind === 'rest';
export const isGroup = (el: Element): el is Group => el.kind === 'group';
export const isSimul = (el: Element): el is Simultaneity => el.kind === 'simul';
export const isPatternRef = (el: Element): el is PatternRef => el.kind === 'patternRef';
