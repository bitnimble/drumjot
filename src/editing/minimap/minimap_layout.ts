/**
 * Pure layout math for the minimap, split out of `minimap.tsx` so it can
 * be unit-tested without a canvas or a MobX store. These run inside the
 * component's `useMemo` / `reaction` (NOT the per-frame canvas paint), so
 * keeping them here is perf-neutral; the paint, pointer, and per-frame
 * observer code stays in `minimap.tsx`.
 */

export type BarLayout = {
  /** Minimap-px x of the bar's left edge. */
  x: number;
  /** Minimap-px width of the bar. */
  width: number;
  /** Bar length in DSL beats (== `StructBar.beats`); used to map
   *  per-note `beat` positions onto the bar's minimap pixel range. */
  beats: number;
};

export type NoteMark = {
  x: number;
  color: string;
};

export const EMPTY_NOTE_MARKS: readonly NoteMark[] = Object.freeze([]);

/** Content-equality for two note-mark lists, used as the `reaction`
 *  `equals` so a recompute that yields identical marks collapses to the
 *  same reference and the canvas paint effect doesn't refire. */
export function noteMarksEqual(a: readonly NoteMark[], b: readonly NoteMark[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].x !== b[i].x || a[i].color !== b[i].color) return false;
  }
  return true;
}

/** Just the timing fields the minimap reads off each `BarTiming`. */
export type MinimapBarTiming = {
  startSec: number;
  durationSec: number;
};

export type BarLayoutResult = {
  bars: BarLayout[];
  totalDuration: number;
  firstStartSec: number;
  hasContent: boolean;
};

const EMPTY_LAYOUT: BarLayoutResult = {
  bars: [],
  totalDuration: 0,
  firstStartSec: 0,
  hasContent: false,
};

/**
 * Map each bar's jot-time onto a minimap pixel range so the strip is
 * fully zoom-invariant: a bar's `x` / `width` derive from its
 * `startSec` / `durationSec` against the whole-song duration, scaled to
 * the available pixel `width`. `beats` is carried through from the
 * structural bars (by index) so per-note positions can be plotted later.
 *
 * Returns an empty (`hasContent: false`) result when there's nothing to
 * lay out: no pixels yet, no bars on either side, or a non-positive
 * total duration.
 */
export function computeBarLayouts(
  timelineBars: readonly MinimapBarTiming[],
  structBars: readonly { beats: number }[],
  width: number
): BarLayoutResult {
  if (width <= 0) return EMPTY_LAYOUT;
  if (timelineBars.length === 0 || structBars.length === 0) return EMPTY_LAYOUT;
  const first = timelineBars[0].startSec;
  const last = timelineBars[timelineBars.length - 1];
  const total = last.startSec + last.durationSec - first;
  if (total <= 0) return EMPTY_LAYOUT;
  const layouts: BarLayout[] = new Array(timelineBars.length);
  for (let i = 0; i < timelineBars.length; i++) {
    const t = timelineBars[i];
    layouts[i] = {
      x: ((t.startSec - first) / total) * width,
      width: (t.durationSec / total) * width,
      beats: structBars[i]?.beats ?? 0,
    };
  }
  return { bars: layouts, totalDuration: total, firstStartSec: first, hasContent: true };
}
