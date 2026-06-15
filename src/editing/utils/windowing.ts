/**
 * Horizontal score virtualisation helpers. Pure beat-space geometry so
 * every row (instrument bars, timeline ticks, lyric words) decides what
 * to render with the same rule and it can be unit-tested without a DOM.
 *
 * The visible window itself is `JotEditorStore.visibleBeatRange`, derived
 * from scroll / viewport / zoom observables; these functions answer
 * "does this element fall inside it?".
 */

import { untracked } from 'mobx';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';

/** A half-open-ish quarter-note-beat window. Both ends inclusive (see
 *  {@link intersectsBeatRange}). */
export type BeatRange = { startBeat: number; endBeat: number };

/**
 * Inline seed for a bars-row's `--bars-row-width` (`layerBeats ×
 * pxPerBeat`, in px). Every beat-anchored child sizes/positions itself
 * as a percentage of this width, so a row with the var unset collapses to
 * 0.
 *
 * `pxPerBeat` is read through `untracked` so setting this in an
 * `observer` row's inline style does NOT subscribe the row to zoom (the
 * row must stay off the zoom re-render path; `ScoreZoomVar` updates the
 * var imperatively on each zoom tick instead). The seed covers the cases
 * `ScoreZoomVar` can't: the row's initial mount, a row mounted *after*
 * load (a freshly loaded audio / lyrics track, which doesn't change
 * pxPerBeat so `ScoreZoomVar` wouldn't re-fire), and any non-zoom
 * re-render of the row (React would otherwise reset the imperatively-set
 * var to a stale inline value - reading the live pxPerBeat here keeps it
 * correct). Mirrors how `--gutter-width` is seeded inline + updated by
 * `GutterWidthVar`.
 */
export function barsRowWidthSeed(structural: StructuralPresenter, layerBeats: number): string {
  return `${untracked(() => structural.pxPerBeat) * layerBeats}px`;
}

/**
 * Whether an element spanning `[startBeat, startBeat + beats]` (in
 * quarter-note beats, bars-row-local) intersects the visible window.
 *
 * A `null` range means windowing is disabled, render everything. This
 * is the pre-layout / no-viewport fallback so initial paint and
 * non-laid-out test environments show the full score.
 *
 * Endpoints are inclusive so an element flush with the buffer edge still
 * renders rather than popping in a frame late.
 */
export function intersectsBeatRange(
  range: BeatRange | null,
  startBeat: number,
  beats: number,
): boolean {
  if (!range) return true;
  return startBeat + beats >= range.startBeat && startBeat <= range.endBeat;
}
