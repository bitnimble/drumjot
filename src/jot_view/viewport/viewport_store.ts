import { makeAutoObservable } from 'mobx';
import { JotViewStore } from '../jot_view_store';

/**
 * Pixels-per-bar at zoom = 1. Same numeric value as `ViewConfig.barWidth`'s
 * own default so existing layouts are unchanged for users who never touch
 * the slider.
 */
export const BASE_BAR_WIDTH = 448;
export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 4.0;

// Sticky gutter column width (px). Default matches the legacy hardcoded
// 132px so existing layouts are unchanged; the user can drag the gutter's
// right edge to widen it when long track names are clipped.
export const DEFAULT_GUTTER_WIDTH = 132;
// Floor at the width needed to fit the row gutter's minimum content.
export const MIN_GUTTER_WIDTH = 128;
export const MAX_GUTTER_WIDTH = 480;

/**
 * Snap a CSS-pixel value to the nearest 1/dpr boundary. The
 * `.scrollViewport`'s `transform: translate3d(--scroll-x × -1px, ...)`
 * is composited at device-pixel resolution; if scroll values are
 * sub-device-pixel (e.g. 100.3 CSS px on a 2x display = 200.6 device
 * px), the compositor bilinearly interpolates the bitmap each frame
 * and the interpolation distribution shifts as scroll advances,
 * producing a visible ~1px back-and-forth wobble during auto-follow.
 * Snapping keeps every scroll value on the device grid.
 * Also used to lock `--playhead-x` to the same grid as `scrollX` so the
 * centred playhead doesn't drift sub-pixel against the bars below it.
 */
export function snapToDevicePx(x: number): number {
  if (typeof window === 'undefined') return x;
  const dpr = window.devicePixelRatio || 1;
  return Math.round(x * dpr) / dpr;
}

/**
 * Score viewport state: the virtual scroll offsets, cached viewport /
 * content extents, the horizontal zoom multiplier, and the sticky gutter
 * width.
 *
 * Pure data: observables + the `visibleBeatRange` computed (derived from
 * scroll + viewport width + the jot's zoom-driven `pxPerBeat`). All
 * mutation, clamping, device-pixel snapping, zoom application, lives on
 * the presenter; it is the only thing that writes these fields.
 */
export class ViewportStore {
  /** Horizontal zoom multiplier; 1.0 = `BASE_BAR_WIDTH` pixels per bar. */
  zoom: number = 1;
  /**
   * Virtual horizontal scroll offset (px) for the score viewport. The
   * score doesn't use native overflow scrolling: `.jotContainer` is
   * `overflow: hidden` and an inner `.scrollViewport` translates by
   * `(-scrollX, -scrollY)` via CSS `transform`. Driving scroll through an
   * observable gives subpixel precision and makes scroll position
   * reactive. Clamped to `[0, contentWidth - viewportWidth]` by the
   * presenter's scroll actions.
   */
  scrollX: number = 0;
  /** Virtual vertical scroll offset (px). See {@link scrollX}. */
  scrollY: number = 0;
  /**
   * Cached viewport (jotContainer clientWidth/Height) and content
   * (scrollViewport offsetWidth/Height) dimensions, fed by ResizeObservers
   * in JotView. Used to clamp scroll to `[0, content - viewport]`, and
   * read by per-frame observers that derive what to paint from the
   * visible viewport. The underscore prefix is historical and signals
   * "go through the presenter's setters" rather than "non-reactive".
   */
  _viewportWidth: number = 0;
  _viewportHeight: number = 0;
  _contentWidth: number = 0;
  _contentHeight: number = 0;
  /** Width (px) of the sticky mixer/score gutter column; user-resizable
   * by dragging the gutter's right edge. */
  gutterWidth: number = DEFAULT_GUTTER_WIDTH;

  /** The active jot, for the zoom-driven `pxPerBeat` the visible-range
   *  math reads. */
  readonly jotViewStore: JotViewStore;

  constructor(jotViewStore: JotViewStore) {
    this.jotViewStore = jotViewStore;
    makeAutoObservable(this, { jotViewStore: false });
  }

  /**
   * Quarter-note-beat window currently on screen (plus a one-viewport
   * buffer on each side so a fast scroll doesn't outrun the rendered
   * region). Drives horizontal score virtualisation: each row renders
   * only the bars / ticks / words whose beat span intersects this range.
   *
   * Derived purely from observables (`scrollX`, `_viewportWidth`) and the
   * jot's zoom-driven `pxPerBeat`, so it re-derives on scroll / zoom /
   * resize and nothing else, no DOM layout reads (AGENTS.md §5.9).
   *
   * `null` means "windowing disabled, render everything": returned before
   * the first ResizeObserver tick (viewport size unknown) and whenever
   * `pxPerBeat` is degenerate, so initial paint / non-laid-out test
   * environments still render the full score.
   */
  get visibleBeatRange(): { startBeat: number; endBeat: number } | null {
    const ppb = this.jotViewStore.structural?.pxPerBeat ?? 0;
    const vw = this._viewportWidth;
    if (ppb <= 0 || vw <= 0) return null;
    const buffer = vw;
    return {
      startBeat: (this.scrollX - buffer) / ppb,
      endBeat: (this.scrollX + vw + buffer) / ppb,
    };
  }
}
