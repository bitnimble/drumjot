import { makeAutoObservable } from 'mobx';
import { px } from 'src/jot';
import { DocumentStore } from '../document/document_store';
import {
  BASE_BAR_WIDTH,
  MAX_GUTTER_WIDTH,
  MAX_ZOOM,
  MIN_GUTTER_WIDTH,
  MIN_ZOOM,
  snapToDevicePx,
  ViewportStore,
} from './viewport_store';

/**
 * Mutations over {@link ViewportStore}, zoom, the virtual scroll
 * offsets (clamped + device-pixel snapped), the cached viewport/content
 * extents, and the gutter width. Reads {@link DocumentStore} only to
 * write the shared `viewConfig.barWidth` that zoom drives.
 */
export class ViewportPresenter {
  readonly viewport: ViewportStore;
  readonly document: DocumentStore;

  constructor(viewport: ViewportStore, document: DocumentStore) {
    this.viewport = viewport;
    this.document = document;
    makeAutoObservable(this, { viewport: false, document: false });
  }

  setZoom(z: number) {
    const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
    this.viewport.zoom = clamped;
    this.document.viewConfig.barWidth = px(BASE_BAR_WIDTH * clamped);
  }

  /** Cache the score viewport's pixel dimensions. Fed by a ResizeObserver
   * on `.jotContainer`. Re-clamps scroll so a resize that shrinks the
   * viewport (or grows it past the content) doesn't leave scroll parked
   * off the new end. */
  setViewportSize(width: number, height: number): void {
    this.viewport._viewportWidth = width;
    this.viewport._viewportHeight = height;
    this.viewport.scrollX = this.clampScrollX(this.viewport.scrollX);
    this.viewport.scrollY = this.clampScrollY(this.viewport.scrollY);
  }

  /** Cache the scroll-content's pixel dimensions (the inner
   * `.scrollViewport` wrapper's offset size). Re-clamps as above. */
  setContentSize(width: number, height: number): void {
    this.viewport._contentWidth = width;
    this.viewport._contentHeight = height;
    this.viewport.scrollX = this.clampScrollX(this.viewport.scrollX);
    this.viewport.scrollY = this.clampScrollY(this.viewport.scrollY);
  }

  setScrollX(x: number): void {
    this.viewport.scrollX = this.clampScrollX(snapToDevicePx(x));
  }

  setScrollY(y: number): void {
    this.viewport.scrollY = this.clampScrollY(snapToDevicePx(y));
  }

  setScrollBy(dx: number, dy: number): void {
    this.viewport.scrollX = this.clampScrollX(snapToDevicePx(this.viewport.scrollX + dx));
    this.viewport.scrollY = this.clampScrollY(snapToDevicePx(this.viewport.scrollY + dy));
  }

  /** Reset the horizontal scroll to the score's start (Stop transitions).
   * Deliberately does NOT touch scrollY, the user's vertical view
   * shouldn't snap back on Stop, only the playhead-tracking axis. */
  resetScrollX(): void {
    this.viewport.scrollX = 0;
  }

  /** Clamp a tentative target to `[0, contentSize - viewportSize]`. */
  clampScrollX(x: number): number {
    const max = Math.max(0, this.viewport._contentWidth - this.viewport._viewportWidth);
    if (!(x > 0)) return 0;
    if (x > max) return max;
    return x;
  }

  clampScrollY(y: number): number {
    const max = Math.max(0, this.viewport._contentHeight - this.viewport._viewportHeight);
    if (!(y > 0)) return 0;
    if (y > max) return max;
    return y;
  }

  /** Resize the sticky gutter column, clamped to a sensible range so a
   * runaway drag can't collapse the controls or push the bars row off
   * screen. */
  setGutterWidth(width: number): void {
    if (!Number.isFinite(width)) return;
    this.viewport.gutterWidth = Math.min(MAX_GUTTER_WIDTH, Math.max(MIN_GUTTER_WIDTH, width));
  }
}
