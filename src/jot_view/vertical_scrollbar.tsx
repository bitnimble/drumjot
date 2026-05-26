/**
 * Synthetic vertical scrollbar for the score viewport. The score uses a
 * transform-based virtual scroll (see `.jotContainer` in jot_view.module.css),
 * so a native scrollbar is unavailable. Wheel zooms instead of scrolling
 * vertically, leaving track lists that overflow the viewport unreachable
 * without middle-click pan; this strip drags `store.scrollY` directly.
 *
 * Only renders when there is actual vertical overflow. Same pattern as
 * `Minimap`: ResizeObservers on both the container and the scroll-content
 * keep the thumb in sync as zoom / mixer changes resize either side.
 */
import { observer } from 'mobx-react-lite';
import React from 'react';
import styles from './vertical_scrollbar.module.css';
import type { JotViewStore } from './store';

const MIN_THUMB_HEIGHT = 24;

export const VerticalScrollbar = observer(
  ({
    store,
    containerRef,
    viewportRef,
  }: {
    store: JotViewStore;
    containerRef: React.RefObject<HTMLDivElement>;
    viewportRef: React.RefObject<HTMLDivElement>;
  }) => {
    const [containerHeight, setContainerHeight] = React.useState(0);
    const [contentHeight, setContentHeight] = React.useState(0);

    React.useLayoutEffect(() => {
      const container = containerRef.current;
      const viewport = viewportRef.current;
      if (!container || !viewport) return;
      const measure = () => {
        setContainerHeight(container.clientHeight);
        setContentHeight(viewport.offsetHeight);
      };
      measure();
      const containerRo = new ResizeObserver(measure);
      const viewportRo = new ResizeObserver(measure);
      containerRo.observe(container);
      viewportRo.observe(viewport);
      return () => {
        containerRo.disconnect();
        viewportRo.disconnect();
      };
    }, [containerRef, viewportRef]);

    const overflow = contentHeight - containerHeight;
    const scrollY = store.scrollY;

    if (overflow <= 0 || containerHeight <= 0) return null;

    const rawThumbHeight = (containerHeight / contentHeight) * containerHeight;
    const thumbHeight = Math.max(MIN_THUMB_HEIGHT, rawThumbHeight);
    const travel = Math.max(1, containerHeight - thumbHeight);
    const thumbTop = (scrollY / overflow) * travel;

    const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const trackRect = e.currentTarget.getBoundingClientRect();
      const localY = e.clientY - trackRect.top;
      const onThumb = localY >= thumbTop && localY <= thumbTop + thumbHeight;
      let grabOffset: number;
      if (onThumb) {
        grabOffset = localY - thumbTop;
      } else {
        // Click off the thumb: centre the thumb on the click, then
        // continue as a drag from there (mirrors Minimap behavior).
        grabOffset = thumbHeight / 2;
        const newTop = Math.max(0, Math.min(travel, localY - grabOffset));
        store.setScrollY((newTop / travel) * overflow);
      }

      let pendingClientY = e.clientY;
      let rafId = 0;
      const flush = () => {
        rafId = 0;
        const y = pendingClientY - trackRect.top - grabOffset;
        const clamped = Math.max(0, Math.min(travel, y));
        store.setScrollY((clamped / travel) * overflow);
      };
      const onMove = (ev: PointerEvent) => {
        pendingClientY = ev.clientY;
        if (rafId === 0) rafId = requestAnimationFrame(flush);
      };
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        window.removeEventListener('pointercancel', onUp);
        if (rafId !== 0) {
          cancelAnimationFrame(rafId);
          flush();
        }
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
      window.addEventListener('pointercancel', onUp);
    };

    return (
      <div
        className={styles.track}
        onPointerDown={onPointerDown}
        aria-label="Vertical scroll"
      >
        <div
          className={styles.thumb}
          style={{ top: `${thumbTop}px`, height: `${thumbHeight}px` }}
          aria-hidden="true"
        />
      </div>
    );
  },
);
