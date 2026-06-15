import React from 'react';
import styles from './gutter_resize_handle.module.css';

/**
 * A 6px-wide vertical strip parked on the right edge of each sticky
 * gutter element. Pointer-down on the strip starts the gutter resize
 * (the actual pointer-move/up logic lives in JotEditor — captured at drag
 * start so deltas stay anchored to the grab point); this component just
 * renders the affordance.
 *
 * One handle per gutter row (rather than a single spanning overlay) keeps
 * the implementation flat: each gutter is already sticky-left-aligned to
 * `--gutter-width`, so the handles visually line up to form a continuous
 * resize edge without any extra positioning machinery.
 */
export const GutterResizeHandle = ({
  onResizeStart,
}: {
  onResizeStart: (e: React.PointerEvent<HTMLDivElement>) => void;
}) => (
  <div
    className={styles.gutterResizeHandle}
    onPointerDown={onResizeStart}
    title="Drag to resize the track gutter."
    aria-label="Resize track gutter"
    role="separator"
  />
);
