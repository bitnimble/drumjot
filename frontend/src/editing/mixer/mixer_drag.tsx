import classNames from 'classnames';
import React from 'react';
import styles from './mixer.module.css';

/**
 * Drag-source identifier carried on the DataTransfer of a mixer-row
 * drag. A custom MIME type lets us reject foreign drops (files,
 * external pages) so the gutter never tries to swallow them.
 */
const MIXER_DRAG_MIME = 'application/x-drumjot-mixer-row';

/** Common drag/drop props passed to every mixer row. */
export type MixerRowDragProps = {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDragStartIdx: (i: number) => void;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
  /** Length of the mixer list (used by the end-of-list drop zone). */
  mixerLength: number;
  /**
   * True when this row starts a new group (its `groupId` differs from
   * the previous row's, or it's not in a group at all and follows a
   * row that was). The row renders a small top margin so adjacent
   * groups read as distinct clusters; same-group rows render flush.
   * The first row in the mixer never receives this — nothing above it
   * to gap against.
   */
  groupStart: boolean;
  /**
   * True when this row ends a group — it has a `groupId` AND the next
   * row has a different (or no) `groupId`. Together with `groupStart`
   * it lets the row know it's on the outer edge of a real group (vs a
   * solo row that just happens to follow a different cluster), so the
   * outer border can render thicker than a regular inter-row separator.
   */
  groupEnd: boolean;
  /** True iff this row is part of a group (`key.groupId !== undefined`). */
  inGroup: boolean;
  /** Pointer-down handler for the gutter-edge resize affordance. */
  onResizeGutterStart: (e: React.PointerEvent<HTMLDivElement>) => void;
};

/**
 * Shared drag-target behaviour for the row gutter: a drag-over either
 * marks "drop above this row" (top half) or "drop below this row"
 * (bottom half), `onDrop` commits the move. Returns the props/style
 * fragments the row should spread onto its wrapper + a boolean for
 * whether the drop indicator should render above this row.
 */
export function useMixerRowDropTarget({
  idx,
  dragFromIdx,
  dropTargetIdx,
  onDropTargetIdx,
  onMoveTrack,
  onResetDrag,
}: {
  idx: number;
  dragFromIdx: number | undefined;
  dropTargetIdx: number | undefined;
  onDropTargetIdx: (i: number | undefined) => void;
  onMoveTrack: (from: number, to: number) => void;
  onResetDrag: () => void;
}) {
  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    if (!e.dataTransfer.types.includes(MIXER_DRAG_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const rect = e.currentTarget.getBoundingClientRect();
    const isTopHalf = e.clientY < rect.top + rect.height / 2;
    const target = isTopHalf ? idx : idx + 1;
    if (target !== dropTargetIdx) onDropTargetIdx(target);
  };
  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    // Don't clear when the pointer just crossed into a child element;
    // only when it actually leaves the row bounds (relatedTarget
    // outside the gutter element).
    const next = e.relatedTarget as Node | null;
    if (next && e.currentTarget.contains(next)) return;
    if (dropTargetIdx === idx || dropTargetIdx === idx + 1) onDropTargetIdx(undefined);
  };
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (dragFromIdx === undefined) return;
    const data = e.dataTransfer.getData(MIXER_DRAG_MIME);
    if (!data) return;
    e.preventDefault();
    const from = parseInt(data, 10);
    if (Number.isFinite(from) && dropTargetIdx !== undefined) {
      onMoveTrack(from, dropTargetIdx);
    }
    onResetDrag();
  };
  const isDropIndicatorAbove = dropTargetIdx === idx && dragFromIdx !== undefined;
  const isDropIndicatorBelow = dropTargetIdx === idx + 1 && dragFromIdx !== undefined;
  return { onDragOver, onDragLeave, onDrop, isDropIndicatorAbove, isDropIndicatorBelow };
}

/**
 * Drag handle (≡) parked on the leftmost edge of every mixer row's
 * gutter. Only this element is `draggable`, so the user can still click
 * mute/solo, drag the volume slider, etc. without accidentally lifting
 * the whole row.
 */
export const MixerDragHandle = ({
  idx,
  onDragStartIdx,
  onResetDrag,
  ariaLabel,
}: {
  idx: number;
  onDragStartIdx: (i: number) => void;
  onResetDrag: () => void;
  ariaLabel: string;
}) => {
  return (
    <div
      className={styles.mixerDragHandle}
      draggable={true}
      // The page-level mousedown listener (createJotEditor's marquee
      // selection) calls `preventDefault()`, which also cancels the
      // subsequent native dragstart — so without this stop the row
      // never lifts and the user just gets a marquee instead. The
      // handle's own mousedown still fires; only the bubbled handler
      // is suppressed.
      onMouseDown={(e) => e.stopPropagation()}
      onDragStart={(e) => {
        e.dataTransfer.setData(MIXER_DRAG_MIME, String(idx));
        // Some browsers refuse the drag with no plain-text payload.
        e.dataTransfer.setData('text/plain', String(idx));
        e.dataTransfer.effectAllowed = 'move';
        onDragStartIdx(idx);
      }}
      onDragEnd={() => {
        // dragend fires whether or not the drop took — clear the
        // ephemeral state either way so a cancelled drag (Escape, drop
        // outside) doesn't leave the indicator stuck.
        onResetDrag();
      }}
      title={`${ariaLabel} (drag to reorder)`}
      aria-label={`Reorder ${ariaLabel}`}
      role="button"
    >
      ⋮⋮
    </div>
  );
};
