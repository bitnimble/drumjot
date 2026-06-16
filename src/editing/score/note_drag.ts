import React from 'react';
import type { StructNote } from 'src/editing/structure/structure_store';
import { EditingPresenterContext } from 'src/editing/editing_contexts';

/** Movement (px) before a press becomes a drag rather than a click. */
const DRAG_THRESHOLD = 4;

/**
 * Pointer-driven note drag-move. A press (no selection modifier) that crosses
 * the threshold promotes to a drag, handed entirely to {@link
 * EditingPresenter}: `beginDragMove` snapshots the selection's positions, the
 * per-lane bars-row pointer-move handlers report the target lane + cursor x
 * (see `InstrumentTrackView`) so the presenter recomputes the preview top-down
 * with no DOM reads, and `commitDragMove` writes the result on release. A press
 * that never crosses the threshold stays a click and falls through to
 * selection.
 *
 * Returns just `onPointerDown`; the dragged glyphs hide and the preview renders
 * off `EditingStore.dragActive` / `dragPreview`, so no per-note drag flag is
 * threaded through. The synthetic click trailing a drag is swallowed at the
 * window (capture phase) so a release over empty bars doesn't seek and a
 * release over a note doesn't reselect.
 */
export function useNoteDrag(): {
  onPointerDown: (e: React.PointerEvent, note: StructNote) => void;
} {
  const editingPresenter = React.useContext(EditingPresenterContext);

  const onPointerDown = (e: React.PointerEvent, note: StructNote) => {
    // Left button only; a selection-modifier press is a click (toggle/extend),
    // not a drag.
    if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (!editingPresenter) return;

    const startX = e.clientX;
    const startY = e.clientY;
    // Don't hold pointer capture on the note: the per-lane bars-row pointer-move
    // handlers must keep firing as the cursor crosses tracks so they can report
    // the lane the cursor is over (no elementFromPoint needed). Touch sets an
    // implicit capture on the target; release it (guarded, releasing a
    // capture the element doesn't hold throws).
    const target = e.currentTarget as Element;
    if (target.hasPointerCapture?.(e.pointerId)) target.releasePointerCapture(e.pointerId);

    let didDrag = false;
    const teardown = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onCancel);
    };
    const onMove = (ev: PointerEvent) => {
      if (didDrag) return;
      if (Math.hypot(ev.clientX - startX, ev.clientY - startY) <= DRAG_THRESHOLD) return;
      didDrag = true;
      // Position updates come from the bars-row handlers; this only kicks it off.
      editingPresenter.beginDragMove(note, startX);
    };
    const onUp = () => {
      teardown();
      if (!didDrag) return;
      swallowNextClick();
      editingPresenter.commitDragMove();
    };
    const onCancel = () => {
      teardown();
      if (didDrag) editingPresenter.cancelDragMove();
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onCancel);
  };

  return { onPointerDown };
}

/**
 * Consume exactly one upcoming `click` at the window's capture phase, so the
 * synthetic click that trails a pointer drag never reaches the bars-row seek
 * handler or a note's select handler. Self-removing on that click, with a
 * macrotask fallback in case no click is synthesised (release outside any
 * clickable target) so a stale listener can't swallow a later real click.
 */
function swallowNextClick(): void {
  const swallow = (e: MouseEvent) => {
    e.stopPropagation();
    teardown();
  };
  const teardown = () => {
    window.removeEventListener('click', swallow, true);
    clearTimeout(fallback);
  };
  window.addEventListener('click', swallow, true);
  const fallback = window.setTimeout(teardown, 0);
}
