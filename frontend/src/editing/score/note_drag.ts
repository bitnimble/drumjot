import React from 'react';
import type { StructNote } from 'src/editing/structure/structure_store';
import type { EditingPresenter } from 'src/editing/editing_presenter';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { EditingPresenterContext } from 'src/editing/editing_contexts';
import { SelectionContext, type SelectionStore } from 'src/editing/selection/selection';
import { SelectionPresenterContext } from 'src/editing/selection/selection_presenter';
import { StructuralContext } from 'src/editing/jot_editor_contexts';
import { notesById } from 'src/editing/score/note_geometry';

/** Movement (px) before a press becomes a drag rather than a click. */
const DRAG_THRESHOLD = 4;

/**
 * Shared pointer-driven drag-move machinery. A press (no selection modifier)
 * that crosses the threshold promotes to a drag handed to {@link
 * EditingPresenter}: `beginDragMove` snapshots the selection's positions, the
 * per-lane bars-row pointer-move handlers report the target lane + cursor x
 * (see `InstrumentTrackView`) so the presenter recomputes the preview top-down
 * with no DOM reads, and `commitDragMove` writes the result on release. A press
 * that never crosses the threshold stays a click and falls through.
 *
 * `startLane` is the lane the gesture's vertical shift is measured FROM. A
 * notehead drag passes the grabbed note's lane (the cursor is on it, so the
 * group never jumps at drag start); a frame drag passes `undefined` so the
 * presenter lazily adopts the first lane the cursor reports (the row under the
 * press), again keeping the start delta at zero wherever inside the frame the
 * user grabbed.
 *
 * The dragged glyphs hide and the preview renders off `EditingStore.dragActive`
 * / `dragPreview`, so no per-note drag flag is threaded through. The synthetic
 * click trailing a drag is swallowed at the window (capture phase) so a release
 * over empty bars doesn't seek and a release over a note doesn't reselect.
 */
function startDrag(
  e: React.PointerEvent,
  editingPresenter: EditingPresenter | null,
  anchor: StructNote,
  startLane: string | undefined
): void {
  // Left button only; a selection-modifier press is a click (toggle/extend),
  // not a drag.
  if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) return;
  if (!editingPresenter) return;

  const startX = e.clientX;
  const startY = e.clientY;
  // Don't hold pointer capture on the target: the per-lane bars-row pointer-move
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
    editingPresenter.beginDragMove(anchor, startX, startLane);
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
}

/**
 * Notehead drag-move: grab a specific note (which becomes the snap anchor) and
 * move the whole selection. The grabbed note's lane is the vertical-shift
 * origin, so the group never jumps when the drag begins.
 */
export function useNoteDrag(): {
  onPointerDown: (e: React.PointerEvent, note: StructNote) => void;
} {
  const editingPresenter = React.useContext(EditingPresenterContext);
  const onPointerDown = (e: React.PointerEvent, note: StructNote) =>
    startDrag(e, editingPresenter, note, note.lane);
  return { onPointerDown };
}

/**
 * Selection-frame drag-move: press anywhere inside the multi-note selection
 * frame (not just on a notehead) to move the whole group. The snap anchor is
 * the selection's pivot (falling back to any member), and the vertical-shift
 * origin is hit-tested from the row under the press, so the selection doesn't
 * jump wherever inside the frame the user grabs. The returned `onClick`
 * forwards ctrl/shift-clicks through the frame to the note beneath so selection
 * editing still works over the overlay.
 */
export function useFrameDrag(): {
  onPointerDown: (e: React.PointerEvent) => void;
  onClick: (e: React.MouseEvent) => void;
} {
  const editingPresenter = React.useContext(EditingPresenterContext);
  const selection = React.useContext(SelectionContext);
  const selectionPresenter = React.useContext(SelectionPresenterContext);
  const structural = React.useContext(StructuralContext);

  const onPointerDown = (e: React.PointerEvent) => {
    // A modifier press is a selection edit, handled on click (below): toggling
    // here would mutate the selection mid-gesture, unmount the frame, and let
    // the trailing click re-hit the exposed note (a double-toggle). Only a plain
    // press becomes the group drag.
    if (e.ctrlKey || e.metaKey || e.shiftKey) return;
    const anchor = frameDragAnchor(selection);
    if (!anchor) return;
    // The vertical-shift origin must be hit-tested through the frame rather than
    // read off the grabbed glyph; a lazy first-move capture misses the start row
    // on a fast cross-lane drag (the frame eats the initial moves before it
    // unmounts).
    const startLane = laneAtPoint(e.currentTarget as HTMLElement, e.clientX, e.clientY);
    startDrag(e, editingPresenter, anchor, startLane);
  };

  // The frame paints above the notes so it can be grabbed anywhere, but that
  // would also swallow ctrl/shift-clicks that edit the selection. Forward those
  // to the note beneath so toggle / extend / shrink keep working. A plain click
  // does nothing (seeks/single-selects inside the frame stay blocked, as
  // intended); a drag's trailing click is already swallowed at the window.
  const onClick = (e: React.MouseEvent) => {
    if (!(e.ctrlKey || e.metaKey || e.shiftKey)) return;
    const note = noteAtPoint(e.currentTarget as HTMLElement, e.clientX, e.clientY, structural);
    if (!note) return;
    if (e.shiftKey) selectionPresenter?.extendTo(note);
    else selectionPresenter?.toggle(note);
  };

  return { onPointerDown, onClick };
}

/** The element directly under a point, hit-tested beneath the (momentarily
 *  transparent) frame. A single DOM read per gesture, never per-move. */
function elementUnderFrame(frame: HTMLElement, x: number, y: number): Element | null {
  const prev = frame.style.pointerEvents;
  frame.style.pointerEvents = 'none';
  const el = document.elementFromPoint(x, y);
  frame.style.pointerEvents = prev;
  return el;
}

/** Lane row under a point (for seeding a frame drag's vertical-shift origin). */
function laneAtPoint(frame: HTMLElement, x: number, y: number): string | undefined {
  return elementUnderFrame(frame, x, y)?.closest<HTMLElement>('[data-lane]')?.dataset.lane;
}

/** The note glyph under a point, resolved to its live {@link StructNote} (for
 *  forwarding a modifier-click through the frame to the note it overlays). */
function noteAtPoint(
  frame: HTMLElement,
  x: number,
  y: number,
  structural: StructuralPresenter | null
): StructNote | undefined {
  const id = elementUnderFrame(frame, x, y)?.closest<HTMLElement>('[data-note-id]')?.dataset.noteId;
  if (!id || !structural) return undefined;
  return notesById(structural.musicalLayers).get(id);
}

/** The note a frame drag snaps to grid: the selection's pivot if it's still a
 *  member, else any member. `undefined` for an empty selection. */
function frameDragAnchor(selection: SelectionStore | null): StructNote | undefined {
  if (!selection) return undefined;
  const notes = selection.effectiveNotes;
  if (notes.size === 0) return undefined;
  const pivotId = selection.anchor?.id;
  for (const n of notes) if (n.id === pivotId) return n;
  return notes.values().next().value;
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
