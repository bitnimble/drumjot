import React from 'react';
import type { StructNote } from 'src/editing/structure/structure_store';
import { SelectionContext } from 'src/editing/selection/selection';
import { SelectionPresenterContext } from 'src/editing/selection/selection_presenter';
import { EditingPresenterContext } from 'src/editing/editing_contexts';
import { StructuralContext } from 'src/editing/jot_editor_contexts';
import { buildLaneMap, laneAtPoint } from './note_geometry';

/** Movement (px) before a press becomes a drag rather than a click. */
const DRAG_THRESHOLD = 4;

/**
 * Pointer-driven note drag-move. On press (no selection modifier) a small
 * movement promotes to a drag: the selected glyphs translate live via direct
 * DOM writes (off the render path, like the zoom vars), and on release the move
 * is committed to the document in one go through `EditingPresenter.moveSelection`
 *, horizontal delta → beats, vertical drop → target lane. A press that never
 * crosses the threshold stays a click and falls through to selection.
 *
 * Returns `onPointerDown` for the note plus `justDragged`, a ref the click
 * handler checks to suppress the synthetic click that follows a drag.
 */
export function useNoteDrag(): {
  onPointerDown: (e: React.PointerEvent, note: StructNote) => void;
  justDragged: React.MutableRefObject<boolean>;
} {
  const selection = React.useContext(SelectionContext);
  const selectionPresenter = React.useContext(SelectionPresenterContext);
  const editingPresenter = React.useContext(EditingPresenterContext);
  const structural = React.useContext(StructuralContext);
  const justDragged = React.useRef(false);

  const onPointerDown = (e: React.PointerEvent, note: StructNote) => {
    // Left button only; a selection-modifier press is a click (toggle/extend),
    // not a drag.
    if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (!selection || !selectionPresenter || !editingPresenter || !structural) return;

    const startX = e.clientX;
    const startY = e.clientY;
    const pxPerBeat = structural.pxPerBeat;
    let dragging = false;
    let els: HTMLElement[] = [];

    const begin = () => {
      dragging = true;
      // Grab this note if it isn't already part of the selection; an existing
      // multi-selection is dragged as a group.
      if (!selection.isSelected(note)) selectionPresenter.replace(note);
      const ids = selection.effectiveIds;
      els = [...document.querySelectorAll<HTMLElement>('[data-note-id]')].filter((el) =>
        ids.has(el.dataset.noteId ?? '')
      );
    };
    const setTransform = (dx: number) => {
      for (const el of els) el.style.transform = `translateX(${dx}px)`;
    };
    const clearTransform = () => {
      for (const el of els) el.style.transform = '';
    };

    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      if (!dragging && Math.hypot(dx, dy) > DRAG_THRESHOLD) begin();
      if (dragging) setTransform(dx);
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      if (!dragging) return;
      clearTransform();
      justDragged.current = true;
      const beatDelta = pxPerBeat > 0 ? (ev.clientX - startX) / pxPerBeat : 0;
      const targetLane = laneAtPoint(ev.clientX, ev.clientY) ?? note.lane;
      const laneMap = buildLaneMap(structural.lanes, note.lane, targetLane);
      editingPresenter.moveSelection(note, beatDelta, laneMap);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  };

  return { onPointerDown, justDragged };
}
