import React from 'react';
import type { StructNote } from 'src/editing/structure/structure_store';
import { SelectionContext } from 'src/editing/selection/selection';
import { SelectionPresenterContext } from 'src/editing/selection/selection_presenter';
import { EditingPresenterContext } from 'src/editing/editing_contexts';
import { StructuralContext } from 'src/editing/jot_editor_contexts';
import { buildLaneMap, laneAtPoint, laneBarsRowTop } from './note_geometry';

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
 * The live preview mirrors what will be committed: it snaps to the grid while
 * snapping is on (discrete jumps to beat lines, not continuous motion) and
 * shifts vertically onto whichever lane row the cursor is over, so a cross-lane
 * drag previews on the destination row before release.
 *
 * Returns `onPointerDown` for the note plus `dragging`, a reactive flag the
 * label reads to stay hidden while a note is being moved. The synthetic click
 * that trails a drag is swallowed at the window (capture phase) so a release
 * over empty bar space doesn't fall through to click-to-seek, and a release
 * over a note doesn't collapse the selection.
 */
export function useNoteDrag(): {
  onPointerDown: (e: React.PointerEvent, note: StructNote) => void;
  dragging: boolean;
} {
  const selection = React.useContext(SelectionContext);
  const selectionPresenter = React.useContext(SelectionPresenterContext);
  const editingPresenter = React.useContext(EditingPresenterContext);
  const structural = React.useContext(StructuralContext);
  const [dragging, setDragging] = React.useState(false);

  const onPointerDown = (e: React.PointerEvent, note: StructNote) => {
    // Left button only; a selection-modifier press is a click (toggle/extend),
    // not a drag.
    if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (!selection || !selectionPresenter || !editingPresenter || !structural) return;

    const startX = e.clientX;
    const startY = e.clientY;
    const pxPerBeat = structural.pxPerBeat;
    const laneOrder = structural.lanes;
    let didDrag = false;
    let els: HTMLElement[] = [];
    // The multi-note selection frame (a ≥2-note bounding box overlay), if shown.
    // It rides along with the glyphs so the dashed box tracks the group mid-drag.
    let frame: HTMLElement | null = null;
    let snapDelta: (rawDeltaBeat: number) => number = (d) => d;
    // Viewport top of the grabbed note's bars row; the cross-lane offset is the
    // delta to the destination row's top (measured, not index-computed).
    let sourceTop: number | null = null;

    const begin = () => {
      didDrag = true;
      setDragging(true);
      // Grab this note if it isn't already part of the selection; an existing
      // multi-selection is dragged as a group.
      if (!selection.isSelected(note)) selectionPresenter.replace(note);
      const ids = selection.effectiveIds;
      els = [...document.querySelectorAll<HTMLElement>('[data-note-id]')].filter((el) =>
        ids.has(el.dataset.noteId ?? '')
      );
      // The dragged glyphs must not intercept `elementFromPoint` mid-drag, or
      // `laneAtPoint` would read their *original* row instead of the one under
      // the cursor; pointer-events are restored on release.
      for (const el of els) el.style.pointerEvents = 'none';
      frame = document.querySelector<HTMLElement>('[data-testid="selection-frame"]');
      snapDelta = editingPresenter.snapDeltaFn(note);
      sourceTop = laneBarsRowTop(note.lane);
    };
    const setTransform = (dx: number, dy: number) => {
      // Compose with the resting `translate(-50%, -50%)` centring so the glyph
      // doesn't drop by half its height the instant a transform is applied.
      const t = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
      for (const el of els) el.style.transform = t;
      // The frame is positioned by `left`/`top` with no resting transform, so it
      // takes the raw offset (no -50% base).
      if (frame) frame.style.transform = `translate(${dx}px, ${dy}px)`;
    };
    const cleanup = () => {
      for (const el of els) {
        el.style.transform = '';
        el.style.pointerEvents = '';
      }
      if (frame) frame.style.transform = '';
    };
    // The vertical pixel offset onto the lane row under the cursor, and the
    // target lane for the eventual commit. The same offset applies to every
    // selected glyph (all anchored on the grabbed note's row), mirroring the
    // uniform row-shift `buildLaneMap` commits.
    const verticalFor = (clientX: number, clientY: number): { dy: number; targetLane: string } => {
      const targetLane = laneAtPoint(clientX, clientY) ?? note.lane;
      if (sourceTop === null || targetLane === note.lane) return { dy: 0, targetLane };
      const targetTop = laneBarsRowTop(targetLane);
      return { dy: targetTop === null ? 0 : targetTop - sourceTop, targetLane };
    };

    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      if (!didDrag && Math.hypot(dx, dy) > DRAG_THRESHOLD) begin();
      if (!didDrag) return;
      const snappedDx = pxPerBeat > 0 ? snapDelta(dx / pxPerBeat) * pxPerBeat : dx;
      setTransform(snappedDx, verticalFor(ev.clientX, ev.clientY).dy);
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      if (!didDrag) return;
      const { targetLane } = verticalFor(ev.clientX, ev.clientY);
      cleanup();
      setDragging(false);
      // Swallow the click synthesised right after this drag so it neither seeks
      // (release over empty bars) nor re-selects (release over a note).
      swallowNextClick();
      const beatDelta = pxPerBeat > 0 ? (ev.clientX - startX) / pxPerBeat : 0;
      const laneMap = buildLaneMap(laneOrder, note.lane, targetLane);
      editingPresenter.moveSelection(note, beatDelta, laneMap);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  };

  return { onPointerDown, dragging };
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
