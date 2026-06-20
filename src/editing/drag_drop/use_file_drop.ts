/**
 * Window-level file drop wiring for the editor. Returns the drag handlers
 * to spread onto the app-shell container, a `dragActive` flag for the
 * "Drop to load" overlay, and the pending-confirm state for the
 * replace-document dialog.
 *
 * Transient interaction state only (drag-over flag + the depth counter that
 * de-bounces child enter/leave, plus the pending plan awaiting confirm), so
 * it lives in React, not a store. The actual loads run on
 * {@link JotEditorPresenter.executeDropPlan}.
 */
import React from 'react';
import { JotEditorPresenter } from 'src/editing/jot_editor_presenter';
import { JotEditorStore } from 'src/editing/jot_editor_store';
import { DropPlan, planDrop } from 'src/editing/drag_drop/file_routing';

/** Only react to OS file drags. Internal mixer / layers drag-and-drop use
 *  custom `application/x-drumjot-*` MIME types (never `'Files'`), so this
 *  cleanly ignores them and lets their own handlers run. */
function isFileDrag(e: React.DragEvent): boolean {
  return Array.from(e.dataTransfer.types).includes('Files');
}

export type FileDrop = {
  /** True while an OS file drag is hovering the editor. Drives the overlay. */
  dragActive: boolean;
  /** A plan whose document load would replace the open score, awaiting the
   *  user's confirmation. `null` when nothing is pending. */
  pendingPlan: DropPlan | null;
  /** Spread onto the app-shell container. */
  dropHandlers: {
    onDragEnter: (e: React.DragEvent) => void;
    onDragOver: (e: React.DragEvent) => void;
    onDragLeave: (e: React.DragEvent) => void;
    onDrop: (e: React.DragEvent) => void;
  };
  /** Run a set of files through the same auto-detect flow as a drop
   *  (classify → confirm-if-replacing → load). Lets non-drop entry points
   *  (e.g. the toolbar's "Load zip" picker) reuse the routing + confirm. */
  openFiles: (files: File[]) => void;
  /** Run the pending plan (the user accepted the replacement). */
  confirmPending: () => void;
  /** Discard the pending plan (the user cancelled). */
  cancelPending: () => void;
};

export function useFileDrop(
  jotEditorStore: JotEditorStore,
  jotEditorPresenter: JotEditorPresenter
): FileDrop {
  const [dragActive, setDragActive] = React.useState(false);
  const [pendingPlan, setPendingPlan] = React.useState<DropPlan | null>(null);
  // Drag enter/leave fire for every descendant the pointer crosses; count
  // them so the overlay only clears when the drag truly leaves the window.
  const dragDepth = React.useRef(0);

  const onDragEnter = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepth.current += 1;
    setDragActive(true);
  };

  const onDragOver = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    // Required for the drop to fire at all; also sets the copy cursor.
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const onDragLeave = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) {
      dragDepth.current = 0;
      setDragActive(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepth.current = 0;
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) void handleFiles(files);
  };

  const handleFiles = async (files: File[]) => {
    const plan = await planDrop(files);
    // A document-replacing load is only destructive when a score is already
    // open; otherwise (empty state) just run it.
    const replacesOpenScore =
      plan.documentLoad != null && jotEditorStore.structural != null;
    if (replacesOpenScore) {
      setPendingPlan(plan);
    } else {
      await jotEditorPresenter.executeDropPlan(plan);
    }
  };

  const confirmPending = () => {
    const plan = pendingPlan;
    setPendingPlan(null);
    if (plan) void jotEditorPresenter.executeDropPlan(plan);
  };

  const cancelPending = () => setPendingPlan(null);

  const openFiles = (files: File[]) => {
    if (files.length > 0) void handleFiles(files);
  };

  return {
    dragActive,
    pendingPlan,
    dropHandlers: { onDragEnter, onDragOver, onDragLeave, onDrop },
    openFiles,
    confirmPending,
    cancelPending,
  };
}
