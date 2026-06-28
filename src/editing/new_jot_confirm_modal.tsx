import React from 'react';
import { ConfirmModal, modalStyles } from 'src/ui/modal/modal';

/**
 * Confirmation prompt shown before File → New jot replaces a session that has
 * unsaved edits. Starting a new jot is a wholesale replace with no undo across
 * the load boundary, so the user gets a chance to back out (and save first).
 * Only shown when the loaded song is dirty; a clean session creates the new jot
 * without asking. Open/close state is transient React state owned by the host.
 */
export const NewJotConfirmModal: React.FC<{
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}> = ({ open, onConfirm, onCancel }) => (
  <ConfirmModal
    open={open}
    onConfirm={onConfirm}
    onCancel={onCancel}
    title="Discard unsaved changes?"
    ariaLabel="Discard unsaved changes?"
    confirmLabel="Discard & start new"
    confirmVariant="danger"
    autoFocus="cancel"
    testId="new-jot-confirm-modal"
    closeTestId="new-jot-confirm-cancel"
    confirmTestId="new-jot-confirm-discard"
  >
    <p>
      This jot has changes you haven't saved. Starting a new jot will replace
      it, and this can't be undone.
    </p>
    <p className={modalStyles.note}>Save first if you want to keep your work.</p>
  </ConfirmModal>
);
