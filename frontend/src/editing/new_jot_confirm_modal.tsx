import React from 'react';
import { X } from 'lucide-react';
import styles from './new_jot_confirm_modal.module.css';

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
}> = ({ open, onConfirm, onCancel }) => {
  if (!open) return null;

  return (
    <div
      className={styles.modalBackdrop}
      role="dialog"
      aria-modal="true"
      aria-label="Discard unsaved changes?"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
      data-testid="new-jot-confirm-modal"
    >
      <div className={styles.modalPanel}>
        <header className={styles.modalHeader}>
          <h3 className={styles.modalTitle}>Discard unsaved changes?</h3>
          <button
            type="button"
            className={styles.modalClose}
            onClick={onCancel}
            aria-label="Cancel"
            data-testid="new-jot-confirm-cancel"
          >
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className={styles.modalBody}>
          <p>
            This jot has changes you haven't saved. Starting a new jot will
            replace it, and this can't be undone.
          </p>
          <p className={styles.note}>Save first if you want to keep your work.</p>
        </div>
        <footer className={styles.modalFooter}>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={onCancel}
            autoFocus
          >
            Cancel
          </button>
          <button
            type="button"
            className={styles.dangerButton}
            onClick={onConfirm}
            data-testid="new-jot-confirm-discard"
          >
            Discard &amp; start new
          </button>
        </footer>
      </div>
    </div>
  );
};
