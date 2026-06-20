import { X } from 'lucide-react';
import React from 'react';
import { describeDocumentLoad, DropPlan } from 'src/editing/drag_drop/file_routing';
import styles from './drop_confirm_modal.module.css';

/**
 * Confirm dialog shown when a dropped file would replace the open score
 * (a `.jot` / MIDI / ParaDB / debug-bundle drop while a song is loaded).
 * Dropping audio / lyrics is additive and never reaches here. Open exactly
 * when `plan` is non-null.
 */
export const DropConfirmModal: React.FC<{
  plan: DropPlan | null;
  onConfirm: () => void;
  onCancel: () => void;
}> = ({ plan, onConfirm, onCancel }) => {
  if (!plan || !plan.documentLoad) return null;
  const audioCount = plan.additive.filter((a) => a.kind === 'audio').length;
  const lyricsCount = plan.additive.filter((a) => a.kind === 'lyrics').length;

  return (
    <div
      className={styles.backdrop}
      role="dialog"
      aria-modal="true"
      aria-label="Replace current score?"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
      data-testid="drop-confirm-modal"
    >
      <div className={styles.panel}>
        <header className={styles.header}>
          <h3 className={styles.title}>Replace current score?</h3>
          <button
            type="button"
            className={styles.close}
            onClick={onCancel}
            aria-label="Cancel"
            data-testid="drop-confirm-cancel-x"
          >
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className={styles.body}>
          <p>
            {describeDocumentLoad(plan.documentLoad)}. This discards the
            currently-loaded score and its edits.
          </p>
          {(audioCount > 0 || lyricsCount > 0) && (
            <p className={styles.note}>
              Also adding{' '}
              {[
                audioCount > 0 && `${audioCount} audio track${audioCount === 1 ? '' : 's'}`,
                lyricsCount > 0 && `${lyricsCount} lyrics track${lyricsCount === 1 ? '' : 's'}`,
              ]
                .filter(Boolean)
                .join(' and ')}
              .
            </p>
          )}
        </div>
        <footer className={styles.footer}>
          <button
            type="button"
            className={styles.cancelButton}
            onClick={onCancel}
            data-testid="drop-confirm-cancel"
          >
            Cancel
          </button>
          <button
            type="button"
            className={styles.confirmButton}
            onClick={onConfirm}
            autoFocus
            data-testid="drop-confirm-replace"
          >
            Replace
          </button>
        </footer>
      </div>
    </div>
  );
};
