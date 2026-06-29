import React from 'react';
import { describeDocumentLoad, DropPlan } from 'src/editing/drag_drop/file_routing';
import { ConfirmModal, modalStyles } from 'src/ui/modal/modal';

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
    <ConfirmModal
      open
      onConfirm={onConfirm}
      onCancel={onCancel}
      title="Replace current score?"
      ariaLabel="Replace current score?"
      confirmLabel="Replace"
      confirmVariant="danger"
      autoFocus="confirm"
      width={440}
      testId="drop-confirm-modal"
      closeTestId="drop-confirm-cancel-x"
      cancelTestId="drop-confirm-cancel"
      confirmTestId="drop-confirm-replace"
    >
      <p>
        {describeDocumentLoad(plan.documentLoad)}. This discards the
        currently-loaded score and its edits.
      </p>
      {(audioCount > 0 || lyricsCount > 0) && (
        <p className={modalStyles.note}>
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
    </ConfirmModal>
  );
};
