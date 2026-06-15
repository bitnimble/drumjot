import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import { MousePointer2 } from 'lucide-react';
import React from 'react';
import { EditingStoreContext, EditingPresenterContext } from './editing_contexts';
import type { EditMode } from './editing_store';
import styles from './editing_toolbar.module.css';

/**
 * A jot note (filled circle) with a small plus badge at its top-right
 * corner, the insert-mode glyph. Drawn inline rather than composed from
 * two lucide icons so the plus sits exactly on the circle's corner.
 */
function InsertNoteIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable="false">
      <circle cx="8" cy="10" r="5.5" fill="currentColor" />
      {/* Plus badge, top-right. Stroked in currentColor over a base-coloured
          disc so it reads as a distinct badge on top of the note. */}
      <circle cx="14" cy="4" r="3.5" fill="var(--color-bg-base)" />
      <path
        d="M14 2.2v3.6M12.2 4h3.6"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * Vertical floating toolbar pinned to the right edge of the page, with the
 * two editing-mode toggles (select / insert). Reads the current mode off
 * {@link EditingStoreContext} and switches it via {@link EditingPresenterContext}.
 */
export const EditingToolbar = observer(function EditingToolbar() {
  const store = React.useContext(EditingStoreContext);
  const presenter = React.useContext(EditingPresenterContext);
  if (!store || !presenter) return null;

  const button = (mode: EditMode, label: string, icon: React.ReactNode) => (
    <button
      type="button"
      className={classNames(styles.modeButton, store.mode === mode && styles.modeButtonActive)}
      aria-pressed={store.mode === mode}
      title={label}
      aria-label={label}
      data-testid={`mode-${mode}`}
      onClick={() => presenter.setMode(mode)}
    >
      {icon}
    </button>
  );

  return (
    <div className={styles.toolbar} data-testid="editing-toolbar">
      {button('select', 'Select mode', <MousePointer2 size={18} />)}
      {button('insert', 'Insert note mode', <InsertNoteIcon />)}
    </div>
  );
});
