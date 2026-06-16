import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import { MousePointer2, Plus } from 'lucide-react';
import React from 'react';
import { EditingStoreContext, EditingPresenterContext } from './editing_contexts';
import type { EditMode } from './editing_store';
import styles from './editing_toolbar.module.css';

/**
 * The insert-mode glyph: a jot note (a filled circle `div`) with a lucide
 * `Plus` badge at its top-right corner. Composed from a styled element + the
 * shared icon set rather than a hand-drawn SVG; positioning lives in the CSS.
 */
function InsertNoteIcon() {
  return (
    <span className={styles.insertIcon} aria-hidden="true">
      <span className={styles.insertIconDot} />
      <Plus className={styles.insertIconPlus} size={11} strokeWidth={2.5} />
    </span>
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
