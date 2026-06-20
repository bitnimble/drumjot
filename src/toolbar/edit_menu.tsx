import { observer } from 'mobx-react-lite';
import React from 'react';
import {
  ActionMenuItem,
  DropdownButton,
  DropdownSection,
  ToggleMenuItem,
} from 'src/ui/dropdown/dropdown';
import {
  EditingStoreContext,
  EditingPresenterContext,
} from 'src/editing/editing_contexts';
import {
  HistoryStoreContext,
  HistoryPresenterContext,
} from 'src/editing/history/history_contexts';
import { shortcutForCommand } from 'src/editing/keyboard/keymap';
import { ToolbarDropdownLabel } from './toolbar';
import styles from './toolbar.module.css';

/**
 * The "Edit" toolbar dropdown: undo/redo + note-editing options. Self-contained
 * `observer` that reads the editing + history stores/presenters off context
 * (the Toolbar renders inside their providers), so it needs no prop plumbing
 * through the app shell.
 *
 * Undo/Redo mirror Loro's UndoManager availability ({@link HistoryStore}); each
 * row is disabled but stays visible when its stack is empty, and shows its
 * keyboard shortcut pulled from the keymap registry (so a rebind reflects here
 * automatically rather than being hardcoded).
 *
 * Snapping targets the grid at the resolution of whichever grid-line families
 * are currently enabled (View → Grid lines), the union of their lines, and
 * applies to both inserting and moving notes.
 */
export const EditMenu = observer(() => {
  const editing = React.useContext(EditingStoreContext);
  const presenter = React.useContext(EditingPresenterContext);
  const history = React.useContext(HistoryStoreContext);
  const historyPresenter = React.useContext(HistoryPresenterContext);
  if (!editing || !presenter) return null;
  const snapping = editing.snappingEnabled;
  return (
    <DropdownButton
      label={<ToolbarDropdownLabel>Edit</ToolbarDropdownLabel>}
      className={styles.playButton}
      title="Undo / redo and note-editing options."
    >
      {(close) => (
        <>
          <DropdownSection label="History">
            <ActionMenuItem
              label="Undo"
              disabled={!history?.canUndo}
              shortcut={shortcutForCommand('undo')}
              onClick={() => {
                historyPresenter?.undo();
                close();
              }}
              testId="edit-menu-undo"
              title="Undo the last edit."
            />
            <ActionMenuItem
              label="Redo"
              disabled={!history?.canRedo}
              shortcut={shortcutForCommand('redo')}
              onClick={() => {
                historyPresenter?.redo();
                close();
              }}
              testId="edit-menu-redo"
              title="Redo the last undone edit."
            />
          </DropdownSection>
          <DropdownSection label="Snapping">
            <ToggleMenuItem
              label="Enable snapping"
              active={snapping}
              onToggle={() => presenter.setSnapping(!snapping)}
              testId="edit-menu-snapping"
              title="Snap inserted and moved notes to the grid, at the resolution of the currently-enabled grid-line families (View → Grid lines; the union of their lines). Off = free placement."
            />
          </DropdownSection>
        </>
      )}
    </DropdownButton>
  );
});
