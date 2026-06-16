import { observer } from 'mobx-react-lite';
import React from 'react';
import { DropdownButton, DropdownSection, ToggleMenuItem } from 'src/ui/dropdown/dropdown';
import {
  EditingStoreContext,
  EditingPresenterContext,
} from 'src/editing/editing_contexts';
import { ToolbarDropdownLabel } from './toolbar';
import styles from './toolbar.module.css';

/**
 * The "Edit" toolbar dropdown: note-editing options. Self-contained
 * `observer` that reads the editing store/presenter off context (the Toolbar
 * renders inside their providers), so it needs no prop plumbing through the
 * app shell.
 *
 * Snapping targets the grid at the resolution of whichever grid-line families
 * are currently enabled (View → Grid lines), the union of their lines, and
 * applies to both inserting and moving notes.
 */
export const EditMenu = observer(() => {
  const editing = React.useContext(EditingStoreContext);
  const presenter = React.useContext(EditingPresenterContext);
  if (!editing || !presenter) return null;
  const snapping = editing.snappingEnabled;
  return (
    <DropdownButton
      label={<ToolbarDropdownLabel>Edit</ToolbarDropdownLabel>}
      className={styles.playButton}
      title="Note-editing options."
    >
      {() => (
        <DropdownSection label="Snapping">
          <ToggleMenuItem
            label="Enable snapping"
            active={snapping}
            onToggle={() => presenter.setSnapping(!snapping)}
            testId="edit-menu-snapping"
            title="Snap inserted and moved notes to the grid, at the resolution of the currently-enabled grid-line families (View → Grid lines; the union of their lines). Off = free placement."
          />
        </DropdownSection>
      )}
    </DropdownButton>
  );
});
