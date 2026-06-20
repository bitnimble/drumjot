import { observer } from 'mobx-react-lite';
import React from 'react';
import { NotePropertiesView } from 'src/editing/note_properties/note_properties_view';
import { NotePropertiesStoreContext } from 'src/editing/note_properties/note_properties_contexts';
import styles from '../sidebar.module.css';
import panelStyles from './note_properties_panel.module.css';

/**
 * Note properties panel: the header + a read-only id line (the selected note's
 * id, or a count hint for many) + the editable {@link NotePropertiesView}.
 */
export const NotePropertiesPanel = observer(function NotePropertiesPanel() {
  const store = React.useContext(NotePropertiesStoreContext);
  const idLabel = store?.noteIdLabel;
  return (
    <div className={styles.panelBody} data-testid="note-properties-panel">
      <h2 className={styles.panelTitle}>Note properties</h2>
      {idLabel && (
        <p className={panelStyles.noteId} data-testid="note-properties-id">
          {idLabel}
        </p>
      )}
      <NotePropertiesView />
    </div>
  );
});
