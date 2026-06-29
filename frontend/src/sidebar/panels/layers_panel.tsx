import React from 'react';
import { LayersView } from 'src/editing/layers/layers_view';
import styles from '../sidebar.module.css';

/**
 * Layers panel: the panel header + the read-only {@link LayersView} tree
 * (layers → groups → tracks) mirroring the score's row layout. Drag-and-drop,
 * colour pickers and the ⋯ menus land in later phases.
 */
export function LayersPanel() {
  return (
    <div className={styles.panelBody} data-testid="layers-panel">
      <h2 className={styles.panelTitle}>Layers</h2>
      <LayersView />
    </div>
  );
}
