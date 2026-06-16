import React from 'react';
import styles from '../sidebar.module.css';

/**
 * Layers panel. Intentionally a blank stub, the actual layers view (reorder,
 * rename, show/hide `||` layers) lands in a later PR. Renders only the panel
 * header so the open state reads as "Layers".
 */
export function LayersPanel() {
  return (
    <div className={styles.panelBody} data-testid="layers-panel">
      <h2 className={styles.panelTitle}>Layers</h2>
    </div>
  );
}
