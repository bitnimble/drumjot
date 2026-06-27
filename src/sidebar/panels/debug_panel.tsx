import React from 'react';
import { DebugView } from 'src/editing/provenance/debug_view';
import styles from '../sidebar.module.css';

/**
 * Debug panel: the panel header + the {@link DebugView} body (run summary,
 * per-stage timings, and the captured log stream from a loaded transcriber
 * debug bundle). Empty until a bundle is loaded via `Load > Load zip`.
 */
export function DebugPanel() {
  return (
    <div className={styles.panelBody} data-testid="debug-panel">
      <h2 className={styles.panelTitle}>Debug</h2>
      <DebugView />
    </div>
  );
}
