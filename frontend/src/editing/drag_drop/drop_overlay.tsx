import { Upload } from 'lucide-react';
import React from 'react';
import styles from './drop_overlay.module.css';

/**
 * Full-window hint shown while an OS file drag hovers the editor. Purely
 * visual: `pointer-events: none` so the drag/drop events still land on the
 * app-shell container that owns the handlers (see {@link useFileDrop}).
 */
export const DropOverlay: React.FC<{ active: boolean }> = ({ active }) => {
  if (!active) return null;
  return (
    <div className={styles.overlay} aria-hidden="true" data-testid="file-drop-overlay">
      <div className={styles.card}>
        <Upload size={32} aria-hidden="true" />
        <div className={styles.label}>Drop to load</div>
        <div className={styles.hint}>
          .jot · audio · MIDI · lyrics · ParaDB / debug .zip
        </div>
      </div>
    </div>
  );
};
