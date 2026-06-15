import { observer } from 'mobx-react-lite';
import React from 'react';
import { EditingStoreContext } from '../editing_contexts';
import styles from './score.module.css';

/**
 * Insert-mode preview note for one instrument row's bars row. An isolated
 * `observer` so a cursor move (which rewrites `editingStore.placeholder` on
 * every pointer move) re-renders only this tiny element, never the memoised
 * bar list. Renders nothing unless the placeholder belongs to this row's lane.
 *
 * Positioned by the same `--placeholder-beat` / `--layer-beats` calc the real
 * notes and filtered-onset overlay use, so it lands exactly where a committed
 * note would draw. Lane colour + note size come in as inline style.
 */
export const PlaceholderNoteView = observer(function PlaceholderNoteView({
  rowLane,
  color,
  trackHeight,
  noteDiameter,
}: {
  rowLane: string;
  color: string;
  trackHeight: number;
  noteDiameter: number;
}) {
  const editing = React.useContext(EditingStoreContext);
  const placeholder = editing?.placeholder;
  if (!placeholder || placeholder.lane !== rowLane) return null;
  return (
    <div
      className={styles.placeholderNote}
      data-testid="placeholder-note"
      style={
        {
          ['--placeholder-beat' as string]: placeholder.absBeat,
          top: trackHeight / 2,
          width: noteDiameter,
          height: noteDiameter,
          color,
        } as React.CSSProperties
      }
    />
  );
});
