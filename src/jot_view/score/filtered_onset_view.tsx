import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { NoteProvenanceEntry } from 'src/jot_view/provenance/debug_zip';
import { ProvenancePresenterContext, ProvenanceStoreContext } from '../provenance/provenance_contexts';
import styles from './score.module.css';
import { NoteProvenanceDetails } from './note_provenance_details';
import { PopoverPortal } from './popover_portal';

/**
 * Renders one rejected onset as a dashed ghost circle at its detected
 * `(bar, beat_in_bar)` position inside an `InstrumentRow`'s bars row.
 * Absolutely positioned via the same `--note-pad-px` / `--px-per-beat`
 * CSS vars the real notes use, but with `--filtered-beat` = the
 * onset's cumulative beat offset from the start of the row (lead-in +
 * prior bars + intra-bar offset) so it lands at the right absolute x
 * without needing per-bar ResolvedBar geometry.
 *
 * Click toggles a stuck-open detail popover (independent of the
 * SelectionStore — a filtered onset is not a real note); hover shows
 * the same popover transiently.
 */
export const FilteredOnsetView = observer(({
  entry,
  beatOffset,
  color,
  trackHeight,
}: {
  entry: NoteProvenanceEntry;
  /** Total beat offset from the start of the bars row (leadInBeats +
   * cumulative bar beats + (beat_in_bar - 1)). The CSS calc derives
   * the pixel `left` from this and the score-root's `--px-per-beat`. */
  beatOffset: number;
  /** Pitch lane colour. Mirrors what the real notes use; falls back to
   * a neutral grey for filtered-only pitches with no rendered notes. */
  color: string;
  trackHeight: number;
}) => {
  const provenance = React.useContext(ProvenanceStoreContext);
  const presenter = React.useContext(ProvenancePresenterContext);
  const pinnedKey = `${entry.pitch}:${entry.detected_time_sec}`;
  const clicked = provenance?.pinnedFilteredOnsetKey === pinnedKey;
  const [hovered, setHovered] = React.useState(false);
  const show = hovered || clicked;
  const stop = (e: React.MouseEvent) => e.stopPropagation();
  const anchorRef = React.useRef<HTMLDivElement>(null);
  const labelRef = React.useRef<HTMLDivElement | null>(null);
  // Click-outside to dismiss the stuck-open popover. Without this the
  // only way to close it is clicking the (small, easy-to-miss) dashed
  // ring again. We treat clicks anywhere outside the anchor OR its
  // label as "outside" so users can interact with the popover itself
  // (e.g. expand Debug details) without dismissing it.
  //
  // Registered in capture phase so the listener runs before React's
  // root-level click delegation; calling stopPropagation prevents the
  // dismissing click from also moving the playhead via the bars-row
  // seek handler (or any other bubbling onClick further up the tree).
  React.useEffect(() => {
    if (!clicked || !presenter) return;
    const onClickCapture = (e: MouseEvent) => {
      const target = e.target as Node;
      if (anchorRef.current?.contains(target)) return;
      // The label is portaled to `document.body`, so `target.contains`
      // would walk a different subtree; keep the same "is this click
      // inside the popover?" check by comparing against `labelRef`
      // which still holds the portaled element.
      if (labelRef.current?.contains(target)) return;
      e.stopPropagation();
      presenter.setPinnedFilteredOnsetKey(undefined);
    };
    document.addEventListener('click', onClickCapture, true);
    return () => document.removeEventListener('click', onClickCapture, true);
  }, [clicked, presenter]);
  return (
    <div
      ref={anchorRef}
      // Same opt-out as real notes so a click on the ghost doesn't move
      // the playhead via the bars-row seek handler.
      data-noseek="true"
      className={classNames(styles.filteredOnset, show && styles.filteredOnsetShowingLabel)}
      style={
        {
          ['--filtered-beat' as string]: beatOffset,
          top: trackHeight / 2,
          color,
        } as React.CSSProperties
      }
      onMouseDown={stop}
      onClick={(e) => {
        stop(e);
        presenter?.setPinnedFilteredOnsetKey(clicked ? undefined : pinnedKey);
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={`Filtered onset · pitch ${entry.pitch} · bar ${entry.bar} beat ${entry.beat_in_bar.toFixed(2)}`}
    >
      <PopoverPortal
        anchorRef={anchorRef}
        show={show}
        className={styles.filteredOnsetLabel}
        extraProps={{ ref: labelRef }}
      >
        <NoteProvenanceDetails entry={entry} startOpen />
      </PopoverPortal>
    </div>
  );
});
