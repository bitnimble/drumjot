import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import { JotTimeline } from 'src/editing/playback/timeline';
import styles from './playback.module.css';
import { formatPlayheadTime, playheadBarBeat } from './playhead_label';

/**
 * Travelling playback marker. Rendered in three places (timeline header,
 * each audio-track row, each instrument row) so the playhead line is
 * visible across the whole vertical stack.
 *
 * Position is NOT driven from this component; it's read from the
 * `--playhead-x` CSS custom property that `PlayheadPosVar` writes once
 * per frame on each `[data-playhead="1"]` element (see jot_editor.tsx).
 * The var is registered `inherits: false`, so the per-tick `setProperty`
 * has to be made per-playhead; `PlayheadPosVar` iterates a cached
 * `DomTargetCache.playheads` Set (maintained via a single
 * `MutationObserver` on the JotEditor root) instead of `querySelectorAll`-ing
 * every frame. The shell only re-renders when `state` / `cued` / `timeline`
 * change (i.e. transport events, not per-frame playback) so a debug bundle
 * with many tracks doesn't pay N × (reconciliation + new style object +
 * new closure) every frame the way it used to.
 *
 * Drag-to-scrub still works because the mousedown handler measures
 * against the parent bars-row's bounding rect, which is unaffected by
 * this element's own transform.
 */
export const Playhead = observer(
  ({
    showLabel = false,
    onSeek,
  }: {
    showLabel?: boolean;
    onSeek: (x: number) => void;
  }) => {
    const timeline = jotPlayer.timeline;
    const active =
      jotPlayer.state === 'playing' ||
      jotPlayer.state === 'paused' ||
      // Idle but the user clicked to position the playhead before
      // pressing Play; show it parked at the cued spot.
      jotPlayer.cued;
    if (!active || timeline.bars.length === 0) return null;

    const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      const parent = e.currentTarget.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      onSeek(e.clientX - rect.left);
      const onMove = (ev: MouseEvent) => {
        onSeek(ev.clientX - rect.left);
      };
      const onUp = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    };

    return (
      <div
        className={styles.playhead}
        onMouseDown={onMouseDown}
        data-noseek
        data-playhead="1"
      >
        {showLabel && <PlayheadLabel timeline={timeline} />}
      </div>
    );
  }
);

/**
 * Time / bar-beat readout shown on top of the timeline-header playhead.
 * Only ONE instance ever mounts (the per-row playheads pass
 * `showLabel={false}`), so reading `jotPlayer.currentTime` here gives
 * us a per-frame re-render of a tiny tree (two text nodes) and nothing
 * else, no shell rerenders, no per-row label rerenders.
 */
const PlayheadLabel = observer(({ timeline }: { timeline: JotTimeline }) => {
  const t = jotPlayer.currentTime;
  const pos = playheadBarBeat(timeline, t);
  return (
    <div className={styles.playheadLabel}>
      <div>{formatPlayheadTime(t)}</div>
      {pos && <div className={styles.playheadLabelBarBeat}>{pos}</div>}
    </div>
  );
});
