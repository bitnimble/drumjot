import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import { SampleLoadProgress } from 'src/editing/playback/sample_storage';
import { TranscribeStage } from 'src/editing/transcribe/transcriber';
import { TranscribeStoreContext } from 'src/editing/transcribe/transcribe_contexts';
import { ProgressBar } from 'src/ui/progress_bar/progress_bar';
import { Spinner } from 'src/ui/spinner/spinner';
import sharedStyles from '../editing/jot_editor.module.css';
import styles from './toolbar.module.css';

/** Human-readable label for one pipeline stage, used in the status
 *  pill. Mirrors the StrEnum values one-for-one but with friendlier
 *  wording where the raw identifier ("stems_per") reads worse than its
 *  description ("separating drum pieces"). */
export function formatStageLabel(stage: TranscribeStage): string {
  switch (stage) {
    case 'stems_all':
      return 'separating drums';
    case 'stems_per':
      return 'separating drum pieces';
    case 'beats':
      return 'tracking beats';
    case 'onsets':
      return 'detecting onsets';
    case 'filter':
      return 'filtering artifact onsets';
    case 'quantise':
      return 'quantising onsets';
    case 'transcribe':
      return 'rendering MIDI';
  }
}

export function samplePct(p: SampleLoadProgress): number {
  return Math.min(100, Math.round((p.loaded / p.total) * 100));
}

export function formatMb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

type SampleLoadPhase = 'connecting' | 'downloading' | 'decoding';

/** Bar fill width per phase. While decoding the bytes are all in, so we
 * pin the bar at 100%; the indeterminate "connecting" / unknown-total
 * fallbacks use a fixed sliver that reads as "working" rather than empty. */
export function sampleProgressFraction(
  phase: SampleLoadPhase | undefined,
  p: SampleLoadProgress | undefined
): number {
  if (phase === 'decoding') return 1;
  if (phase === 'connecting' || !p) return 0.08;
  if (p.fromCache) return 1;
  return p.total > 0 ? samplePct(p) / 100 : 0.4;
}

export function sampleProgressLabel(
  phase: SampleLoadPhase | undefined,
  p: SampleLoadProgress | undefined
): string {
  if (phase === 'connecting' || !p) return 'Drums · waiting for server…';
  if (phase === 'decoding') return 'Drums · decoding samples…';
  if (p.fromCache) return 'Drums · loading from cache';
  return p.total > 0
    ? `Drums · downloading ${formatMb(p.loaded)} / ${formatMb(p.total)}`
    : `Drums · downloading ${formatMb(p.loaded)}`;
}

/**
 * Top-right drum-sample download indicator. Reads `jotPlayer` directly so
 * the toolbar around it doesn't re-render on every progress tick.
 */
export const DrumLoadingIndicator = observer(() => {
  if (jotPlayer.state !== 'loading') return null;
  const progress = jotPlayer.sampleLoadProgress;
  const phase = jotPlayer.sampleLoadPhase;
  return (
    <span
      className={styles.sampleProgress}
      title="One-time download of the GeneralUser GS SoundFont (~30 MB). Cached in the browser after the first load — instant next time."
    >
      <ProgressBar
        className={styles.sampleProgressTrack}
        value={sampleProgressFraction(phase, progress)}
        ariaLabel={sampleProgressLabel(phase, progress)}
      />
      <span>{sampleProgressLabel(phase, progress)}</span>
    </span>
  );
});

/**
 * Busy pill for the Whisper lyric-alignment flow. Shown whenever any
 * lyrics row currently has an alignment in flight; the pill doesn't
 * surface *which* row (per-track aligns can run concurrently and the
 * per-row spinner already covers that), it just signals "the backend
 * is doing lyrics work". `queued` reads as a wait state (the GPU is busy
 * with another job) and flips to "Aligning lyrics…" once the work
 * actually starts. Returns to nothing on completion; success is
 * signalled by the row's lines upgrading, failure by an error toast.
 */
export const LyricsAlignBusyPill = observer(
  ({ phase }: { phase: 'idle' | 'queued' | 'aligning' }) => {
    if (phase === 'idle') return null;
    const queued = phase === 'queued';
    return (
      <span
        className={classNames(sharedStyles.statusPill, sharedStyles.statusPillBusy)}
        title={
          queued
            ? 'Waiting for the GPU (another job is running)…'
            : 'Extracting vocals + aligning lyrics…'
        }
        data-testid="lyrics-align-busy"
      >
        <Spinner size={10} tone="accent" className={styles.statusPillSpinner} />
        {queued ? 'Queued…' : 'Aligning lyrics…'}
      </span>
    );
  }
);

/**
 * Persistent top-right pill for any in-flight transcription, fed from the
 * {@link TranscribeStore} via context. Covers both flows: per-track `append`
 * (inserting into the current jot) and `replace` (recent / resume). Surfaces
 * the live pipeline stage + substage alongside the filename. When several
 * tracks transcribe at once it shows the first plus an "+N more" count; the
 * per-track gutter spinners show exactly which rows are working. Completion
 * drops back to nothing; the result surfaces as a toast.
 */
export const TranscribeBusyPill = observer(() => {
  const store = React.useContext(TranscribeStoreContext);
  if (!store) return null;
  const tracks = [...store.trackStatuses.values()];
  const replace = store.replaceStatus;
  // Prefer the per-track append flow; fall back to the replace flow.
  const active =
    tracks.length > 0
      ? tracks[0]
      : replace.phase === 'uploading'
        ? { filename: replace.filename, stage: replace.stage, substage: replace.substage }
        : undefined;
  if (!active) return null;
  const extra = tracks.length > 1 ? ` +${tracks.length - 1} more` : '';
  const stagePart = active.stage
    ? ` · ${formatStageLabel(active.stage)}${active.substage ? ` (${active.substage})` : ''}`
    : '';
  return (
    <span
      className={classNames(sharedStyles.statusPill, sharedStyles.statusPillBusy)}
      title={active.substage ?? active.stage ?? 'starting'}
      data-testid="transcribe-busy"
    >
      <span className={styles.statusPillSpinner} aria-hidden="true" />
      Transcribing {active.filename}
      {stagePart}
      {extra}…
    </span>
  );
});
