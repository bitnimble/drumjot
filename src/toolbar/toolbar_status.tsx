import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import { jotPlayer } from 'src/editing/playback/player';
import { SampleLoadProgress } from 'src/editing/playback/sample_storage';
import { TranscribeStage } from 'src/editing/transcribe/transcriber';
import sharedStyles from '../editing/jot_editor.module.css';
import type { TranscribeStatus } from 'src/editing/transcribe/transcribe_store';
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
export function sampleProgressWidth(
  phase: SampleLoadPhase | undefined,
  p: SampleLoadProgress | undefined
): string {
  if (phase === 'decoding') return '100%';
  if (phase === 'connecting' || !p) return '8%';
  if (p.fromCache) return '100%';
  return p.total > 0 ? `${samplePct(p)}%` : '40%';
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
      <span className={styles.sampleProgressTrack}>
        <span
          className={styles.sampleProgressFill}
          style={{ width: sampleProgressWidth(phase, progress) }}
        />
      </span>
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
        <span className={styles.statusPillSpinner} aria-hidden="true" />
        {queued ? 'Queued…' : 'Aligning lyrics…'}
      </span>
    );
  }
);

/**
 * Busy pill for an in-flight transcribe / resume call. Surfaces the
 * live pipeline stage (and substage detail, if any) alongside the
 * filename so the operator can see what the server is actually
 * working on; fed from the NDJSON progress stream via
 * `JotEditorStore.applyProgress`. Completion (success or failure) drops
 * back to nothing; the user-visible result surfaces as a toast.
 */
export const TranscribeBusyPill = observer(({ status }: { status: TranscribeStatus }) => {
  if (status.phase !== 'uploading') return null;
  const stagePart = status.stage
    ? ` · ${formatStageLabel(status.stage)}${status.substage ? ` (${status.substage})` : ''}`
    : '';
  return (
    <span
      className={classNames(sharedStyles.statusPill, sharedStyles.statusPillBusy)}
      title={status.substage ?? status.stage ?? 'starting'}
    >
      <span className={styles.statusPillSpinner} aria-hidden="true" />
      Transcribing {status.filename}
      {stagePart}…
    </span>
  );
});
