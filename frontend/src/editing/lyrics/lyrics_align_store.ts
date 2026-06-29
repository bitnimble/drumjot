import { makeAutoObservable } from 'mobx';
import { LyricsTrackId } from 'src/lyrics/store';

/** Long-running lyric-alignment indicator. `queued` is the wait state
 *  while the request sits behind another in-flight GPU job (a transcribe
 *  or another align); `aligning` is once it owns the GPU and forced
 *  alignment is actually running. Success and failure surface as toasts
 *  (see `./toasts.ts`). */
export type LyricsAlignStatus =
  | { phase: 'idle' }
  | { phase: 'queued'; detail: string }
  | { phase: 'aligning'; detail: string };

/**
 * Lyrics-alignment UI state: the per-track align status (drives the
 * per-row spinner + the toolbar busy pill) and the two lyrics-modal
 * visibility flags.
 *
 * Pure data: observables + the `lyricsAlignBusyPhase` computed. The align
 * orchestration (forced-alignment requests, the in-flight
 * `AbortController`s, LRCLIB / plain-text flows) lives on the presenter;
 * it writes `lyricsAlignStatuses` as work progresses.
 */
export class LyricsAlignStore {
  /**
   * Per-track alignment status. Each row aligning at the same time has
   * its own status entry; absence of an entry means that row is idle.
   * Observable so `lyricsAlignBusyPhase` and the per-row spinner
   * re-render on change.
   */
  lyricsAlignStatuses: Map<LyricsTrackId, LyricsAlignStatus> = new Map();
  /** Lyrics search modal visibility. */
  lyricsSearchOpen: boolean = false;
  /** Lyrics plain-text load modal visibility. */
  lyricsTextOpen: boolean = false;

  constructor() {
    makeAutoObservable(this);
  }

  /** Aggregate lyrics-alignment state across all rows, for the toolbar
   *  busy pill (which doesn't display *which* row; the per-row spinner
   *  does). `aligning` wins over `queued` so that once any row owns the
   *  GPU the pill reads as actively working; `queued` shows only while
   *  every in-flight row is still waiting its turn. The backend
   *  serialises GPU work, so at most one row is `aligning` at a time. */
  get lyricsAlignBusyPhase(): 'idle' | 'queued' | 'aligning' {
    let anyQueued = false;
    for (const s of this.lyricsAlignStatuses.values()) {
      if (s.phase === 'aligning') return 'aligning';
      if (s.phase === 'queued') anyQueued = true;
    }
    return anyQueued ? 'queued' : 'idle';
  }
}
