import { makeAutoObservable } from 'mobx';
import { DocumentStore } from '../document/document_store';

/**
 * Transport / playhead-follow UI state. Pure data: observables + one
 * computed derived from the loaded jot. The transport orchestration
 * (play/pause/seek, drum-offset reschedule, follow re-enable logic) lives
 * on the presenter and is the only thing that writes these.
 */
export class PlaybackStore {
  /**
   * When true, the score auto-scrolls horizontally during playback to
   * keep the playhead pinned to the viewport's centre. Toggle off via the
   * button above the playhead label to scroll freely while playing.
   * Session-only; resets to true on reload.
   */
  followPlayhead: boolean = true;
  /**
   * When true, transitioning to the playing state re-enables
   * {@link followPlayhead} if the user disabled it *during* the previous
   * playback session (pan, minimap drag, or the follow-button toggle
   * while playing). An off-state set while idle/paused is treated as
   * deliberate and survives the play press. Session-only, defaults on.
   */
  autoFollowOnPlay: boolean = true;
  /**
   * Internal: was the current `followPlayhead === false` set during
   * playback (transient, eligible for auto-re-enable on next play) or
   * during idle/paused (deliberate, must survive). Always false while
   * `followPlayhead` is true. Written by the presenter's follow logic.
   */
  followDisabledIsTransient: boolean = false;

  /** The active jot, for the drum-offset readout. */
  readonly document: DocumentStore;

  constructor(document: DocumentStore) {
    this.document = document;
    makeAutoObservable(this, { document: false });
  }

  /** Current beat-grid offset (quarter-note beats) on the loaded jot. */
  get drumOffsetBeats(): number {
    return this.document.currentJot?.drumOffsetBeats ?? 0;
  }
}
