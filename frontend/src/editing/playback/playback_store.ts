import { makeAutoObservable } from 'mobx';
import type { MixerStore } from '../mixer/mixer_store';
import { PASSTHROUGH_FILTER, type PlayerFilter } from './player';
import { PASSTHROUGH_AUDIO_TRACK_FILTER, type AudioTrackFilter } from './audio_tracks';
import { JotEditorStore } from '../jot_editor_store';
import { type Epochs, makeEpochs } from './epochs';

/**
 * Transport / playhead-follow UI state, plus the engine-facing
 * mute/solo/volume **filter computeds** the player pulls directly.
 *
 * Pure data: observables + computeds. The transport orchestration
 * (play/pause/seek, drum-offset reschedule, follow re-enable logic) and
 * the reactions that re-apply the filter to the audio graph live on the
 * presenter; this store only derives state.
 *
 * The filter getters ({@link laneFilter} / {@link audioTrackFilter} /
 * the section-audible booleans) are computed over the {@link MixerStore}'s
 * authoritative mute/solo/volume state. `jotPlayer` reads them directly
 * (via {@link JotPlayer.attachPlayback}); a `PlaybackPresenter` reaction
 * fires the imperative reschedule / gain re-apply when they change. When
 * no mixer is wired (stories / a standalone engine) they fall back to the
 * PASSTHROUGH (everything audible) filters.
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

  /**
   * The `songLeadIn` epoch (jot seconds, <= 0): the jot time at which the
   * recorded audio begins, i.e. the recording's pre-drum lead-in. The live,
   * user-tunable audio↔drum alignment. The audio engine reads it (mirrored as
   * `jotPlayer.songLeadInSec`) and maps every audio track via `media = jot -
   * songLeadIn`; the renderer/waveform/lyrics read it for the same mapping.
   *
   * The single source of truth: seeded from each loaded jot's
   * `globalMetadata.songLeadIn` and nudged by the Offset / Beat-offset
   * controls, both through `PlaybackPresenter.setSongLeadIn` (the only
   * writer, which clamps to <= 0 and asks the engine to reposition audio).
   */
  songLeadInSec: number = 0;

  /** The active jot, for the drum-offset readout. */
  readonly jotEditorStore: JotEditorStore;
  /** Authoritative mute/solo/volume source for the filter computeds.
   *  Undefined in stories / a standalone engine → PASSTHROUGH filters. */
  readonly mixer: MixerStore | undefined;

  constructor(jotEditorStore: JotEditorStore, mixer?: MixerStore) {
    this.jotEditorStore = jotEditorStore;
    this.mixer = mixer;
    makeAutoObservable(this, { jotEditorStore: false, mixer: false });
  }

  /** Current beat-grid offset (quarter-note beats) on the loaded jot. */
  get drumOffsetBeats(): number {
    return this.jotEditorStore.structural?.drumOffsetBeats ?? 0;
  }

  /**
   * The song's time anchors in jot seconds (bar-1 downbeat = 0):
   * `{ drums: 0, songLeadIn, fullLeadIn }`, with `fullLeadIn <= songLeadIn
   * <= drums`. The named replacement for the old scattered `drumsT0Sec`
   * arithmetic; map audio↔jot with `jotToMedia` / `mediaToJot`.
   *
   *  - `songLeadIn` is the live {@link songLeadInSec} (audio start).
   *  - `fullLeadIn` is the rendered left edge, the first bar's start in the
   *    tempo timeline, which includes the view-only virtual lead-in bar, and
   *    is the leftmost seekable / visible point. It falls back to `songLeadIn`
   *    before a timeline exists. Read off the document's derived `tempoTimeline`
   *    (always available once loaded, unlike the engine's live timeline which is
   *    empty while idle), so this store needs no reference to the tempo peer.
   */
  get epochs(): Epochs {
    const fullLeadIn = this.jotEditorStore.jot?.tempoTimeline.bars[0]?.startSec ?? this.songLeadInSec;
    return makeEpochs(this.songLeadInSec, fullLeadIn);
  }

  /**
   * Live {@link PlayerFilter} the drum scheduler reads, delegated from the
   * mixer (which owns the build + also consumes it for its own
   * per-row-audibility computeds). PASSTHROUGH when no mixer is wired.
   */
  get trackFilter(): PlayerFilter {
    return this.mixer?.trackFilter ?? PASSTHROUGH_FILTER;
  }

  /** Mirror of {@link trackFilter} for the audio-track domain. */
  get audioTrackFilter(): AudioTrackFilter {
    return this.mixer?.audioTrackFilter ?? PASSTHROUGH_AUDIO_TRACK_FILTER;
  }

  /** Whether the drum bus is audible (master mute/solo + cross-domain
   *  solo folded in). Delegates to the mixer's section computed, which the
   *  master-row UI also reads. True when no mixer is wired. */
  get drumMasterAudible(): boolean {
    return this.mixer?.isDrumSectionAudible ?? true;
  }

  /** Mirror of {@link drumMasterAudible} for the audio-track bus. */
  get audioMasterAudible(): boolean {
    return this.mixer?.isAudioSectionAudible ?? true;
  }
}
