import { comparer, makeAutoObservable, reaction } from 'mobx';
import { jotPlayer } from 'src/editing/playback/player';
import { xToTime } from 'src/editing/playback/timeline';
import { JotEditorStore } from '../jot_editor_store';
import { PlaybackStore } from './playback_store';

/**
 * Transport + playhead-follow orchestration over {@link PlaybackStore}.
 * The sole writer of the follow-playhead flags and the bridge between the
 * UI's transport controls and the {@link jotPlayer} singleton. Reads
 * {@link JotEditorStore} for the loaded song's peers (play / seek /
 * drum-offset all need the laid-out `structural` + `tempo`).
 */
export class PlaybackPresenter {
  readonly playback: PlaybackStore;
  readonly jotEditorStore: JotEditorStore;

  constructor(playback: PlaybackStore, jotEditorStore: JotEditorStore) {
    this.playback = playback;
    this.jotEditorStore = jotEditorStore;
    makeAutoObservable(this, { playback: false, jotEditorStore: false });
    // Seed the player's live drum↔audio offset from each loaded jot's
    // transcribed lead-in (`globalMetadata.drumsT0Sec`). Tracking
    // `document.source` (an observable reference) re-fires whenever a new
    // jot is loaded, resetting the offset to that recording's value; manual
    // nudges via the Offset control persist until the next load. We read
    // the raw `source.globalMetadata` (not a laid-out peer) so seeding
    // doesn't force a layout pass.
    reaction(
      () => {
        const raw = this.jotEditorStore.source?.globalMetadata.drumsT0Sec;
        return typeof raw === 'number' && raw > 0 ? raw : 0;
      },
      (offsetSec) => jotPlayer.setDrumsT0Sec(offsetSec),
      { fireImmediately: true }
    );

    // Pull-model filter wiring. The engine reads its mute/solo/volume
    // filter + section-audibility off the PlaybackStore computeds directly
    // (via `attachPlayback`); these reactions only fire the *imperative*
    // re-apply on the live audio graph when those computeds change.
    //
    // They MUST be `reaction`s, not `autorun`s: each effect reads AND
    // writes player observables (the reschedule reads `state` /
    // `currentTime` and rewrites the schedule), so an autorun would depend
    // on what it writes and never converge. The filter getters snapshot
    // their Sets/Maps so `comparer.structural` can see real changes.
    // `fireImmediately` seeds the graph; all four are no-ops until a
    // context / playback exists.
    jotPlayer.attachPlayback(this.playback);
    reaction(() => this.playback.laneFilter, () => jotPlayer.applyLaneFilter(), {
      fireImmediately: true,
      equals: comparer.structural,
    });
    reaction(() => this.playback.audioTrackFilter, () => jotPlayer.applyAudioTrackFilter(), {
      fireImmediately: true,
      equals: comparer.structural,
    });
    reaction(() => this.playback.audioMasterAudible, () => jotPlayer.applyAudioBusGain(), {
      fireImmediately: true,
    });
    reaction(() => this.playback.drumMasterAudible, () => jotPlayer.applyDrumBusGain(), {
      fireImmediately: true,
    });
  }

  setAutoFollowOnPlay(on: boolean) {
    this.playback.autoFollowOnPlay = on;
  }

  toggleFollowPlayhead() {
    this.setFollowPlayhead(!this.playback.followPlayhead);
  }

  /**
   * Set {@link PlaybackStore.followPlayhead} and tag whether the off-state
   * is transient (set while playing) or deliberate (set while idle/paused).
   * Idempotent: redundant calls don't reshuffle the transient tag so e.g.
   * a pan during playback can't promote an already-deliberate off-state
   * into a transient one.
   */
  setFollowPlayhead(on: boolean) {
    if (on === this.playback.followPlayhead) return;
    this.playback.followPlayhead = on;
    this.playback.followDisabledIsTransient = on ? false : jotPlayer.state === 'playing';
  }

  async playCurrent(): Promise<void> {
    const { structural, tempo } = this.jotEditorStore;
    if (!structural || !tempo) return;
    // Pass the laid-out structural presenter + tempo (not the raw source) so
    // the player's timeline reads live bar widths, the playhead then tracks
    // correctly across zoom changes.
    await jotPlayer.play(structural, tempo);
  }

  stopPlayback(): void {
    jotPlayer.stop();
  }

  /**
   * Slide every drum note across the bar grid by `beats` quarter-note
   * beats to realign a consistently mis-detected groove (see
   * `StructuralPresenter.drumOffsetBeats`). Reflows the score reactively and
   * reschedules in-flight playback so the change is heard immediately.
   */
  setDrumOffset(beats: number): void {
    const { structural, tempo } = this.jotEditorStore;
    if (!structural || !tempo) return;
    // Slider semantics: the user is re-labeling note positions on the
    // notational grid (e.g. "this hit is on 1/48, not 3/48"), not
    // re-timing the drums against the audio recording. So when the
    // shift moves every note by Δ beats in jot time, compensate the
    // audio offset by the same magnitude in the opposite direction so
    // the audio-track waveform tracks the noteheads instead of sliding
    // out from under them. Uses the dominant bpm (the tempo the song
    // spends the most audio time at, excluding lead-in bars) rather
    // than globalMetadata.bpm, because transcribed bundles store a
    // back-solved lead-in tempo as the first setTempo event and that
    // value can be very different from the song's actual rate. Per-bar
    // tempo variation still leaves a few-ms-per-note residual; same
    // caveat as the Drum-offset row in the debug panel.
    const deltaBeats = beats - structural.drumOffsetBeats;
    if (Math.abs(deltaBeats) > 1e-12) {
      const { dominantBpm } = tempo.dominantBpmAndTime;
      const bpm = dominantBpm ?? 120;
      const deltaSec = (deltaBeats * 60) / bpm;
      jotPlayer.setDrumsT0Sec(jotPlayer.drumsT0Sec - deltaSec);
    }
    structural.setDrumOffset(beats);
    jotPlayer.refreshDrumSchedule(structural);
  }

  /**
   * Click-to-seek. `x` is a pixel offset within the bars row, the same
   * coordinate space `bar.x` / the playhead use (origin at the left
   * edge of the bars region, after the gutter). While playing this
   * scrubs live; while idle it parks the playhead and the next Play
   * starts from there. Uses the live timeline when one exists so a
   * mid-playback scrub reads the exact bars being played.
   */
  seekToX(x: number): void {
    const { tempo } = this.jotEditorStore;
    if (!tempo) return;
    const timeline = jotPlayer.timeline.bars.length > 0 ? jotPlayer.timeline : tempo.timeline;
    jotPlayer.seek(tempo, xToTime(timeline, x));
  }

  /**
   * Single transport action shared by the spacebar shortcut and the
   * toolbar's play/pause button:
   *   idle    -> play the current jot from the start
   *   playing -> pause (freezes the clock, playhead stays put)
   *   paused  -> resume from the same spot
   * `loading` is intentionally a no-op so a double-press during the
   * one-time sample fetch can't stack two `play()` calls.
   */
  async togglePlayPause(): Promise<void> {
    switch (jotPlayer.state) {
      case 'idle':
        this.maybeReenableFollowOnPlay();
        await this.playCurrent();
        break;
      case 'playing':
        await jotPlayer.pause();
        break;
      case 'paused':
        this.maybeReenableFollowOnPlay();
        await jotPlayer.resume();
        break;
    }
  }

  /**
   * Restore {@link PlaybackStore.followPlayhead} on the idle/paused →
   * playing transition when the off-state was set during the previous
   * playback session (pan, minimap drag, or follow-button toggle while
   * playing). No-op when {@link PlaybackStore.autoFollowOnPlay} is off,
   * when follow is already on, or when the user deliberately disabled it
   * while idle/paused.
   */
  private maybeReenableFollowOnPlay() {
    if (!this.playback.autoFollowOnPlay) return;
    if (this.playback.followPlayhead) return;
    if (!this.playback.followDisabledIsTransient) return;
    this.setFollowPlayhead(true);
  }
}
