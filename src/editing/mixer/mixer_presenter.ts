import { comparer, makeAutoObservable, reaction } from 'mobx';
import { AudioTrackId } from 'src/editing/playback/audio_tracks';
import { jotPlayer } from 'src/editing/playback/player';
import { JotEditorStore } from '../jot_editor_store';
import { MixerStore, clampVolume } from './mixer_store';
import type { Resettable } from '../session_reset';
import { toastStore } from '../../ui/toasts/toasts';

/**
 * Mutations over {@link MixerStore}, per-row + master mute/solo/volume,
 * and the audio-track split stubs. Owns the instrument-track-view-model
 * prune reaction and wires the store in as the player's
 * {@link MixerContext} for audio-track colour inheritance. Row order now
 * lives in the doc (`jot.ordering`, written by `LayersPresenter`); this
 * presenter no longer owns it. The engine's mute/solo/volume filter is
 * not pushed from here; the player pulls it off the PlaybackStore
 * computeds (see `PlaybackPresenter`).
 */
export class MixerPresenter implements Resettable {
  readonly mixer: MixerStore;
  readonly jotEditorStore: JotEditorStore;

  constructor(mixer: MixerStore, jotEditorStore: JotEditorStore) {
    this.mixer = mixer;
    this.jotEditorStore = jotEditorStore;
    makeAutoObservable(this, { mixer: false, jotEditorStore: false });

    // Wire the mixer store in as the player's mixer context so freshly-
    // constructed AudioTracks can resolve grouped-instrument colour
    // inheritance. Done before any reactions fire so loadAudioTrack calls
    // made during the same tick see a populated context.
    jotPlayer.attachMixerContext(this.mixer);

    // Prune instrument-track view-models for lanes no longer present in
    // the active jot. The override is store-owned and survives jot
    // reloads, so a lane that comes back in a later jot picks up its
    // previous override; we only forget lanes that disappeared from the
    // current jot to keep the map from growing unboundedly across a long
    // session. `fireImmediately` runs the prune once on boot.
    reaction(
      () => new Set(this.mixer.jotLanes),
      (lanes) => {
        for (const p of Array.from(this.mixer.instrumentTracks.keys())) {
          if (!lanes.has(p)) this.mixer.instrumentTracks.delete(p);
        }
      },
      { fireImmediately: true, equals: comparer.structural }
    );

    // NOTE: the engine no longer has mixer state pushed into it here. The
    // player PULLS its mute/solo/volume filter + section-audibility off the
    // PlaybackStore computeds (which delegate to this store's `laneFilter`
    // / `audioTrackFilter` / `isAudio|DrumSectionAudible`); the reactions
    // that fire the imperative audio-graph re-apply live in
    // `PlaybackPresenter`.
  }

  toggleAudioMasterMute() {
    this.mixer.audioMasterMuted = !this.mixer.audioMasterMuted;
  }

  toggleDrumMasterMute() {
    this.mixer.drumMasterMuted = !this.mixer.drumMasterMuted;
  }

  /** Enabling solo clears the matching master-mute so the section can
   * actually be heard; mirrors `toggleSolo` for per-row state. */
  toggleAudioMasterSolo() {
    if (this.mixer.audioMasterSoloed) {
      this.mixer.audioMasterSoloed = false;
    } else {
      this.mixer.audioMasterSoloed = true;
      this.mixer.audioMasterMuted = false;
    }
  }

  toggleDrumMasterSolo() {
    if (this.mixer.drumMasterSoloed) {
      this.mixer.drumMasterSoloed = false;
    } else {
      this.mixer.drumMasterSoloed = true;
      this.mixer.drumMasterMuted = false;
    }
  }

  toggleMute(track: string) {
    if (this.mixer.mutedTracks.has(track)) this.mixer.mutedTracks.delete(track);
    else this.mixer.mutedTracks.add(track);
  }

  toggleSolo(track: string) {
    if (this.mixer.soloedTracks.has(track)) {
      this.mixer.soloedTracks.delete(track);
    } else {
      this.mixer.soloedTracks.add(track);
      this.mixer.mutedTracks.delete(track);
    }
  }

  setTrackVolume(track: string, v: number) {
    this.mixer.trackVolumes.set(track, clampVolume(v));
  }

  toggleAudioTrackMute(id: AudioTrackId) {
    if (this.mixer.mutedAudioTracks.has(id)) this.mixer.mutedAudioTracks.delete(id);
    else this.mixer.mutedAudioTracks.add(id);
  }

  toggleAudioTrackSolo(id: AudioTrackId) {
    if (this.mixer.soloedAudioTracks.has(id)) {
      this.mixer.soloedAudioTracks.delete(id);
    } else {
      this.mixer.soloedAudioTracks.add(id);
      this.mixer.mutedAudioTracks.delete(id);
    }
  }

  setAudioTrackVolume(id: AudioTrackId, v: number) {
    this.mixer.audioTrackVolumes.set(id, clampVolume(v));
  }

  /** Mute a batch of audio tracks. Used by the song loaders
   * (JotEditorPresenter) to default per-lane stems / drum tracks to
   * muted so the audible drums come from the score scheduler. */
  muteAudioTracks(ids: readonly AudioTrackId[]): void {
    for (const id of ids) this.mixer.mutedAudioTracks.add(id);
  }

  /** Drop a removed audio track's mute/solo/volume so it doesn't linger
   * (ids are never reused), and so clearing the only soloed track doesn't
   * leave a phantom solo silencing everything else. The colour override
   * lives on the AudioTrack instance and is freed by the player. */
  clearAudioTrack(id: AudioTrackId): void {
    jotPlayer.clearAudioTrack(id);
    this.mixer.mutedAudioTracks.delete(id);
    this.mixer.soloedAudioTracks.delete(id);
    this.mixer.audioTrackVolumes.delete(id);
  }

  /** Mark an audio track as currently being split. Drives the per-row
   *  spinner; safe to call multiple times (latest call wins). */
  beginAudioTrackSplit(id: AudioTrackId, kind: 'mix' | 'pieces'): void {
    this.mixer.audioTrackSplitStatuses.set(id, { phase: 'splitting', kind });
  }

  /** Clear the splitting status once the work has finished. */
  endAudioTrackSplit(id: AudioTrackId): void {
    this.mixer.audioTrackSplitStatuses.delete(id);
  }

  /** Stub: "Split into drums + backing" from the audio-track overflow
   * menu. The transcriber-side wiring is deferred; surface a status pill
   * so the click visibly does something in the meantime. */
  splitAudioTrackFromMix(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drums + backing on "${name}" isn't wired up yet.`);
  }

  /** Stub: "Split into kick / snare / hi-hat / cymbals". See
   * {@link splitAudioTrackFromMix}. */
  splitAudioTrackDrumPieces(id: AudioTrackId): void {
    const track = jotPlayer.audioTracks.get(id);
    const name = track ? track.filename : id;
    toastStore.showError(`Split into drum pieces on "${name}" isn't wired up yet.`);
  }

  /** Drop every loaded audio track. Used when a new source replaces the
   * current song so the previous song's tracks don't linger. */
  clearAllAudioTracks(): void {
    for (const id of Array.from(jotPlayer.audioTracks.keys())) {
      this.clearAudioTrack(id);
    }
  }

  /** Reset the per-track mixer (mute/solo/volume). Keyed by track key
   * (`layerId/lane`), not by song, so without this a setting on one song
   * bleeds onto the next song's matching rows. */
  resetTrackMixer(): void {
    this.mixer.mutedTracks.clear();
    this.mixer.soloedTracks.clear();
    this.mixer.trackVolumes.clear();
  }

  /**
   * Session reset: drop every audio track, clear all per-row + master
   * mute/solo/volume, and forget the per-instrument colour view-models +
   * split stubs, so no mixer state from the previous song leaks onto the
   * new one (the page-refresh semantic). A loaded save file's drum-lane
   * mixer is re-applied afterwards via {@link applyTrackMixerState};
   * audio-track mixer state is session-only (the audio files themselves
   * aren't in the `.jot`, and their ids aren't stable across a reload).
   */
  reset(): void {
    this.clearAllAudioTracks();
    this.resetTrackMixer();
    this.mixer.mutedAudioTracks.clear();
    this.mixer.soloedAudioTracks.clear();
    this.mixer.audioTrackVolumes.clear();
    this.mixer.audioMasterMuted = false;
    this.mixer.drumMasterMuted = false;
    this.mixer.audioMasterSoloed = false;
    this.mixer.drumMasterSoloed = false;
    this.mixer.instrumentTracks.clear();
    this.mixer.audioTrackSplitStatuses.clear();
  }

  /**
   * Re-apply a saved drum-lane mixer snapshot from a loaded `.jot` file's
   * editor metadata. Only the drum domain round-trips: its keys
   * (`layerId/lane`) are part of the persisted document, so they're stable
   * across save/load, unlike session-generated audio-track ids. Called
   * after {@link reset} + the new song is installed.
   */
  applyTrackMixerState(state: TrackMixerState): void {
    this.mixer.mutedTracks = new Set(state.mutedTracks);
    this.mixer.soloedTracks = new Set(state.soloedTracks);
    this.mixer.trackVolumes = new Map(state.trackVolumes);
    this.mixer.drumMasterMuted = state.drumMasterMuted;
    this.mixer.drumMasterSoloed = state.drumMasterSoloed;
  }

  /** Re-apply a saved audio track's per-track mixer state to its freshly-
   *  minted id when restoring a `.jot` save bundle's embedded audio. Unlike
   *  drum tracks, audio-track ids are session-generated, so this is keyed by
   *  the new id the re-decode produced, not a stable key. */
  applyAudioTrackState(
    id: AudioTrackId,
    state: { muted: boolean; soloed: boolean; volume: number }
  ): void {
    if (state.muted) this.mixer.mutedAudioTracks.add(id);
    if (state.soloed) this.mixer.soloedAudioTracks.add(id);
    this.mixer.audioTrackVolumes.set(id, clampVolume(state.volume));
  }

  /** Snapshot the drum-lane mixer for persistence (the inverse of
   *  {@link applyTrackMixerState}). */
  trackMixerState(): TrackMixerState {
    return {
      mutedTracks: Array.from(this.mixer.mutedTracks),
      soloedTracks: Array.from(this.mixer.soloedTracks),
      trackVolumes: Array.from(this.mixer.trackVolumes.entries()),
      drumMasterMuted: this.mixer.drumMasterMuted,
      drumMasterSoloed: this.mixer.drumMasterSoloed,
    };
  }
}

/**
 * Serialisable drum-lane mixer state for the `.jot` save format's editor
 * metadata. Drum rows only (keys are `layerId/lane`, stable across a
 * reload); audio-track mute/solo/volume is session-only and not persisted.
 */
export type TrackMixerState = {
  mutedTracks: string[];
  soloedTracks: string[];
  trackVolumes: Array<[string, number]>;
  drumMasterMuted: boolean;
  drumMasterSoloed: boolean;
};
