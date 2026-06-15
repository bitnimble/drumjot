import type { AudioTrackId } from 'src/editing/playback/audio_tracks';

/**
 * Callback + state contract the mixer's instrument (drum-lane) rows are
 * driven by. Supplied by the view layer (wired to MixerPresenter); the
 * row components and `MixerView` only see this prop interface, never the
 * stores directly.
 */
export type LayerControls = {
  mutedLanes: ReadonlySet<string>;
  soloedLanes: ReadonlySet<string>;
  /** True if the row would currently make sound; false = muted via M or solo exclusion. */
  isLaneAudible: (lane: string) => boolean;
  /** Current row fader value, 0..1 (1 = full). */
  volumeFor: (lane: string) => number;
  onSetVolume: (lane: string, v: number) => void;
  onToggleMute: (lane: string) => void;
  onToggleSolo: (lane: string) => void;
  /** Drum section master M/S. The master acts at the bus, not by editing
   * the per-row M/S sets; `masterAudible` reflects the resolved state
   * (master mute + cross-domain solo) so the master row can dim itself. */
  masterMuted: boolean;
  masterSoloed: boolean;
  masterAudible: boolean;
  onToggleMasterMute: () => void;
  onToggleMasterSolo: () => void;
};

/** Callback + state contract for the mixer's audio-track rows. */
export type AudioTrackControls = {
  mutedAudioTracks: ReadonlySet<AudioTrackId>;
  soloedAudioTracks: ReadonlySet<AudioTrackId>;
  isAudioTrackAudible: (id: AudioTrackId) => boolean;
  volumeFor: (id: AudioTrackId) => number;
  onSetVolume: (id: AudioTrackId, v: number) => void;
  onToggleMute: (id: AudioTrackId) => void;
  onToggleSolo: (id: AudioTrackId) => void;
  /** Drop a loaded audio track (exposed in the row's overflow menu). */
  onClear: (id: AudioTrackId) => void;
  /** Overflow menu: run stage 1 (`stems_all`) on this track,
   *  isolating drums + drumless backing from a full-mix recording. */
  onSplitFromMix: (id: AudioTrackId) => void;
  /** Overflow menu: run stage 2 (`stems_per`) on this track,
   *  splitting a drum-only recording into per-instrument pieces. */
  onSplitDrumPieces: (id: AudioTrackId) => void;
  /** Audio section master M/S; same semantics as on {@link LayerControls}. */
  masterMuted: boolean;
  masterSoloed: boolean;
  masterAudible: boolean;
  onToggleMasterMute: () => void;
  onToggleMasterSolo: () => void;
};
