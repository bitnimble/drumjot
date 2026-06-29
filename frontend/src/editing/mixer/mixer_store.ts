import { makeAutoObservable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { Instrument } from 'src/schema/dsl/dsl';
import { defaultMixerSortKey } from 'src/instruments/mixer_order';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { AudioTrackFilter, AudioTrackId, isAudioTrackAudibleUnder } from 'src/editing/playback/audio_tracks';
import { isAudibleUnder, PlayerFilter } from 'src/editing/playback/player';
import {
  INSTRUMENT_FALLBACK_COLOR,
  InstrumentTrack,
  MixerContext,
} from 'src/editing/tracks/tracks';
import { audioTrackEntityId, groupSiblingInstrumentLanes } from 'src/schema/ordering';
import { JotEditorStore } from '../jot_editor_store';

// Row volume faders are pure attenuation (0 = silent, 1 = unscaled).
// The kit's overall loudness is handled by the drum master gain.
export const VOLUME_STEP = 0.05;

export function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.max(0, Math.min(1, v));
}

/**
 * Lanes that appear anywhere in the rendered jot, sorted into the
 * default mixer ordering. A lane that shows up in two layers is listed
 * once at its first appearance; ordering reads each lane's resolved
 * `Instrument` (from the first bar that has a track for it).
 *
 * Reads the zoom-invariant structural layers (not pixels) so the
 * mixer-order reaction that wraps this doesn't re-evaluate on every wheel
 * tick; lane identity is a function of the source DSL, not the layout.
 */
export function collectJotLanes(structural: StructuralPresenter | undefined): string[] {
  if (!structural) return [];
  // `structural.lanes` is the `computed.struct` lane set (lanes that carry a
  // note), so this, and the mixer-order reaction wrapping it, is stable across
  // an in-lane note edit; only a lane appearing/disappearing perturbs it.
  const instrumentFor = (lane: string): Instrument => structural.instrumentFor(lane);
  const out = [...structural.lanes];
  out.sort((a, b) => {
    const ka = defaultMixerSortKey(a, instrumentFor(a));
    const kb = defaultMixerSortKey(b, instrumentFor(b));
    if (ka[0] !== kb[0]) return ka[0] - kb[0];
    if (ka[1] !== kb[1]) return ka[1] - kb[1];
    return ka[2].localeCompare(kb[2]);
  });
  return out;
}

/** Per-row stem-split status; read by the audio-track row to render a
 *  loading spinner alongside the label while a split is in flight. */
export type AudioTrackSplitStatus = { phase: 'splitting'; kind: 'mix' | 'pieces' };

/**
 * Mixer state: per-row mute/solo/volume (drum lanes + audio tracks),
 * section masters, per-instrument colour view-models, and the
 * user-customisable row order. Implements {@link MixerContext} so the
 * player's audio-track colour inheritance can resolve through it.
 *
 * Pure data: observables + derived computeds + read accessors. The only
 * mutation here is {@link getInstrumentTrack}'s lazy memoisation cache;
 * every real mutation (toggles, volumes, reorder, reset) lives on the
 * presenter.
 */
export class MixerStore implements MixerContext {
  /** Track keys (`layerId/lane`, see {@link trackKey}) the user has muted via
   *  the row-gutter M button. */
  mutedTracks: Set<string> = new Set();
  /** Track keys the user has soloed. When non-empty, ONLY these rows are
   *  audible (cross-domain with audio tracks via {@link soloActive}). */
  soloedTracks: Set<string> = new Set();
  /** Audio-track ids the user has muted via the gutter M button. */
  mutedAudioTracks: Set<AudioTrackId> = new Set();
  /** Soloed audio-track ids; same semantics as {@link soloedTracks}. */
  soloedAudioTracks: Set<AudioTrackId> = new Set();
  /**
   * Section-master mute / solo. These act on the whole bus, not by
   * editing the per-row M/S sets. Master-solo is folded into
   * {@link soloActive} so it participates in the same cross-domain "if
   * anything is soloed, non-soloed rows fall silent" rule.
   */
  audioMasterMuted: boolean = false;
  drumMasterMuted: boolean = false;
  audioMasterSoloed: boolean = false;
  drumMasterSoloed: boolean = false;
  /**
   * Per-row volume faders, 0..1 (1 = full), keyed by track key
   * (`layerId/lane`). Sparse: a row absent from the map plays at full volume.
   * Track volumes scale note velocity; audio volumes scale the track's GainNode.
   */
  trackVolumes: Map<string, number> = new Map();
  audioTrackVolumes: Map<AudioTrackId, number> = new Map();
  /**
   * Per-lane {@link InstrumentTrack} view-models keyed by DSL lane
   * letter (per-instrument note-colour override). Survives jot reloads;
   * the presenter prunes entries for lanes no longer present.
   */
  instrumentTracks: Map<string, InstrumentTrack> = new Map();
  /**
   * Per-track stem-split status; sparse map (a row absent renders without
   * a spinner). The presenter brackets split work with begin/end.
   */
  audioTrackSplitStatuses: Map<AudioTrackId, AudioTrackSplitStatus> = new Map();

  /** Active jot, for the mixer-ordered lane list. */
  readonly jotEditorStore: JotEditorStore;

  constructor(jotEditorStore: JotEditorStore) {
    this.jotEditorStore = jotEditorStore;
    makeAutoObservable(this, { jotEditorStore: false });
  }

  /**
   * Solo is one global mode across both the lane and audio-track domains:
   * any soloed row (drum OR music) puts every non-soloed row into the
   * "solo-excluded" state. The two section-master solos count too.
   */
  get soloActive(): boolean {
    return (
      this.soloedTracks.size > 0 ||
      this.soloedAudioTracks.size > 0 ||
      this.audioMasterSoloed ||
      this.drumMasterSoloed
    );
  }

  /**
   * Whether the audio section's bus is currently audible. Master mute
   * always wins; under an active solo the section is audible only if it
   * is master-soloed OR has at least one soloed row.
   */
  get isAudioSectionAudible(): boolean {
    if (this.audioMasterMuted) return false;
    if (!this.soloActive) return true;
    return this.audioMasterSoloed || this.soloedAudioTracks.size > 0;
  }

  /** Mirror of {@link isAudioSectionAudible} for the drum section. */
  get isDrumSectionAudible(): boolean {
    if (this.drumMasterMuted) return false;
    if (!this.soloActive) return true;
    return this.drumMasterSoloed || this.soloedTracks.size > 0;
  }

  /**
   * Live {@link PlayerFilter} view onto the per-lane mute/solo/volume
   * state. Sets and Maps are *snapshotted* on each read so the downstream
   * `reaction(..., comparer.structural)` that pushes this to the player
   * can actually detect changes; sharing the live references would
   * defeat the comparer (prev and next would point at the same mutated
   * instance).
   */
  get trackFilter(): PlayerFilter {
    return {
      mutedTracks: new Set(this.mutedTracks),
      soloedTracks: new Set(this.soloedTracks),
      soloActive: this.soloActive,
      sectionMasterMuted: this.drumMasterMuted,
      sectionMasterSoloed: this.drumMasterSoloed,
      volumes: new Map(this.trackVolumes),
    };
  }

  /** Mirror of {@link trackFilter} for the audio-track domain. */
  get audioTrackFilter(): AudioTrackFilter {
    return {
      mutedAudioTracks: new Set(this.mutedAudioTracks),
      soloedAudioTracks: new Set(this.soloedAudioTracks),
      soloActive: this.soloActive,
      sectionMasterMuted: this.audioMasterMuted,
      sectionMasterSoloed: this.audioMasterSoloed,
      volumes: new Map(this.audioTrackVolumes),
    };
  }

  /**
   * Lanes that appear anywhere in the rendered jot, in the default mixer
   * ordering. Thin wrapper over {@link collectJotLanes} so consumers
   * track a single MobX-memoised computed rather than re-walking the jot.
   */
  get jotLanes(): readonly string[] {
    return collectJotLanes(this.jotEditorStore.structural);
  }

  /**
   * Instrument lanes sharing the audio track's group in `jot.ordering`,
   * in slot order; empty when the audio row sits in a loose run. Serves
   * the {@link MixerContext} the player calls back into for audio-track
   * colour inheritance. Reads the doc ordering, so the audio colour
   * computeds that call it react to regroups.
   */
  groupInstrumentLanesForAudio(audioId: AudioTrackId): string[] {
    const jot = this.jotEditorStore.jot;
    if (!jot) return [];
    return groupSiblingInstrumentLanes(jot, audioTrackEntityId(audioId));
  }

  /**
   * Whether a drum track (`layerId/lane` key) is audible under the live
   * mute/solo/volume state. `computedFn` memoises per-argument, so a per-row
   * gutter observer only re-renders when its own track's audibility flips.
   */
  isTrackAudible = computedFn((track: string): boolean => {
    return isAudibleUnder(track, this.trackFilter);
  });

  /** Mirror of {@link isTrackAudible} for the audio-track domain. */
  isAudioTrackAudible = computedFn((id: AudioTrackId): boolean => {
    return isAudioTrackAudibleUnder(id, this.audioTrackFilter);
  });

  trackVolume(track: string): number {
    return this.trackVolumes.get(track) ?? 1;
  }

  audioTrackVolume(id: AudioTrackId): number {
    return this.audioTrackVolumes.get(id) ?? 1;
  }

  /**
   * Lazily-constructed {@link InstrumentTrack} for a DSL lane. The
   * track's fallback closure reads the active jot's palette default, so a
   * jot reload that re-shuffles palette slots updates unfilled tracks
   * automatically. Memoised in {@link instrumentTracks} so every callsite
   * reads/writes the same observable; the presenter prunes dead entries.
   * Also serves the {@link MixerContext} the player calls back into.
   */
  getInstrumentTrack(lane: string): InstrumentTrack {
    let track = this.instrumentTracks.get(lane);
    if (track) return track;
    track = new InstrumentTrack(
      lane,
      () => this.jotEditorStore.palette?.paletteColorFor(lane) ?? INSTRUMENT_FALLBACK_COLOR
    );
    this.instrumentTracks.set(lane, track);
    return track;
  }
}
