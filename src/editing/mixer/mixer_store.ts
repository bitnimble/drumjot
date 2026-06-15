import { makeAutoObservable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { Instrument } from 'src/schema/dsl/dsl';
import { DrumInstrumentKind, defaultKindForLane } from 'src/instruments/instruments';
import type { StructuralPresenter } from 'src/editing/structure/structural_presenter';
import { AudioTrackFilter, AudioTrackId, isAudioTrackAudibleUnder } from 'src/editing/playback/audio_tracks';
import { isAudibleUnder, PlayerFilter } from 'src/editing/playback/player';
import {
  INSTRUMENT_FALLBACK_COLOR,
  InstrumentTrack,
  MixerContext,
  TrackKey,
} from 'src/editing/tracks/tracks';
import { JotEditorStore } from '../jot_editor_store';

// Row volume faders are pure attenuation (0 = silent, 1 = unscaled).
// The kit's overall loudness is handled by the drum master gain.
export const VOLUME_STEP = 0.05;

export function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.max(0, Math.min(1, v));
}

/**
 * Default top-to-bottom mixer ordering for drum-instrument kinds when
 * the user hasn't manually reordered rows: top-of-kit cymbals first,
 * then drums from high to low, with kick last. `custom` falls to the very
 * bottom. Drives {@link collectJotLanes}.
 */
const DEFAULT_MIXER_KIND_ORDER: readonly DrumInstrumentKind[] = [
  'crash',
  'ride',
  'hihat',
  'tom',
  'snare',
  'kick',
  'custom',
];

/**
 * Best-effort `DrumInstrumentKind` from an instrument's display name.
 * Used to recover a sensible mixer position for rows whose loader stamped
 * `kind: 'custom'` despite a recognisable name. Substring-based; the
 * patterns mirror the names produced by the RLRR / MIDI / transcriber
 * loaders.
 */
function inferKindFromInstrumentName(name: string | undefined): DrumInstrumentKind | undefined {
  if (!name) return undefined;
  const n = name.toLowerCase();
  if (/\bkick\b|\bbass\s*drum\b/.test(n)) return 'kick';
  if (/\bsnare\b/.test(n)) return 'snare';
  if (/hi.?hat/.test(n)) return 'hihat';
  if (/\bride\b/.test(n)) return 'ride';
  if (/\bcrash\b|\bchina\b|\bsplash\b/.test(n)) return 'crash';
  if (/\bfloor\s*tom\b|\btom\b/.test(n)) return 'tom';
  return undefined;
}

/**
 * Floor toms render below regular toms within the tom group. Detected
 * from the instrument name; lane letter `f` is the GM importer's
 * convention for the floor tom so it counts even with no display name.
 */
function isFloorTom(instrument: Instrument | undefined, lane: string): boolean {
  if (instrument?.name && /floor/i.test(instrument.name)) return true;
  return lane === 'f';
}

/**
 * Sort tuple for the default mixer order: [kind rank, intra-kind rank,
 * lane]. Kind comes from the parsed `Instrument` when available;
 * `kind: 'custom'` falls back to a name heuristic, then to the lane
 * letter's default kind. Intra-kind rank only matters for toms today
 * (regular before floor).
 */
function defaultMixerSortKey(
  lane: string,
  instrument: Instrument | undefined
): [number, number, string] {
  let kind: DrumInstrumentKind = instrument?.kind ?? 'custom';
  if (kind === 'custom') {
    const fromName = inferKindFromInstrumentName(instrument?.name);
    if (fromName) kind = fromName;
  }
  if (kind === 'custom') {
    const fromLetter = defaultKindForLane(lane);
    if (fromLetter !== 'custom') kind = fromLetter;
  }
  const kindRank = DEFAULT_MIXER_KIND_ORDER.indexOf(kind);
  const subRank = kind === 'tom' && isFloorTom(instrument, lane) ? 1 : 0;
  return [kindRank === -1 ? DEFAULT_MIXER_KIND_ORDER.length : kindRank, subRank, lane];
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
  const mapping = structural.source.globalMetadata.instrumentMapping;
  const instrumentFor = (lane: string): Instrument => mapping?.[lane] ?? { kind: 'custom' };
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
  /** DSL lanes the user has muted via the row-gutter M button. */
  mutedLanes: Set<string> = new Set();
  /** DSL lanes the user has soloed. When non-empty, ONLY these rows are
   *  audible (cross-domain with audio tracks via {@link soloActive}). */
  soloedLanes: Set<string> = new Set();
  /** Audio-track ids the user has muted via the gutter M button. */
  mutedAudioTracks: Set<AudioTrackId> = new Set();
  /** Soloed audio-track ids; same semantics as {@link soloedLanes}. */
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
   * Per-row volume faders, 0..1 (1 = full). Sparse: a row absent from the
   * map plays at full volume. Lane volumes scale note velocity; audio
   * volumes scale the track's GainNode.
   */
  laneVolumes: Map<string, number> = new Map();
  audioTrackVolumes: Map<AudioTrackId, number> = new Map();
  /**
   * Per-lane {@link InstrumentTrack} view-models keyed by DSL lane
   * letter (per-instrument note-colour override). Survives jot reloads;
   * the presenter prunes entries for lanes no longer present.
   */
  instrumentTracks: Map<string, InstrumentTrack> = new Map();
  /**
   * User-customizable order of mixer rows. Each entry is a loaded audio
   * track id, a DSL lane letter, or a lyrics track; the mixer renders
   * rows in this exact order. Kept in sync with the live track/lane set
   * by the presenter.
   */
  trackOrder: TrackKey[] = [];
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
      this.soloedLanes.size > 0 ||
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
    return this.drumMasterSoloed || this.soloedLanes.size > 0;
  }

  /**
   * Live {@link PlayerFilter} view onto the per-lane mute/solo/volume
   * state. Sets and Maps are *snapshotted* on each read so the downstream
   * `reaction(..., comparer.structural)` that pushes this to the player
   * can actually detect changes; sharing the live references would
   * defeat the comparer (prev and next would point at the same mutated
   * instance).
   */
  get laneFilter(): PlayerFilter {
    return {
      mutedLanes: new Set(this.mutedLanes),
      soloedLanes: new Set(this.soloedLanes),
      soloActive: this.soloActive,
      sectionMasterMuted: this.drumMasterMuted,
      sectionMasterSoloed: this.drumMasterSoloed,
      volumes: new Map(this.laneVolumes),
    };
  }

  /** Mirror of {@link laneFilter} for the audio-track domain. */
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
   * Drum-lane lane order derived from {@link trackOrder}, dropping audio
   * + lyrics rows. Pattern brackets use this to know whether a row is the
   * topmost / bottommost participant of a pattern span.
   */
  get laneOrder(): readonly string[] {
    return this.trackOrder.flatMap((k) => (k.kind === 'instrument' ? [k.lane] : []));
  }

  /**
   * Index of the topmost instrument row in {@link trackOrder}. The mixer
   * hosts score-wide chrome (tuplet brackets, lead-in label) on that row.
   * `-1` when no instrument row exists yet.
   */
  get firstInstrumentIdx(): number {
    return this.trackOrder.findIndex((k) => k.kind === 'instrument');
  }

  /**
   * Whether a drum lane is audible under the live mute/solo/volume state.
   * `computedFn` memoises per-argument, so a per-row gutter observer only
   * re-renders when its own lane's audibility flips.
   */
  isLaneAudible = computedFn((lane: string): boolean => {
    return isAudibleUnder(lane, this.laneFilter);
  });

  /** Mirror of {@link isLaneAudible} for the audio-track domain. */
  isAudioTrackAudible = computedFn((id: AudioTrackId): boolean => {
    return isAudioTrackAudibleUnder(id, this.audioTrackFilter);
  });

  laneVolume(lane: string): number {
    return this.laneVolumes.get(lane) ?? 1;
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
