import { makeAutoObservable } from 'mobx';
import { computedFn } from 'mobx-utils';
import { Instrument } from 'src/dsl';
import { DrumInstrumentKind, defaultKindForPitch } from 'src/instruments';
import { RenderedJot } from 'src/jot';
import {
  AudioTrackFilter,
  AudioTrackId,
  isAudibleUnder,
  isAudioTrackAudibleUnder,
  PlayerFilter,
} from 'src/playback';
import {
  INSTRUMENT_FALLBACK_COLOR,
  InstrumentTrack,
  MixerContext,
  TrackKey,
} from 'src/tracks';
import { DocumentStore } from './document_store';

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
 * bottom. Drives {@link collectJotPitches}.
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
 * from the instrument name; pitch letter `f` is the GM importer's
 * convention for the floor tom so it counts even with no display name.
 */
function isFloorTom(instrument: Instrument | undefined, pitch: string): boolean {
  if (instrument?.name && /floor/i.test(instrument.name)) return true;
  return pitch === 'f';
}

/**
 * Sort tuple for the default mixer order: [kind rank, intra-kind rank,
 * pitch]. Kind comes from the parsed `Instrument` when available;
 * `kind: 'custom'` falls back to a name heuristic, then to the pitch
 * letter's default kind. Intra-kind rank only matters for toms today
 * (regular before floor).
 */
function defaultMixerSortKey(
  pitch: string,
  instrument: Instrument | undefined
): [number, number, string] {
  let kind: DrumInstrumentKind = instrument?.kind ?? 'custom';
  if (kind === 'custom') {
    const fromName = inferKindFromInstrumentName(instrument?.name);
    if (fromName) kind = fromName;
  }
  if (kind === 'custom') {
    const fromLetter = defaultKindForPitch(pitch);
    if (fromLetter !== 'custom') kind = fromLetter;
  }
  const kindRank = DEFAULT_MIXER_KIND_ORDER.indexOf(kind);
  const subRank = kind === 'tom' && isFloorTom(instrument, pitch) ? 1 : 0;
  return [kindRank === -1 ? DEFAULT_MIXER_KIND_ORDER.length : kindRank, subRank, pitch];
}

/**
 * Pitches that appear anywhere in the rendered jot, sorted into the
 * default mixer ordering. A pitch that shows up in two voices is listed
 * once at its first appearance; ordering reads each pitch's resolved
 * `Instrument` (from the first bar that has a track for it).
 *
 * Reads the zoom-invariant structural cache (not `jot.resolved`) so the
 * mixer-order reaction that wraps this doesn't re-evaluate on every wheel
 * tick; pitch identity is a function of the source DSL, not the layout.
 */
export function collectJotPitches(jot: RenderedJot | undefined): string[] {
  if (!jot) return [];
  const out: string[] = [];
  const instrumentByPitch = new Map<string, Instrument>();
  for (const voice of jot.structure.voices) {
    for (const p of voice.pitches) {
      if (!out.includes(p)) out.push(p);
    }
    for (const bar of voice.bars) {
      for (const [pitch, track] of Object.entries(bar.tracks)) {
        if (!instrumentByPitch.has(pitch)) {
          instrumentByPitch.set(pitch, track.instrument);
        }
      }
    }
  }
  out.sort((a, b) => {
    const ka = defaultMixerSortKey(a, instrumentByPitch.get(a));
    const kb = defaultMixerSortKey(b, instrumentByPitch.get(b));
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
 * Mixer state: per-row mute/solo/volume (drum pitches + audio tracks),
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
  /** DSL pitches the user has muted via the row-gutter M button. */
  mutedPitches: Set<string> = new Set();
  /** DSL pitches the user has soloed. When non-empty, ONLY these rows are
   *  audible (cross-domain with audio tracks via {@link soloActive}). */
  soloedPitches: Set<string> = new Set();
  /** Audio-track ids the user has muted via the gutter M button. */
  mutedAudioTracks: Set<AudioTrackId> = new Set();
  /** Soloed audio-track ids; same semantics as {@link soloedPitches}. */
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
   * map plays at full volume. Pitch volumes scale note velocity; audio
   * volumes scale the track's GainNode.
   */
  pitchVolumes: Map<string, number> = new Map();
  audioTrackVolumes: Map<AudioTrackId, number> = new Map();
  /**
   * Per-pitch {@link InstrumentTrack} view-models keyed by DSL pitch
   * letter (per-instrument note-colour override). Survives jot reloads;
   * the presenter prunes entries for pitches no longer present.
   */
  instrumentTracks: Map<string, InstrumentTrack> = new Map();
  /**
   * User-customizable order of mixer rows. Each entry is a loaded audio
   * track id, a DSL pitch letter, or a lyrics track; the mixer renders
   * rows in this exact order. Kept in sync with the live track/pitch set
   * by the presenter.
   */
  trackOrder: TrackKey[] = [];
  /**
   * Per-track stem-split status; sparse map (a row absent renders without
   * a spinner). The presenter brackets split work with begin/end.
   */
  audioTrackSplitStatuses: Map<AudioTrackId, AudioTrackSplitStatus> = new Map();

  /** Active jot, for the mixer-ordered pitch list. */
  readonly document: DocumentStore;

  constructor(document: DocumentStore) {
    this.document = document;
    makeAutoObservable(this, { document: false });
  }

  /**
   * Solo is one global mode across both the pitch and audio-track domains:
   * any soloed row (drum OR music) puts every non-soloed row into the
   * "solo-excluded" state. The two section-master solos count too.
   */
  get soloActive(): boolean {
    return (
      this.soloedPitches.size > 0 ||
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
    return this.drumMasterSoloed || this.soloedPitches.size > 0;
  }

  /**
   * Live {@link PlayerFilter} view onto the per-pitch mute/solo/volume
   * state. Sets and Maps are *snapshotted* on each read so the downstream
   * `reaction(..., comparer.structural)` that pushes this to the player
   * can actually detect changes; sharing the live references would
   * defeat the comparer (prev and next would point at the same mutated
   * instance).
   */
  get pitchFilter(): PlayerFilter {
    return {
      mutedPitches: new Set(this.mutedPitches),
      soloedPitches: new Set(this.soloedPitches),
      soloActive: this.soloActive,
      sectionMasterMuted: this.drumMasterMuted,
      sectionMasterSoloed: this.drumMasterSoloed,
      volumes: new Map(this.pitchVolumes),
    };
  }

  /** Mirror of {@link pitchFilter} for the audio-track domain. */
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
   * Pitches that appear anywhere in the rendered jot, in the default mixer
   * ordering. Thin wrapper over {@link collectJotPitches} so consumers
   * track a single MobX-memoised computed rather than re-walking the jot.
   */
  get jotPitches(): readonly string[] {
    return collectJotPitches(this.document.currentJot);
  }

  /**
   * Drum-pitch lane order derived from {@link trackOrder}, dropping audio
   * + lyrics rows. Pattern brackets use this to know whether a row is the
   * topmost / bottommost participant of a pattern span.
   */
  get pitchOrder(): readonly string[] {
    return this.trackOrder.flatMap((k) => (k.kind === 'instrument' ? [k.pitch] : []));
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
   * Whether a drum pitch is audible under the live mute/solo/volume state.
   * `computedFn` memoises per-argument, so a per-row gutter observer only
   * re-renders when its own pitch's audibility flips.
   */
  isPitchAudible = computedFn((pitch: string): boolean => {
    return isAudibleUnder(pitch, this.pitchFilter);
  });

  /** Mirror of {@link isPitchAudible} for the audio-track domain. */
  isAudioTrackAudible = computedFn((id: AudioTrackId): boolean => {
    return isAudioTrackAudibleUnder(id, this.audioTrackFilter);
  });

  pitchVolume(pitch: string): number {
    return this.pitchVolumes.get(pitch) ?? 1;
  }

  audioTrackVolume(id: AudioTrackId): number {
    return this.audioTrackVolumes.get(id) ?? 1;
  }

  /**
   * Lazily-constructed {@link InstrumentTrack} for a DSL pitch. The
   * track's fallback closure reads the active jot's palette default, so a
   * jot reload that re-shuffles palette slots updates unfilled tracks
   * automatically. Memoised in {@link instrumentTracks} so every callsite
   * reads/writes the same observable; the presenter prunes dead entries.
   * Also serves the {@link MixerContext} the player calls back into.
   */
  getInstrumentTrack(pitch: string): InstrumentTrack {
    let track = this.instrumentTracks.get(pitch);
    if (track) return track;
    track = new InstrumentTrack(
      pitch,
      () => this.document.currentJot?.defaultPaletteColorFor(pitch) ?? INSTRUMENT_FALLBACK_COLOR
    );
    this.instrumentTracks.set(pitch, track);
    return track;
  }
}
