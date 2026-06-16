/**
 * Unified `Track` abstraction. Every mixer row, regardless of kind, is a
 * `Track` with at least an observable `color`. This file defines the
 * shared interface, the {@link InstrumentTrack} class (per-lane view-
 * model for the score's note-colour override), and the {@link MixerContext}
 * the rest of the app uses to wire audio-track colour inheritance back
 * to the active instrument tracks.
 *
 * Why each track gets a class instead of staying a plain object: the
 * colour picker in the overflow menu writes to `track.color` directly,
 * so the read-and-write path is the same field. For instrument tracks
 * the field is an outright store-owned override (no fallback computed
 * by the class itself, since the palette default is a function of the
 * active jot). For audio tracks the field is a computed getter that
 * falls back through the {@link resolveAudioInheritedColor} chain so
 * the user only ever has to pick a colour when they actually want to
 * override the inheritance.
 */

import { makeAutoObservable } from 'mobx';
import type { AudioTrackId } from 'src/editing/playback/audio_tracks';

/**
 * Pre-determined swatch row shown in the colour-picker popover. Doubles
 * as `ViewConfig.palette`'s default value so the picker palette and the
 * jot's lane-colour assignments share one source of truth; a colour
 * picked from the picker's first swatch will visually match a lane that
 * happened to land on the same palette slot.
 */
export const PICKER_PALETTE: readonly string[] = [
  '#FF8C55',
  '#5BA8E8',
  '#7BC74D',
  '#C77DFF',
  '#FFD166',
  '#EF476F',
  '#06AED5',
  '#8D6E63',
];

/** Neutral grey for instrument tracks with no palette default available. */
export const INSTRUMENT_FALLBACK_COLOR = '#7e7e7e';
/** Neutral blue for audio tracks with no override and no grouped instrument
 *  to inherit from. Matches the legacy `laneColor ?? '#5BA8E8'` waveform
 *  fallback the chunk worker has baked in. */
export const AUDIO_FALLBACK_COLOR = '#5BA8E8';
/** Lyrics tracks have no visible colour today; constant neutral so the
 *  Track interface is still satisfied. */
export const LYRICS_FALLBACK_COLOR = '#7e7e7e';

/** Shared contract every mixer row's track satisfies. */
export interface Track {
  readonly kind: 'audio' | 'instrument' | 'lyrics';
  /** Currently-effective colour. For audio tracks this resolves through
   *  the override → grouped instrument → neutral chain; for instrument
   *  tracks it resolves through override → jot palette default → neutral;
   *  for lyrics tracks it returns a fixed neutral. Always a `#rrggbb`
   *  string (or a CSS-var expression in the rare "no palette default
   *  available" fallback for instruments). */
  readonly color: string;
}

/**
 * Stable per-(layer, lane) key for the drum mute/solo/volume filter. A layer
 * holds at most one track per lane (the DSL constraint), so `${layerId}/${lane}`
 * uniquely identifies an instrument track. Single-layer songs key as
 * `v0/<lane>`, so behaviour matches the old per-lane filter; multi-layer songs
 * get independent control of the same lane in different layers. Layer ids are
 * converter slugs (`v0`, `v1`) and lanes are single letters, so neither
 * contains a slash. `jotToEvents` stamps each event's `layerId`, so the
 * scheduler can recompute this key per event.
 */
export function trackKey(layerId: string, lane: string): string {
  return `${layerId}/${lane}`;
}

/**
 * Lookup surface the audio-track colour computation needs to walk
 * grouped instrument tracks. Implemented by the UI store
 * (`MixerStore`) and handed to the player at startup so freshly-
 * loaded {@link AudioTrack}s can resolve their inherited colour without
 * either side importing the other directly.
 */
export interface MixerContext {
  /** Instrument lanes sharing the audio track's group in `jot.ordering`,
   *  in slot order. Empty when the audio row sits in a loose run (not
   *  grouped). Reads the doc ordering, so calling it inside a MobX
   *  derivation tracks regroups. The single "which instrument(s) is this
   *  audio paired with" source; both the audio track's derived `lane`
   *  and {@link resolveAudioInheritedColor} read it. */
  groupInstrumentLanesForAudio(audioId: AudioTrackId): string[];
  /** Get (or lazily create) the {@link InstrumentTrack} for a DSL lane.
   *  Used as the terminal step of the audio-track inheritance chain so
   *  the inherited colour stays reactive to per-instrument overrides. */
  getInstrumentTrack(lane: string): InstrumentTrack;
}

/**
 * Per-lane view-model for the score's note-colour override. Lives in
 * the UI store keyed by DSL lane letter; survives jot reloads (override
 * is store-owned, not jot-owned), so a kit colour customisation persists
 * across song loads.
 *
 * The fallback closure is supplied at construction by the store and
 * resolves the active jot's palette default for this lane on every
 * read; so loading a new jot with a different palette assignment for
 * the same letter automatically updates the unfilled track without any
 * imperative refresh from the store side.
 */
export class InstrumentTrack implements Track {
  readonly kind = 'instrument' as const;
  /** Per-track user override (`#rrggbb`). `undefined` means "fall back
   *  to the active jot's palette assignment for this lane". */
  _color: string | undefined = undefined;

  /**
   * @param lane       DSL lane letter this row represents.
   * @param fallback    Producer of the default colour to use when no
   *                    override is set. Called on every read of
   *                    {@link color}; closes over the active jot so a
   *                    jot replace re-reads automatically.
   */
  constructor(
    readonly lane: string,
    private readonly fallback: () => string,
  ) {
    makeAutoObservable<this, 'fallback'>(this, { fallback: false });
  }

  get color(): string {
    return this._color ?? this.fallback();
  }

  set color(c: string) {
    this._color = c;
  }

  /** Drop the override so the row reverts to the palette default. */
  clearColor(): void {
    this._color = undefined;
  }

  /** Whether the user has set an explicit override on this row. Drives
   *  the picker popover's Reset enable state. */
  get hasOverride(): boolean {
    return this._color !== undefined;
  }
}

/**
 * Compute the colour an audio track inherits when its own override is
 * unset. Resolution order:
 *
 *   1. Collect the instrument lanes sharing the audio track's group in
 *      `jot.ordering` (via {@link MixerContext.groupInstrumentLanesForAudio}).
 *      If the audio's explicit `lane` link matches one, use that
 *      instrument's `color`; otherwise the first instrument in the group.
 *   2. If the audio row sits in a loose run (no group), fall back to its
 *      load-time `lane` mapping when set; so a per-lane stem placed loose
 *      (the default for freshly-synced audio) still tints to its
 *      instrument, else return undefined (a plain backing track stays
 *      neutral).
 *
 * Callers fall back to {@link AUDIO_FALLBACK_COLOR} when this returns
 * undefined. Pure (reads only through `ctx`); the read of
 * `groupInstrumentLanesForAudio` is what keeps the colour reactive to
 * regroups.
 */
export function resolveAudioInheritedColor(
  audioId: AudioTrackId,
  audioLane: string | undefined,
  ctx: MixerContext,
): string | undefined {
  const instrumentsInGroup = ctx.groupInstrumentLanesForAudio(audioId);
  if (instrumentsInGroup.length === 0) {
    return audioLane !== undefined ? ctx.getInstrumentTrack(audioLane).color : undefined;
  }
  if (audioLane !== undefined && instrumentsInGroup.includes(audioLane)) {
    return ctx.getInstrumentTrack(audioLane).color;
  }
  return ctx.getInstrumentTrack(instrumentsInGroup[0]).color;
}
