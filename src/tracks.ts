/**
 * Unified `Track` abstraction. Every mixer row, regardless of kind, is a
 * `Track` with at least an observable `color`. This file defines the
 * shared interface, the {@link InstrumentTrack} class (per-pitch view-
 * model for the score's note-colour override), the {@link MixerContext}
 * the rest of the app uses to wire audio-track colour inheritance back
 * to the active instrument tracks, and the {@link TrackKey} identity
 * union the mixer's row-order list keys on.
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
import type { AudioTrackId } from 'src/playback/audio_tracks';
import type { LyricsTrackId } from 'src/lyrics/store';

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
 *  to inherit from. Matches the legacy `pitchColor ?? '#5BA8E8'` waveform
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
 * Identity tag for a mixer row. Distinct from the track instance itself
 * because the row order is persisted as an array of these (small,
 * serializable, stable across reloads); the actual {@link Track}
 * instance is looked up lazily by id/pitch on render.
 *
 * `groupId` is a UI-only clustering tag; consecutive entries that share
 * the same id render flush (no inter-row gap); a transition to a
 * different id (or to/from `undefined`) draws the small inter-group
 * gap. The audio-track colour inheritance uses it as the "which
 * instrument tracks count as 'grouped' with me" lookup key (see
 * {@link resolveAudioInheritedColor}). Identity for sync/move purposes
 * is `kind + id/pitch` only; `trackKeyEq` ignores `groupId`.
 */
export type TrackKey =
  | { kind: 'audio'; id: AudioTrackId; groupId?: string }
  | { kind: 'instrument'; pitch: string; groupId?: string }
  | { kind: 'lyrics'; id: LyricsTrackId; groupId?: string };

export function trackKeyEq(a: TrackKey, b: TrackKey): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'audio') return a.id === (b as { kind: 'audio'; id: AudioTrackId }).id;
  if (a.kind === 'instrument') {
    return a.pitch === (b as { kind: 'instrument'; pitch: string }).pitch;
  }
  return a.id === (b as { kind: 'lyrics'; id: LyricsTrackId }).id;
}

/**
 * Lookup surface the audio-track colour computation needs to walk
 * grouped instrument tracks. Implemented by the UI store
 * (`JotViewStore`) and handed to the player at startup so freshly-
 * loaded {@link AudioTrack}s can resolve their inherited colour without
 * either side importing the other directly.
 */
export interface MixerContext {
  /** Live row-order list. The audio-track colour computation walks this
   *  twice: once to find its own entry (and read its `groupId`), again
   *  to find instrument keys sharing the same `groupId`. Trivially small
   *  in practice (<20 entries), so the cost is negligible. */
  readonly trackOrder: readonly TrackKey[];
  /** Get (or lazily create) the {@link InstrumentTrack} for a DSL pitch.
   *  Used as the terminal step of the audio-track inheritance chain so
   *  the inherited colour stays reactive to per-instrument overrides. */
  getInstrumentTrack(pitch: string): InstrumentTrack;
}

/**
 * Per-pitch view-model for the score's note-colour override. Lives in
 * the UI store keyed by DSL pitch letter; survives jot reloads (override
 * is store-owned, not jot-owned), so a kit colour customisation persists
 * across song loads.
 *
 * The fallback closure is supplied at construction by the store and
 * resolves the active jot's palette default for this pitch on every
 * read; so loading a new jot with a different palette assignment for
 * the same letter automatically updates the unfilled track without any
 * imperative refresh from the store side.
 */
export class InstrumentTrack implements Track {
  readonly kind = 'instrument' as const;
  /** Per-track user override (`#rrggbb`). `undefined` means "fall back
   *  to the active jot's palette assignment for this pitch". */
  _color: string | undefined = undefined;

  /**
   * @param pitch       DSL pitch letter this row represents.
   * @param fallback    Producer of the default colour to use when no
   *                    override is set. Called on every read of
   *                    {@link color}; closes over the active jot so a
   *                    jot replace re-reads automatically.
   */
  constructor(
    readonly pitch: string,
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
 *   1. Find the audio's `TrackKey` in `ctx.trackOrder`. If it has no
 *      `groupId`, return undefined (no group means no inheritance).
 *   2. Collect instrument keys sharing the same `groupId`. If the audio
 *      track has an explicit `pitch` link that matches one of them, use
 *      that instrument's `color`. Otherwise use the first instrument
 *      in the group (by `trackOrder` index).
 *   3. If no instrument keys are in the group, return undefined.
 *
 * Callers fall back to {@link AUDIO_FALLBACK_COLOR} when this returns
 * undefined. Kept as a pure function so the same logic can be unit-
 * tested directly against a synthetic `MixerContext` if we ever need
 * to.
 */
export function resolveAudioInheritedColor(
  audioId: AudioTrackId,
  audioPitch: string | undefined,
  ctx: MixerContext,
): string | undefined {
  let groupId: string | undefined;
  for (const k of ctx.trackOrder) {
    if (k.kind === 'audio' && k.id === audioId) {
      groupId = k.groupId;
      break;
    }
  }
  if (groupId === undefined) return undefined;
  const instrumentsInGroup: string[] = [];
  for (const k of ctx.trackOrder) {
    if (k.kind === 'instrument' && k.groupId === groupId) {
      instrumentsInGroup.push(k.pitch);
    }
  }
  if (instrumentsInGroup.length === 0) return undefined;
  if (audioPitch !== undefined && instrumentsInGroup.includes(audioPitch)) {
    return ctx.getInstrumentTrack(audioPitch).color;
  }
  return ctx.getInstrumentTrack(instrumentsInGroup[0]).color;
}
