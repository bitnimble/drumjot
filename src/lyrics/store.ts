/**
 * Session-only store for the time-aligned lyrics rows. Lifecycle is
 * "owned by the current jot": every loader that replaces the song
 * (`loadJotFile`, `loadParadbMap`, `applyDebugBundle`, etc.) clears
 * this store along with the audio tracks, so a stale lyric set from
 * one song can't bleed onto the next.
 *
 * The store doesn't persist anywhere; no `globalMetadata` field, no
 * localStorage; if the user wants the same lyrics next time they
 * reload they re-run the LRCLIB search / re-load the file.
 *
 * Multi-track: callers `add()` a track and get back a stable
 * `LyricsTrackId`; subsequent loads are additive (a new file or LRCLIB
 * pick creates another row rather than replacing the singleton).
 * Duets, side-by-side comparison of alignments, and multi-language
 * lyrics all become independent rows in the mixer keyed by id. The
 * source-label collision suffix (` (2)`, ` (3)`) is computed in
 * `add()` so callers always pass the natural label.
 */

import { makeAutoObservable } from 'mobx';
import { JotTimeline } from 'src/jot_view/playback/timeline';
import { LYRICS_FALLBACK_COLOR } from 'src/jot_view/tracks/tracks';
import { LyricLine } from './lrc';

export type LyricsSource = 'lrclib' | 'file' | 'plaintext';

export type LyricsTrackId = string;

/**
 * One lyrics row. Immutable from the consumer's perspective; the store
 * swaps the whole object on mutation (offset nudge, word-level
 * upgrade), so React/MobX observers re-render off identity changes.
 */
export type LyricsTrack = {
  readonly id: LyricsTrackId;
  readonly lines: readonly LyricLine[];
  readonly source: LyricsSource;
  readonly sourceLabel: string;
  readonly offsetSec: number;
  /**
   * Satisfies the unified {@link import('src/jot_view/tracks/tracks').Track} interface.
   * Lyrics rows have no visible per-row colour today; the fixed neutral
   * value here is enough to let downstream code (the picker, future
   * tinting) treat every mixer row uniformly. The overflow menu
   * deliberately omits a Colour control for this kind, since changing
   * the value has no current visual effect.
   */
  readonly color: string;
};

/** Slider bounds for the user-facing time-offset nudger. Mirrors the
 *  drumsT0Sec / drum-offset pattern: a single uniform shift across one
 *  lyric row, expressed in audio seconds. ±60s covers the realistic
 *  range of nudges (file-loaded LRC from a different cut, LRCLIB match
 *  against a remaster/edit) while still acting as a sanity tripwire for
 *  the "wrong song entirely" case, which would be off by minutes. */
export const LYRICS_OFFSET_MIN_SEC = -60;
export const LYRICS_OFFSET_MAX_SEC = 60;
export const LYRICS_OFFSET_STEP_SEC = 0.01;

/**
 * Module-level monotonic id allocator. Session-scoped, so a page reload
 * resets the sequence; the store never sees a `lyrics-0`. Ids leak into
 * `TrackKey` and React keys, so collisions must not happen within a
 * session even if the user nukes and re-adds many rows.
 */
let nextLyricsTrackSeq = 1;
function allocLyricsTrackId(): LyricsTrackId {
  return `lyrics-${nextLyricsTrackSeq++}`;
}

export class LyricsStore {
  // ObservableMap via MobX. Insertion order = render order seed for
  // `syncTrackOrder`'s "slot a new lyrics row next to existing ones"
  // policy. Replacing an entry preserves its insertion position.
  private tracksMap: Map<LyricsTrackId, LyricsTrack> = new Map();

  constructor() {
    makeAutoObservable(this);
  }

  /** Insert a new track, returning its allocated id. Source-label
   *  collisions are disambiguated with ` (2)`, ` (3)`, etc. so callers
   *  always pass the natural label (e.g. `LRCLIB · X - Y`). */
  add(
    lines: readonly LyricLine[],
    opts: { source: LyricsSource; sourceLabel: string },
  ): LyricsTrackId {
    const id = allocLyricsTrackId();
    const sourceLabel = this.uniqueSourceLabel(opts.sourceLabel);
    this.tracksMap.set(id, {
      id,
      lines,
      source: opts.source,
      sourceLabel,
      offsetSec: 0,
      color: LYRICS_FALLBACK_COLOR,
    });
    return id;
  }

  /** Swap a track's lines in place. Preserves `offsetSec` (the user may
   *  have nudged); preserves `source` / `sourceLabel` unless explicitly
   *  overridden via `opts`. No-op when `id` is unknown (the caller's
   *  align job may have raced a removal). */
  replace(
    id: LyricsTrackId,
    lines: readonly LyricLine[],
    opts: { source?: LyricsSource; sourceLabel?: string } = {},
  ): void {
    const existing = this.tracksMap.get(id);
    if (!existing) return;
    this.tracksMap.set(id, {
      ...existing,
      lines,
      source: opts.source ?? existing.source,
      sourceLabel: opts.sourceLabel ?? existing.sourceLabel,
    });
  }

  /** Drop one track. No-op when `id` is unknown. */
  remove(id: LyricsTrackId): void {
    this.tracksMap.delete(id);
  }

  /** Drop every track. Called by wholesale-song-reload paths. */
  clear(): void {
    this.tracksMap.clear();
  }

  /** Update one track's offset, clamping to `[LYRICS_OFFSET_MIN_SEC,
   *  LYRICS_OFFSET_MAX_SEC]`. Non-finite values are rejected (preserves
   *  the previous value). No-op when `id` is unknown. */
  setOffsetSec(id: LyricsTrackId, sec: number): void {
    if (!Number.isFinite(sec)) return;
    const existing = this.tracksMap.get(id);
    if (!existing) return;
    const clamped = Math.max(LYRICS_OFFSET_MIN_SEC, Math.min(LYRICS_OFFSET_MAX_SEC, sec));
    this.tracksMap.set(id, { ...existing, offsetSec: clamped });
  }

  get(id: LyricsTrackId): LyricsTrack | undefined {
    return this.tracksMap.get(id);
  }

  /** Snapshot of ids in insertion order. Consumers (`syncTrackOrder`,
   *  test code) iterate via this rather than poking at the internal
   *  Map. */
  get trackIds(): readonly LyricsTrackId[] {
    return Array.from(this.tracksMap.keys());
  }

  get hasAnyLyrics(): boolean {
    return this.tracksMap.size > 0;
  }

  private uniqueSourceLabel(label: string): string {
    const existing = new Set<string>();
    for (const t of this.tracksMap.values()) existing.add(t.sourceLabel);
    if (!existing.has(label)) return label;
    let n = 2;
    while (existing.has(`${label} (${n})`)) n++;
    return `${label} (${n})`;
  }
}

export const lyricsStore = new LyricsStore();

/**
 * Convert an audio-time second to a beat offset on the row's bars-row.
 * Mirrors how `AudioTrackWaveformCanvas` maps bar slices: the audio
 * recording's `t` lands at jot time `t - drumsT0Sec`; that jot time is
 * looked up against the per-bar `BarTiming` table, linearly interpolated
 * within its bar, and the resulting fraction is converted back to beats
 * by walking the structural bars' `beats` sums.
 *
 * Returns `undefined` when the time falls outside `[firstBar.startSec,
 * lastBar.endSec)`; the row drops out-of-range lines from rendering
 * rather than clamping them to the edges, since clamped piles of
 * lines at bar 0 / the final bar would read as broken alignment.
 */
export function audioSecToBeat(
  audioTimeSec: number,
  timeline: JotTimeline,
  drumsT0Sec: number,
  structuralBeats: readonly number[],
): number | undefined {
  const jotTime = audioTimeSec - drumsT0Sec;
  const bars = timeline.bars;
  if (bars.length === 0 || structuralBeats.length !== bars.length) return undefined;
  const first = bars[0];
  const last = bars[bars.length - 1];
  if (jotTime < first.startSec) return undefined;
  if (jotTime >= last.startSec + last.durationSec) return undefined;
  let cumBeats = 0;
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    if (jotTime < bar.startSec + bar.durationSec) {
      const within = bar.durationSec > 0 ? (jotTime - bar.startSec) / bar.durationSec : 0;
      return cumBeats + within * structuralBeats[i];
    }
    cumBeats += structuralBeats[i];
  }
  return undefined;
}
