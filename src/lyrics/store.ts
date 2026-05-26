/**
 * Session-only store for the time-aligned lyrics row. Lifecycle is
 * "owned by the current jot": every loader that replaces the song
 * (`loadJotFile`, `loadParadbMap`, `applyDebugBundle`, etc.) clears
 * this store along with the audio tracks, so a stale lyric set from
 * one song can't bleed onto the next.
 *
 * The store doesn't persist anywhere; no `globalMetadata` field, no
 * localStorage; if the user wants the same lyrics next time they reload
 * they re-run the LRCLIB search / re-load the file. Confirmed product
 * decision for v1; revisit when there's a real persistence story for
 * mid-edit jot state.
 */

import { makeAutoObservable } from 'mobx';
import { JotTimeline } from 'src/playback';
import { LyricLine, activeLineIndexAt } from './lrc';

export type LyricsSource = 'lrclib' | 'file' | 'plaintext';

/** Slider bounds for the user-facing time-offset nudger. Mirrors the
 *  drumsT0Sec / drum-offset pattern: a single uniform shift across the
 *  whole lyric row, expressed in audio seconds. ±60s covers the realistic
 *  range of nudges (file-loaded LRC from a different cut, LRCLIB match
 *  against a remaster/edit) while still acting as a sanity tripwire for
 *  the "wrong song entirely" case, which would be off by minutes. */
export const LYRICS_OFFSET_MIN_SEC = -60;
export const LYRICS_OFFSET_MAX_SEC = 60;
export const LYRICS_OFFSET_STEP_SEC = 0.01;

export class LyricsStore {
  lines: readonly LyricLine[] = [];
  source: LyricsSource | undefined = undefined;
  /** Human-readable label for the row gutter (e.g.
   *  `LRCLIB · Song Title - Artist Name`, `File · my-song.lrc`). */
  sourceLabel: string | undefined = undefined;
  offsetSec: number = 0;

  constructor() {
    makeAutoObservable(this);
  }

  load(lines: readonly LyricLine[], opts: { source: LyricsSource; sourceLabel: string }): void {
    this.lines = lines;
    this.source = opts.source;
    this.sourceLabel = opts.sourceLabel;
    // A new source resets the offset; the previous tune's nudge is
    // meaningless against a fresh recording's lyrics.
    this.offsetSec = 0;
  }

  clear(): void {
    this.lines = [];
    this.source = undefined;
    this.sourceLabel = undefined;
    this.offsetSec = 0;
  }

  setOffsetSec(sec: number): void {
    if (!Number.isFinite(sec)) return;
    this.offsetSec = Math.max(LYRICS_OFFSET_MIN_SEC, Math.min(LYRICS_OFFSET_MAX_SEC, sec));
  }

  get hasLyrics(): boolean {
    return this.lines.length > 0;
  }

  /** Index of the line whose `[startSec + offsetSec, nextStart + offsetSec)`
   *  contains `audioTimeSec`, or `undefined` if the playhead sits before
   *  the first line. */
  activeLineIndexAt(audioTimeSec: number): number | undefined {
    return activeLineIndexAt(this.lines, audioTimeSec, this.offsetSec);
  }

  /** Index of the word inside `lineIndex` that the playhead is currently
   *  inside (the last word whose `startSec + offsetSec <= audioTimeSec`).
   *  Returns `undefined` when the line has no word-level alignment, or
   *  when the playhead sits before the line's first word. Word-aligned
   *  lyrics (LRCLIB with the word-level upgrade applied) carry `words`;
   *  plain LRCLIB / file lyrics typically don't, so the caller falls
   *  back to whole-line highlighting in that case. */
  activeWordIndexAt(
    lineIndex: number,
    audioTimeSec: number,
  ): number | undefined {
    const line = this.lines[lineIndex];
    if (!line || !line.words || line.words.length === 0) return undefined;
    const shifted = audioTimeSec - this.offsetSec;
    let active: number | undefined;
    for (let i = 0; i < line.words.length; i++) {
      if (line.words[i].startSec <= shifted) active = i;
      else break;
    }
    return active;
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
