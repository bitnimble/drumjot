import { LyricLine, LyricWord } from 'src/lyrics/lrc';
import { audioSecToBeat } from 'src/lyrics/store';
import { JotTimeline } from 'src/editing/playback/timeline';

/** Per-word position metadata derived from the lyrics store + timeline.
 *  Stable under playhead movement; rebuilt only when `lines`, `offsetSec`,
 *  `timeline`, `songLeadIn`, `structuralBeats`, or `layerBeats` change. */
export type PositionedWord = {
  /** Index back into the source `line.words` array, so the JSX can
   *  compare against `activeWordIndexAt`'s return value even when
   *  out-of-range words at the line edges have been dropped. */
  sourceIdx: number;
  text: string;
  beatOffset: number;
  /** Width of this word's cell in beats: `endBeat - startBeat`.
   *  Drives the trailing-rule render in CSS via the `--lyric-word-
   *  beat-width` var; combined with `--lyric-word-shift` the cell's
   *  right edge stays anchored to the word's `endSec`. */
  beatWidth: number;
  /** Original word entry from the lyrics store, kept by reference
   *  so the JSX can build the debug tooltip (model raw times,
   *  fallback marker) without re-indexing into `line.words`. */
  source: LyricWord;
};
export type PositionedLine = {
  i: number;
  text: string;
  startBeat: number;
  endBeat: number;
  /** When defined, the row renders one absolutely-positioned span
   *  per word inside the line container (beat offsets are relative
   *  to `startBeat`). When undefined, the line falls back to the
   *  inline text (LRCLIB-style). */
  wordPositions: PositionedWord[] | undefined;
};

/** Floor for a word's cell width in beats when the aligner emits an
 *  end-time we can't resolve against the timeline (out-of-range, or
 *  collapsed by upstream clamping). Matches the Python aligner's
 *  0.05 s last-ditch epsilon scaled to "noticeable but not silly":
 *  a quarter of a beat is small enough to read as a point on the
 *  bars row at any reasonable zoom. */
export const MIN_BEAT_WIDTH = 0.05;

/** Pure beat-positioning pass for the lyrics row. Walks every line and
 *  every word once, resolving audio-sec → beat against the supplied
 *  timeline. Extracted out of the render so the result can be memoised
 *  on its real dependencies (lines / offset / timeline / structure)
 *  rather than rebuilt on every playhead tick. */
export function positionLyricLines(
  lines: readonly LyricLine[],
  timeline: JotTimeline,
  songLeadIn: number,
  structuralBeats: readonly number[],
  offsetSec: number,
  layerBeats: number,
): PositionedLine[] {
  const out: PositionedLine[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Blank lines (LRC instrumental gap stamps with no text or words)
    // produce no visible chip - rendering an empty span just leaves a
    // bare start-beat tick floating above the audio waveform.
    if (line.text.trim() === '' && (!line.words || line.words.length === 0)) {
      continue;
    }
    const lineSec = line.startSec + offsetSec;

    let startBeat: number | undefined;
    let endBeat: number | undefined;
    let wordPositions: PositionedWord[] | undefined;

    if (line.words && line.words.length > 0) {
      // Walk the words once, dropping any whose start beat falls
      // outside the timeline (rare; usually the whole line is in-
      // range or out). End-beats are resolved against the timeline
      // too; an out-of-range end falls back to `startBeat +
      // MIN_BEAT_WIDTH` so the cell has a defined, visible width.
      // The sourceIdx is preserved so word-level highlighting still
      // matches `activeWordIndexAt` (indexed against the unfiltered
      // source array) when edge words are dropped.
      const inRange: {
        sourceIdx: number;
        source: LyricWord;
        startBeat: number;
        endBeat: number;
      }[] = [];
      for (let wi = 0; wi < line.words.length; wi++) {
        const w = line.words[wi];
        const ws = audioSecToBeat(
          w.startSec + offsetSec,
          timeline,
          songLeadIn,
          structuralBeats,
        );
        if (ws === undefined) continue;
        const weRaw = audioSecToBeat(
          w.endSec + offsetSec,
          timeline,
          songLeadIn,
          structuralBeats,
        );
        const we =
          weRaw !== undefined && weRaw > ws ? weRaw : ws + MIN_BEAT_WIDTH;
        inRange.push({ sourceIdx: wi, source: w, startBeat: ws, endBeat: we });
      }
      if (inRange.length > 0) {
        startBeat = inRange[0].startBeat;
        endBeat = inRange[inRange.length - 1].endBeat;
        wordPositions = inRange.map((w) => ({
          sourceIdx: w.sourceIdx,
          text: w.source.text,
          beatOffset: w.startBeat - startBeat!,
          beatWidth: w.endBeat - w.startBeat,
          source: w.source,
        }));
      }
    } else {
      startBeat = audioSecToBeat(lineSec, timeline, songLeadIn, structuralBeats);
      if (startBeat !== undefined) {
        // End beat = next-line's start (clamped to layerBeats) so the
        // text has a defined max-width region. The final line uses
        // layerBeats as the bound.
        endBeat = layerBeats;
        for (let j = i + 1; j < lines.length; j++) {
          const next = audioSecToBeat(
            lines[j].startSec + offsetSec,
            timeline,
            songLeadIn,
            structuralBeats,
          );
          if (next !== undefined) {
            endBeat = next;
            break;
          }
        }
      }
    }

    if (startBeat === undefined || endBeat === undefined) continue;
    // Tiny non-zero floor so consecutive same-timestamp lines (or a
    // single-word line) still establish a visible positioning context
    // rather than collapsing to width 0.
    if (endBeat - startBeat < 0.05) endBeat = startBeat + 0.05;
    out.push({
      i,
      text: line.text,
      startBeat,
      endBeat,
      wordPositions,
    });
  }
  return out;
}
