/**
 * Minimal LRC parser. LRC is `[mm:ss.cc]text` per line, where the bracket
 * stamp marks the moment the line begins (in seconds, relative to the
 * recording). One physical line can carry multiple stamps so the same
 * lyric reuses across timestamps (`[00:01.23][00:30.45]chorus line`);
 * we expand those into separate logical lines.
 *
 * Empty-text stamps (`[00:34.12]`) are kept; they mark instrumental
 * gaps. Lines without any timestamp are dropped (LRC metadata tags like
 * `[ar:Artist]` / `[ti:Title]` land here and aren't lyrics).
 *
 * Word-level (enhanced LRC) syntax is not parsed today; the `LyricLine`
 * shape leaves `words` optional so a future parser can populate it
 * without breaking the renderer.
 */

export type LyricWord = {
  startSec: number;
  /** Audio-time the word's last phoneme releases (wav2vec2 forced-
   *  alignment end-time). Used by the lyrics row to size each word's
   *  visual cell so sustained notes render as a held bar rather than a
   *  point. Always present from the `/lyrics/align` backend; absent on
   *  word-less sources (the `words` array itself is optional). */
  endSec: number;
  text: string;
  /** Pre-substitution model output for `startSec`. Present whenever
   *  the aligner emitted a start time; absent when our fallback chain
   *  substituted (i.e. `startSec` is `segment_start`, not a model
   *  claim). Used by the debug tooltip in the lyrics row so the user
   *  can see what wav2vec2 actually said vs what we render. */
  rawStartSec?: number;
  /** Same idea for `endSec`; absent when our fallback chain produced
   *  the end-time. Compare with `endFallback` to know WHICH fallback
   *  fired. */
  rawEndSec?: number;
  /** Marker for when our code adjusted `endSec` away from the model's
   *  output. Absent when the rendered value matches the raw value.
   *  With the current ctc-forced-aligner pipeline only `inverted-clamp`
   *  fires; the other values are reserved for future per-segment
   *  aligners that might need similar fix-ups, so the wire vocabulary
   *  stays stable across backend swaps:
   *    - `"inverted-clamp"` model emitted `end <= start`; bumped to
   *                         `start + 0.05s`
   *    - `"next-start"`     (reserved) end borrowed from next word's start
   *    - `"segment-end"`    (reserved) end clamped to segment boundary
   *    - `"epsilon"`        (reserved) last-ditch `start + 0.05s` */
  endFallback?: 'next-start' | 'segment-end' | 'epsilon' | 'inverted-clamp';
  /** The Latin romaji actually fed to the aligner, present only when
   *  `text` is a non-Latin display surface that differs from what was
   *  aligned - i.e. Japanese tokens romanized via the backend's
   *  `jp_romaji` pass (君 displayed, `kimi` aligned). Absent for English
   *  / Chinese words, where `text` already is the aligned form. Debug-
   *  tooltip only; the renderer shows `text`. */
  romaji?: string;
};

export type LyricLine = {
  /** Audio-time (seconds from the start of the source recording) the
   *  line begins. */
  startSec: number;
  text: string;
  /** Optional sub-line word timings. Omitted for line-only sources
   *  (standard LRC). Present when enhanced LRC or forced alignment is
   *  the source; the renderer can highlight the active word inside the
   *  active line when present. */
  words?: LyricWord[];
};

/**
 * Strip noise from a lyric line: parenthetical asides (echo / harmony
 * voices like `(I'm screaming, I love you so)`) and standalone music
 * notation glyphs (♩ ♪ ♫ ♬ ♭ ♮ ♯ and the U+1D100–1D1FF musical-symbols
 * block). These aren't sung by the main vocal so they throw off the
 * forced-aligner's timing and they read as junk in the row.
 *
 * Pure text transform; doesn't decide whether the resulting empty line
 * should be dropped; that's the caller's call (LRC keeps originally-
 * empty stamps as instrumental gaps; pasted text drops empties).
 */
export function stripLyricNoise(text: string): string {
  const NOISE_RE = /\([^)]*\)|[♩-♯]|[\u{1D100}-\u{1D1FF}]/gu;
  return text.replace(NOISE_RE, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Parse an LRC document into time-ordered {@link LyricLine}s. Malformed
 * stamps inside an otherwise-good document are skipped silently; an
 * input with no parseable stamps returns an empty array.
 *
 * Lines whose visible text is entirely parenthetical asides or music
 * glyphs are dropped (see {@link stripLyricNoise}). Stamps with no
 * trailing text at all are still kept, since those mark instrumental
 * gaps the renderer relies on.
 */
export function parseLrc(input: string): LyricLine[] {
  const lines: LyricLine[] = [];
  const stampRe = /\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]/g;
  for (const raw of input.split(/\r?\n/)) {
    const stamps: number[] = [];
    let lastIdx = 0;
    stampRe.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = stampRe.exec(raw)) !== null) {
      const sec = stampToSeconds(m[1], m[2], m[3]);
      if (sec === undefined) continue;
      stamps.push(sec);
      lastIdx = stampRe.lastIndex;
    }
    if (stamps.length === 0) continue;
    const rawText = raw.slice(lastIdx).trim();
    const text = stripLyricNoise(rawText);
    // Preserve originally-empty stamps (instrumental gaps) but drop
    // lines that emptied because they were *only* noise.
    if (text.length === 0 && rawText.length > 0) continue;
    for (const s of stamps) lines.push({ startSec: s, text });
  }
  lines.sort((a, b) => a.startSec - b.startSec);
  return lines;
}

function stampToSeconds(mm: string, ss: string, frac: string | undefined): number | undefined {
  const minutes = parseInt(mm, 10);
  const seconds = parseInt(ss, 10);
  if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) return undefined;
  let fractional = 0;
  if (frac !== undefined && frac.length > 0) {
    const denom = Math.pow(10, frac.length);
    const num = parseInt(frac, 10);
    if (!Number.isFinite(num)) return undefined;
    fractional = num / denom;
  }
  return minutes * 60 + seconds + fractional;
}

/**
 * Index of the last line whose `startSec + offsetSec <= audioTimeSec`,
 * or `undefined` if the playhead is before the first line. Linear scan
 * since LRC files are typically under a few hundred lines.
 */
export function activeLineIndexAt(
  lines: readonly LyricLine[],
  audioTimeSec: number,
  offsetSec: number,
): number | undefined {
  let active: number | undefined;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startSec + offsetSec <= audioTimeSec) active = i;
    else break;
  }
  return active;
}

/**
 * Index of the word inside `lines[lineIndex]` that the playhead is
 * currently inside (the last word whose `startSec + offsetSec <=
 * audioTimeSec`). Returns `undefined` when the line has no word-level
 * alignment, or when the playhead sits before the line's first word.
 *
 * Word-aligned lyrics (LRCLIB with the word-level upgrade applied)
 * carry `words`; plain LRCLIB / file lyrics typically don't, so the
 * caller falls back to whole-line highlighting in that case.
 */
export function activeWordIndexAt(
  lines: readonly LyricLine[],
  lineIndex: number,
  audioTimeSec: number,
  offsetSec: number,
): number | undefined {
  const line = lines[lineIndex];
  if (!line || !line.words || line.words.length === 0) return undefined;
  const shifted = audioTimeSec - offsetSec;
  let active: number | undefined;
  for (let i = 0; i < line.words.length; i++) {
    if (line.words[i].startSec <= shifted) active = i;
    else break;
  }
  return active;
}
