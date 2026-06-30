/**
 * Enhanced-LRC serializer / parser that round-trips word-aligned lyrics
 * *including per-word durations* through a single text file.
 *
 * Standard enhanced LRC marks only word *start* times with inline
 * `<mm:ss.cc>` tags; the word's end is left implicit (the next word's
 * start). Our model carries an explicit `endSec` per word (sustains,
 * gaps), so we extend the format: every word is bracketed by a start AND
 * an end tag,
 *
 *     [00:12.340]<00:12.340>Hello<00:12.900> <00:13.000>world<00:13.450>
 *
 * The `<end>` tag is our addition; the gap between a word's end and the
 * next word's start encodes silence. Other tools ignore the trailing
 * tag and still read the line; our own loader recovers exact durations.
 * Lines with no word timings serialize as plain `[mm:ss.ccc]text`.
 *
 * A leading `[offset:±ms]` line round-trips the live offset slider
 * (`LyricsTrack.offsetSec`). Unlike the standard tag we do NOT bake it
 * into the timestamps; it restores the row's nudge verbatim.
 *
 * Word text is backslash-escaped (`\` `<` `>`) since the model permits
 * those literally (e.g. a `<3`) and they would otherwise collide with
 * the tag delimiters. Plain-line text is written verbatim, mirroring the
 * line-level {@link parseLrc} contract.
 *
 * Only the essential fields (text, startSec, endSec) are encoded; the
 * aligner's debug provenance (`rawStartSec` / `rawEndSec` / `endFallback`)
 * and the Japanese `romaji` surface are intentionally dropped, they
 * don't affect rendering or playback, and furigana is recomputed
 * client-side on load.
 */
import { LyricLine, LyricWord, stripLyricNoise } from './lrc';

/** Last-ditch cell width (seconds) for a word whose end can't be
 *  resolved from the file (start-only foreign files, or a final word
 *  with no following start). Mirrors the renderer's MIN_BEAT_WIDTH
 *  intent and the Python aligner's 0.05s epsilon. */
const EPSILON_SEC = 0.05;

function pad(n: number, width: number): string {
  return String(n).padStart(width, '0');
}

/** Format an audio-second as `mm:ss.ccc` (millisecond precision).
 *  Negative inputs clamp to zero, lyric timestamps are non-negative. */
export function secondsToStamp(sec: number): string {
  const totalMs = Math.max(0, Math.round(sec * 1000));
  const mm = Math.floor(totalMs / 60000);
  const rem = totalMs % 60000;
  const ss = Math.floor(rem / 1000);
  const ms = rem % 1000;
  return `${pad(mm, 2)}:${pad(ss, 2)}.${pad(ms, 3)}`;
}

/** Parse the captured groups of a `mm:ss[.frac]` stamp into seconds.
 *  `frac` is interpreted by its digit count (`.cc` = centiseconds,
 *  `.ccc` = milliseconds). Returns undefined on a non-numeric stamp. */
function stampPartsToSeconds(
  mm: string,
  ss: string,
  frac: string | undefined,
): number | undefined {
  const minutes = parseInt(mm, 10);
  const seconds = parseInt(ss, 10);
  if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) return undefined;
  let fractional = 0;
  if (frac !== undefined && frac.length > 0) {
    const num = parseInt(frac, 10);
    if (!Number.isFinite(num)) return undefined;
    fractional = num / Math.pow(10, frac.length);
  }
  return minutes * 60 + seconds + fractional;
}

function escapeWordText(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/</g, '\\<').replace(/>/g, '\\>');
}

/**
 * Serialize lyric lines to an enhanced-LRC document. Word-aligned lines
 * emit `<start>text<end>` per word; line-only lines stay plain. Times
 * are written raw (the offset is encoded separately, never baked in).
 */
export function serializeEnhancedLrc(
  lines: readonly LyricLine[],
  opts?: { offsetSec?: number },
): string {
  const out: string[] = [];
  const offMs =
    opts?.offsetSec !== undefined && Number.isFinite(opts.offsetSec)
      ? Math.round(opts.offsetSec * 1000)
      : 0;
  if (offMs !== 0) out.push(`[offset:${offMs}]`);
  for (const line of lines) {
    const stamp = secondsToStamp(line.startSec);
    if (line.words && line.words.length > 0) {
      const parts = line.words.map(
        (w) =>
          `<${secondsToStamp(w.startSec)}>${escapeWordText(w.text)}<${secondsToStamp(w.endSec)}>`,
      );
      out.push(`[${stamp}]${parts.join(' ')}`);
    } else {
      out.push(`[${stamp}]${line.text}`);
    }
  }
  return out.join('\n') + '\n';
}

type WordToken = { type: 'tag'; sec: number } | { type: 'text'; value: string };

/** Inline tag matched at the very start of the remaining string. */
const WORD_TAG_RE = /^<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>/;

/** Split a word-aligned line's remainder (everything after the line
 *  stamp) into an alternating-ish stream of time tags and text runs.
 *  Text runs honour backslash escapes and stop at the next unescaped
 *  `<`; a stray unescaped `<` that isn't a valid tag is consumed as a
 *  literal character so the scan always makes progress. */
function tokenizeWordLine(s: string): WordToken[] {
  const tokens: WordToken[] = [];
  let i = 0;
  const n = s.length;
  while (i < n) {
    if (s[i] === '<') {
      const m = WORD_TAG_RE.exec(s.slice(i));
      if (m) {
        const sec = stampPartsToSeconds(m[1], m[2], m[3]);
        if (sec !== undefined) {
          tokens.push({ type: 'tag', sec });
          i += m[0].length;
          continue;
        }
      }
    }
    let val = '';
    while (i < n) {
      const c = s[i];
      if (c === '\\' && i + 1 < n) {
        val += s[i + 1];
        i += 2;
        continue;
      }
      if (c === '<') break;
      val += c;
      i++;
    }
    if (val === '' && i < n && s[i] === '<') {
      // Unescaped '<' that didn't form a valid tag: take it literally.
      val = '<';
      i++;
    }
    if (val !== '') tokens.push({ type: 'text', value: val });
  }
  return tokens;
}

/** Assemble {@link LyricWord}s from a word-aligned line's remainder.
 *  Each word is `start-tag, text, [end-tag]`. The end tag is recognized
 *  by lookahead: a tag immediately followed by more text belongs to the
 *  *next* word (a start tag) and leaves the current word end-less; a tag
 *  followed by whitespace / another tag / end-of-line is this word's end
 *  tag and is consumed. End-less words (foreign start-only files) fall
 *  back to the next word's start, else `start + EPSILON_SEC`. */
function parseWordSegments(remainder: string): LyricWord[] {
  const tokens = tokenizeWordLine(remainder);
  const partial: { startSec: number; endSec?: number; text: string }[] = [];
  let j = 0;
  while (j < tokens.length) {
    const t = tokens[j];
    if (t.type !== 'tag') {
      j++; // stray separator / leading text
      continue;
    }
    const startSec = t.sec;
    j++;
    let text = '';
    const textTok = tokens[j];
    if (textTok && textTok.type === 'text') {
      text = textTok.value;
      j++;
    }
    let endSec: number | undefined;
    const endTok = tokens[j];
    if (endTok && endTok.type === 'tag') {
      const after = tokens[j + 1];
      const afterIsText =
        after !== undefined && after.type === 'text' && after.value.trim() !== '';
      endSec = endTok.sec;
      if (!afterIsText) j++; // consume as this word's end tag
    }
    text = text.trim();
    if (text === '') continue;
    partial.push({ startSec, endSec, text });
  }

  const words: LyricWord[] = [];
  for (let k = 0; k < partial.length; k++) {
    const w = partial[k];
    let end = w.endSec;
    if (end === undefined || end <= w.startSec) {
      const nextStart = partial[k + 1]?.startSec;
      end =
        nextStart !== undefined && nextStart > w.startSec
          ? nextStart
          : w.startSec + EPSILON_SEC;
    }
    words.push({ startSec: w.startSec, endSec: end, text: w.text });
  }
  return words;
}

const LINE_STAMP_RE = /\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]/g;
const OFFSET_RE = /^\s*\[offset:\s*([+-]?\d+)\s*\]\s*$/i;
const WORD_TAG_DETECT_RE = /<\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?>/;

/**
 * Parse an enhanced-LRC document into time-ordered lines plus the
 * decoded offset. A superset of {@link parseLrc}: lines bearing inline
 * `<time>` tags become word-aligned (`words` populated with start/end);
 * lines without them parse exactly as line-level LRC (multi-stamp
 * expansion, noise stripping, instrumental-gap preservation). Tolerant
 * of standard start-only enhanced LRC from other tools.
 */
export function parseEnhancedLrc(input: string): {
  lines: LyricLine[];
  offsetSec: number;
} {
  const lines: LyricLine[] = [];
  let offsetSec = 0;
  for (const raw of input.split(/\r?\n/)) {
    const offM = raw.match(OFFSET_RE);
    if (offM) {
      const ms = parseInt(offM[1], 10);
      if (Number.isFinite(ms)) offsetSec = ms / 1000;
      continue;
    }

    const stamps: number[] = [];
    let lastIdx = 0;
    LINE_STAMP_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = LINE_STAMP_RE.exec(raw)) !== null) {
      const sec = stampPartsToSeconds(m[1], m[2], m[3]);
      if (sec === undefined) continue;
      stamps.push(sec);
      lastIdx = LINE_STAMP_RE.lastIndex;
    }
    if (stamps.length === 0) continue;
    const remainder = raw.slice(lastIdx);

    if (WORD_TAG_DETECT_RE.test(remainder)) {
      const words = parseWordSegments(remainder);
      if (words.length > 0) {
        // Word-aligned lines carry a single line stamp; `text` is the
        // space-joined token surface (drives the chip tooltip only, the
        // renderer uses `words`). For CJK this re-inserts token spaces,
        // which is cosmetic and tooltip-only.
        const text = words.map((w) => w.text).join(' ');
        lines.push({ startSec: stamps[0], text, words });
        continue;
      }
      // Malformed (tag-shaped but no parseable words): fall through to
      // plain handling so the line isn't silently dropped.
    }

    const text = stripLyricNoise(remainder);
    // Preserve originally-empty stamps (instrumental gaps); drop lines
    // that emptied because they were only noise.
    if (text.length === 0 && remainder.trim().length > 0) continue;
    for (const s of stamps) lines.push({ startSec: s, text });
  }
  lines.sort((a, b) => a.startSec - b.startSec);
  return { lines, offsetSec };
}
