/**
 * Furigana (ruby) annotation for Japanese lyrics.
 *
 * The word-aligned lyrics row renders one chip per token. For Japanese
 * tokens containing kanji we want the hiragana reading stacked above the
 * kanji (and only the kanji, trailing/embedded okurigana stays bare),
 * the way furigana is printed in song booklets.
 *
 * The readings come from kuromoji running entirely in the browser
 * (`@sglkc/kuromoji`, a browser-capable fork): a morphological tokenizer
 * over the display text, no backend involvement and no use of the
 * aligner's Latin `romaji`. kuromoji emits a katakana reading per token;
 * we convert it to hiragana and fit it to the token's kanji runs.
 *
 * The dictionary is a few MB and loads lazily the first time a kanji is
 * seen (see {@link FuriganaAnnotator.ensureLoaded}); songs with no kanji
 * never pay for it. Until the dict resolves (or for tokens we can't fit),
 * `segmentsFor` returns the bare text, so the row renders normally and
 * upgrades in place once readings arrive.
 *
 * Context matters: kuromoji's reading depends on the whole line, so
 * {@link FuriganaAnnotator.segmentsForWords} tokenizes a line's chips
 * *together* and slices the result back onto each chip (実 reads じつ
 * inside 実は, not the lone-token み). When the forced aligner split a
 * compound across chips (盲目 · 的) but kuromoji kept it as one token, the
 * single fitted reading can't be cut; a second lazy asset, a trimmed
 * JmdictFurigana per-kanji split table (see `jmdict_furigana_loader.ts` +
 * `scripts/build-furigana-dict.ts`), supplies the per-kanji division
 * (盲/もう · 目/もく · 的/てき). Words it can't divide fall back to a
 * standalone (chip-local) tokenize.
 *
 * This module's pure helpers (`hasKanji`, `toHiragana`, `fitFurigana`,
 * `sliceLineSegments`, `expandCompounds`) carry the testable logic; the
 * singleton wires them to kuromoji + the split table + MobX.
 */
import type { IpadicFeatures, Tokenizer } from '@sglkc/kuromoji';
import { makeAutoObservable, runInAction } from 'mobx';
import {
  loadFuriganaSplitMap,
  type FuriganaSplitMap,
} from './jmdict_furigana_loader';

/** One run of base text with an optional reading. A reading is present
 *  only on kanji runs; okurigana / kana / punctuation runs carry just
 *  `base`. A whole word is an ordered array of these: the renderer maps
 *  them to `<ruby>` base text + `<rt>` pairs, the measurer to per-run
 *  width. */
export type RubySegment = {
  base: string;
  /** Hiragana reading for a kanji run. Absent on bare (kana/other) runs. */
  reading?: string;
};

/** Kanji (Han) code-point ranges we treat as needing furigana: CJK
 *  Unified Ideographs, Extension A, the compatibility block, and
 *  Extension B (lyrics occasionally reach for rarer glyphs). The
 *  iteration is code-point-wise (`for…of`) so Extension B's astral chars
 *  are handled without surrogate bookkeeping. */
function isKanjiCp(cp: number): boolean {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x20000 && cp <= 0x2a6df)
  );
}

/** True when `text` contains at least one kanji, the trigger for lazily
 *  loading the kuromoji dictionary and annotating. Pure; cheap enough to
 *  call per token. */
export function hasKanji(text: string): boolean {
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (cp !== undefined && isKanjiCp(cp)) return true;
  }
  return false;
}

/** Convert a katakana reading to hiragana (furigana convention). Shifts
 *  the katakana block (U+30A1–U+30F6) down by 0x60; leaves the prolonged
 *  sound mark `ー`, punctuation, and any already-hiragana chars intact. */
export function toHiragana(text: string): string {
  let out = '';
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    out += cp >= 0x30a1 && cp <= 0x30f6 ? String.fromCodePoint(cp - 0x60) : ch;
  }
  return out;
}

/** Coalesce neighbouring reading-less runs so the rendered `<ruby>` and
 *  the width walk see one bare text node instead of several. */
function mergeBareRuns(segs: RubySegment[]): RubySegment[] {
  const out: RubySegment[] = [];
  for (const s of segs) {
    const prev = out[out.length - 1];
    if (s.reading === undefined && prev && prev.reading === undefined) {
      prev.base += s.base;
    } else {
      out.push({ ...s });
    }
  }
  return out;
}

/**
 * Fit a token's hiragana `reading` onto its kanji runs, leaving kana
 * (okurigana) bare. Returns one {@link RubySegment} per run.
 *
 * The fit is anchor-based: kana runs in the surface must appear verbatim
 * in the reading, so each kana run pins a position and the kanji run
 * before it takes whatever reading sits between the previous anchor and
 * that kana. This resolves interleaved okurigana correctly:
 *
 *   食べる / たべる → 食=た · べる            (single okurigana run)
 *   取り引き / とりひき → 取=と · り · 引=ひ · き  (interleaved)
 *   今日   / きょう  → 今日=きょう             (jukujikun, whole compound)
 *
 * When the anchors don't line up (ateji, a reading that doesn't contain
 * the surface kana, leftover reading) we return the bare surface with no
 * furigana rather than guess, a wrong reading reads worse than none.
 * An all-kana surface likewise returns bare.
 */
export function fitFurigana(surface: string, reading: string): RubySegment[] {
  // Group the surface into maximal kanji / non-kanji runs.
  type Run = { kanji: boolean; text: string };
  const runs: Run[] = [];
  for (const ch of surface) {
    const kanji = isKanjiCp(ch.codePointAt(0)!);
    const prev = runs[runs.length - 1];
    if (prev && prev.kanji === kanji) prev.text += ch;
    else runs.push({ kanji, text: ch });
  }

  if (!runs.some((r) => r.kanji)) return [{ base: surface }];

  const bare: RubySegment[] = [{ base: surface }];
  const segs: RubySegment[] = [];
  let ri = 0;
  for (let i = 0; i < runs.length; i++) {
    const run = runs[i];
    if (!run.kanji) {
      // Anchor: the kana run must sit at the current reading position.
      if (reading.slice(ri, ri + run.text.length) !== run.text) return bare;
      segs.push({ base: run.text });
      ri += run.text.length;
      continue;
    }
    const next = runs[i + 1]; // always a kana run when present
    let slice: string;
    if (!next) {
      slice = reading.slice(ri);
      ri = reading.length;
    } else {
      const idx = reading.indexOf(next.text, ri);
      if (idx < 0) return bare;
      slice = reading.slice(ri, idx);
      ri = idx;
    }
    if (slice.length === 0) return bare;
    segs.push({ base: run.text, reading: slice });
  }
  // Reading fully consumed? A leftover tail means the anchors were wrong.
  if (ri !== reading.length) return bare;
  return mergeBareRuns(segs);
}

/**
 * Slice a whole-line segmentation back onto its individual display words.
 *
 * Readings must be fit with sentence context (a single kanji can read
 * very differently alone vs. inside a compound: 実 alone tokenizes to the
 * noun み, but 実は is the adverb じつは), so {@link FuriganaAnnotator}
 * tokenizes the words joined into one line and then this helper hands each
 * word its contiguous span of the result. `words` concatenate (in order,
 * no separators) to the same text the segments cover, so the spans are
 * just the cumulative `base` lengths.
 *
 * Returns one entry per word: its sliced segments, or `null` when the word
 * can't be cleanly sliced because a reading-bearing run straddles its
 * boundary. That happens when kuromoji groups MORE characters into one
 * fitted reading than the aligner gave to a single chip (盲目的 tokenizes
 * as one モウモクテキ token, but the aligner split it into 盲目 · 的). A
 * fitted reading can't be cut mid-run, so the caller re-tokenizes that
 * word on its own instead (盲目 → もうもく, 的 → てき). Bare runs slice
 * freely, including the single whole-line placeholder segment returned
 * while the dictionary loads, so every word renders (bare) in the meantime
 * without forcing a premature standalone tokenize.
 */
export function sliceLineSegments(
  lineSegs: readonly RubySegment[],
  words: readonly string[],
): (RubySegment[] | null)[] {
  const result: (RubySegment[] | null)[] = [];
  let spanStart = 0;
  for (const word of words) {
    const spanEnd = spanStart + word.length;
    result.push(sliceSpan(lineSegs, spanStart, spanEnd));
    spanStart = spanEnd;
  }
  return result;
}

/** Cumulative UTF-16 offsets where one word ends and the next begins, for
 *  a line built by joining `words`. Interior boundaries only (the line
 *  start and end aren't included), so {@link expandCompounds} can ask "does
 *  a chip boundary fall *inside* this reading run?". */
export function wordBoundaries(words: readonly string[]): Set<number> {
  const out = new Set<number>();
  let acc = 0;
  for (let i = 0; i < words.length - 1; i++) {
    acc += words[i].length;
    out.add(acc);
  }
  return out;
}

/**
 * Split any reading run that a chip boundary cuts into finer per-kanji
 * segments, using `lookup` (the JmdictFurigana split table).
 *
 * kuromoji groups a compound into one token with one reading (盲目的 →
 * もうもくてき), which {@link fitFurigana} keeps as a single run. When the
 * forced aligner split that compound across chips (盲目 · 的), the run
 * straddles a chip boundary and can't be sliced. `lookup` supplies the
 * precomputed per-kanji division (盲/もう · 目/もく · 的/てき) so the run
 * becomes individually-sliceable. Runs not cut by a boundary are left
 * whole (so a compound sitting on one chip keeps its single ruby), as are
 * runs `lookup` can't divide (unknown word, or a jukujikun the table only
 * has as one span), those fall through to the caller's standalone
 * fallback. Returns a new segment list; `lineSegs` is unchanged.
 */
export function expandCompounds(
  lineSegs: readonly RubySegment[],
  boundaries: ReadonlySet<number>,
  lookup: (base: string, reading: string) => RubySegment[] | null,
): RubySegment[] {
  const out: RubySegment[] = [];
  let off = 0;
  for (const seg of lineSegs) {
    const start = off;
    const end = off + seg.base.length;
    off = end;
    if (seg.reading === undefined || !hasInteriorBoundary(boundaries, start, end)) {
      out.push(seg);
      continue;
    }
    const split = lookup(seg.base, seg.reading);
    if (split && split.reduce((n, s) => n + s.base.length, 0) === seg.base.length) {
      out.push(...split);
    } else {
      out.push(seg);
    }
  }
  return out;
}

/** True when some boundary `b` falls strictly inside `(start, end)`. */
function hasInteriorBoundary(
  boundaries: ReadonlySet<number>,
  start: number,
  end: number,
): boolean {
  for (let b = start + 1; b < end; b++) {
    if (boundaries.has(b)) return true;
  }
  return false;
}

/** Segments overlapping the half-open char span `[start, end)` of a line
 *  segmentation. Bare runs are sliced to the overlap; a reading run is
 *  taken whole only when fully inside the span, otherwise the span
 *  straddles a fitted reading and we return `null` (the caller re-
 *  tokenizes the word standalone). `null` too when nothing overlaps (a
 *  degenerate span past the segmentation). Char offsets are UTF-16 units,
 *  consistent with the `word.length` spans in {@link sliceLineSegments}. */
function sliceSpan(
  lineSegs: readonly RubySegment[],
  start: number,
  end: number,
): RubySegment[] | null {
  const out: RubySegment[] = [];
  let off = 0;
  for (const seg of lineSegs) {
    const segStart = off;
    const segEnd = off + seg.base.length;
    off = segEnd;
    if (segEnd <= start || segStart >= end) continue; // no overlap
    if (seg.reading !== undefined) {
      if (segStart < start || segEnd > end) return null; // straddle
      out.push({ base: seg.base, reading: seg.reading });
    } else {
      const a = Math.max(segStart, start) - segStart;
      const b = Math.min(segEnd, end) - segStart;
      out.push({ base: seg.base.slice(a, b) });
    }
  }
  return out.length > 0 ? mergeBareRuns(out) : null;
}

/** Tokenize `text` and annotate every token, concatenating the per-token
 *  segments. Tokens with no usable reading (out-of-dictionary, `'*'`)
 *  fall through as bare text. */
function annotateText(
  tokenizer: Tokenizer<IpadicFeatures>,
  text: string,
): RubySegment[] {
  const segs: RubySegment[] = [];
  for (const t of tokenizer.tokenize(text)) {
    const surface = t.surface_form;
    const reading = t.reading;
    if (!reading || reading === '*') {
      segs.push({ base: surface });
      continue;
    }
    segs.push(...fitFurigana(surface, toHiragana(reading)));
  }
  return mergeBareRuns(segs);
}

/** Build the browser tokenizer. The loader module is dynamic-imported so
 *  kuromoji + its dict code are code-split out of the main bundle and only
 *  fetched when a song actually has kanji. `dicPath` is served from
 *  `public/` (see `scripts/copy-kuromoji-dict.ts`) and honours Vite's base
 *  URL. We use our own loader rather than kuromoji's `builder()` because the
 *  dev server serves the dict with `Content-Encoding: gzip` and kuromoji's
 *  loader double-decodes it; see `kuromoji_loader.ts`. */
async function buildTokenizer(): Promise<Tokenizer<IpadicFeatures>> {
  const { buildBrowserTokenizer } = await import('./kuromoji_loader');
  const base =
    (import.meta as unknown as { env?: { BASE_URL?: string } }).env?.BASE_URL ??
    '/';
  return buildBrowserTokenizer(`${base}kuromoji-dict`);
}

/** URL of the per-kanji split table, served from `public/` and honouring
 *  Vite's base URL (same pattern as the kuromoji dict path). */
function furiganaSplitsUrl(): string {
  const base =
    (import.meta as unknown as { env?: { BASE_URL?: string } }).env?.BASE_URL ??
    '/';
  return `${base}jmdict-furigana/furigana.txt.gz`;
}

/**
 * Lazy, cached furigana provider, exposed as a MobX-observable singleton
 * mirroring `lyricsMeasurer`'s pattern. Renderer and measurer both call
 * {@link segmentsFor}; reactivity is gated on the `revision` counter
 * (bumped when a token resolves) rather than an observable map, so a
 * burst of resolutions re-renders the row a handful of times then quiesces.
 */
class FuriganaAnnotator {
  /** True once the kuromoji dictionary has built. Drives nothing on its
   *  own; exposed for parity with `lyricsMeasurer.fontReady` and tests. */
  ready = false;
  /** Bumped each time a token's segments land in the cache. Read it in a
   *  reactive context (render / `useMemo` dep) to re-pull resolved
   *  readings. */
  revision = 0;

  private cache = new Map<string, RubySegment[]>();
  private pending = new Set<string>();
  private tokenizerPromise: Promise<Tokenizer<IpadicFeatures>> | undefined;
  /** Per-kanji split table (JmdictFurigana), loaded lazily the first time a
   *  compound straddles a chip boundary. Absent until then. */
  private furiganaSplits: FuriganaSplitMap | undefined;
  private furiganaSplitsPromise: Promise<FuriganaSplitMap> | undefined;

  constructor() {
    // Private fields must be named in the generic so MobX leaves them
    // unobserved (the cache + pending set + builder/loader promises +
    // split table are plumbing, not reactive state; reactivity is the
    // `revision` counter).
    makeAutoObservable<
      FuriganaAnnotator,
      | 'cache'
      | 'pending'
      | 'tokenizerPromise'
      | 'furiganaSplits'
      | 'furiganaSplitsPromise'
    >(this, {
      cache: false,
      pending: false,
      tokenizerPromise: false,
      furiganaSplits: false,
      furiganaSplitsPromise: false,
    });
  }

  /** Segments for a token's display text. Synchronous and cheap: returns
   *  the cached fit, or bare text while scheduling an async tokenize.
   *  Reading `this.revision` subscribes callers so they re-render when the
   *  async result lands. Tokens with no kanji never touch the dictionary. */
  segmentsFor(text: string): RubySegment[] {
    // Subscribe to resolution updates (coarse but quiescent, see above).
    void this.revision;
    if (!hasKanji(text)) return [{ base: text }];
    const cached = this.cache.get(text);
    if (cached) return cached;
    this.schedule(text);
    return [{ base: text }];
  }

  /** Context-aware furigana for a run of display words that together form
   *  one lyric line. The words are tokenized *as a single line* (joined,
   *  no separators) so each reading disambiguates against its neighbours,
   *  then the line's segmentation is sliced back onto each word; see
   *  {@link sliceLineSegments}. Returns one segment list per input word,
   *  in order. This is the path the word-aligned lyrics row uses: the
   *  forced aligner may split a compound across chips (実 / は), and
   *  tokenizing each chip alone reads 実 as the noun み instead of the
   *  じつ of 実は. Reactivity rides on {@link segmentsFor}'s `revision`
   *  read, so callers re-render in place when the dictionary resolves. */
  segmentsForWords(words: readonly string[]): RubySegment[][] {
    if (words.length === 0) return [];
    const line = this.segmentsFor(words.join(''));
    // Pre-split any compound run a chip boundary cuts, using the
    // JmdictFurigana table, so the slice below finds clean per-kanji
    // boundaries (盲目的 → 盲/もう · 目/もく · 的/てき).
    const expanded = expandCompounds(line, wordBoundaries(words), (b, r) =>
      this.lookupSplit(b, r),
    );
    const sliced = sliceLineSegments(expanded, words);
    // A remaining `null` slice means a reading run still straddles this
    // chip (the split table is unknown for this word, hasn't loaded yet, or
    // it's an indivisible jukujikun). Fall back to tokenizing the chip on
    // its own; for a real sub-word that reads correctly (盲目 → もうもく),
    // and `segmentsFor` returns bare text when there's no fittable reading.
    let straddled = false;
    const out = sliced.map((segs, i) => {
      if (segs) return segs;
      straddled = true;
      return this.segmentsFor(words[i]);
    });
    // Lazy-load the split table on first straddle so the next render can
    // divide the compound instead of leaning on the standalone fallback.
    if (straddled) this.ensureFuriganaSplitsLoaded();
    return out;
  }

  /** Per-kanji division for an all-kanji compound `base` read `reading`,
   *  or null when the split table is unloaded / lacks the word. A surface
   *  with a single recorded reading is trusted even if kuromoji's reading
   *  drifts in spelling (long vowels, small kana); homographs must match
   *  the reading exactly. */
  private lookupSplit(base: string, reading: string): RubySegment[] | null {
    const entries = this.furiganaSplits?.get(base);
    if (!entries || entries.length === 0) return null;
    if (entries.length === 1) return entries[0].segs;
    const hit = entries.find((e) => e.reading === reading);
    return hit ? hit.segs : null;
  }

  /** Kick off the one-time fetch + parse of the split table. Idempotent; on
   *  success bumps `revision` so in-place renders re-split. A failure clears
   *  the promise so a later straddle can retry. */
  private ensureFuriganaSplitsLoaded(): void {
    if (this.furiganaSplitsPromise || this.furiganaSplits) return;
    this.furiganaSplitsPromise = loadFuriganaSplitMap(furiganaSplitsUrl());
    void this.furiganaSplitsPromise
      .then((map) => {
        runInAction(() => {
          this.furiganaSplits = map;
          this.revision++;
        });
      })
      .catch(() => {
        this.furiganaSplitsPromise = undefined;
      });
  }

  private schedule(text: string): void {
    if (this.pending.has(text)) return;
    this.pending.add(text);
    void this.ensureLoaded()
      .then((tokenizer) => {
        const segs = annotateText(tokenizer, text);
        runInAction(() => {
          this.cache.set(text, segs);
          this.revision++;
        });
      })
      .catch(() => {
        // Dictionary failed to load or tokenize threw; leave the bare
        // text in place. Don't cache so a later request can retry.
      })
      .finally(() => {
        this.pending.delete(text);
      });
  }

  private ensureLoaded(): Promise<Tokenizer<IpadicFeatures>> {
    if (this.tokenizerPromise) return this.tokenizerPromise;
    this.tokenizerPromise = buildTokenizer();
    void this.tokenizerPromise
      .then(() => {
        runInAction(() => {
          this.ready = true;
        });
      })
      .catch(() => {
        // Allow a future call to rebuild after a transient failure.
        this.tokenizerPromise = undefined;
      });
    return this.tokenizerPromise;
  }
}

export const furiganaAnnotator = new FuriganaAnnotator();
