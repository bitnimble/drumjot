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
 * This module's pure helpers (`hasKanji`, `toHiragana`, `fitFurigana`)
 * carry the testable logic; the singleton wires them to kuromoji + MobX.
 */
import type { IpadicFeatures, Tokenizer } from '@sglkc/kuromoji';
import { makeAutoObservable, runInAction } from 'mobx';

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

/** Build the browser tokenizer. Dynamic-imported so kuromoji + its dict
 *  loader are code-split out of the main bundle and only fetched when a
 *  song actually has kanji. `dicPath` is served from `public/` (see
 *  `scripts/copy-kuromoji-dict.mjs`) and honours Vite's base URL. */
async function buildTokenizer(): Promise<Tokenizer<IpadicFeatures>> {
  const { builder } = await import('@sglkc/kuromoji');
  const base =
    (import.meta as unknown as { env?: { BASE_URL?: string } }).env?.BASE_URL ??
    '/';
  const dicPath = `${base}kuromoji-dict`;
  return new Promise<Tokenizer<IpadicFeatures>>((resolve, reject) => {
    builder({ dicPath }).build((err, tokenizer) => {
      if (err) reject(err);
      else resolve(tokenizer);
    });
  });
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

  constructor() {
    // Private fields must be named in the generic so MobX leaves them
    // unobserved (the cache + pending set + builder promise are plumbing,
    // not reactive state; reactivity is the `revision` counter).
    makeAutoObservable<
      FuriganaAnnotator,
      'cache' | 'pending' | 'tokenizerPromise'
    >(this, {
      cache: false,
      pending: false,
      tokenizerPromise: false,
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
