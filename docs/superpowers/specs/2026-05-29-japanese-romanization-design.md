# Japanese-aware romanization for lyrics alignment

**Date:** 2026-05-29
**Status:** Design approved, pending implementation plan

## Problem

Lyrics forced-alignment (`transcriber/app/pipeline/lyrics_align.py`) romanizes
non-Latin text before feeding it to the CTC aligner. Romanization happens
inside the third-party `ctc_forced_aligner.preprocess_text(romanize=True,
language=iso3)`, which uses **uroman** under the hood.

uroman has no Japanese kanji-reading capability: it interprets every Han
character as Chinese and emits a Mandarin-ish reading, regardless of the
`language` hint (`jpn` is not even in uroman's special-handling language
list, the `::lcode` mechanism only disambiguates *shared scripts*, e.g.
Cyrillic for Russian vs. Ukrainian). Kana are romanized acceptably; kanji
are not.

This corrupts alignment for Japanese content in **two** code paths:

1. **English-prominent / mixed tracks**, a mostly-English line with kanji
   routes to `language="eng"`, and the kanji are romanized as Chinese.
2. **Pure-J-pop tracks**, route to `language="jpn"` (char-level
   tokenization) but each kanji *still* goes through uroman → Chinese
   reading.

Same root cause, same fix seam. Scope of this work covers **both** (scope
"B" from brainstorming).

The romaji that reaches the aligner should approximate **how the Japanese is
actually sung**, so the aligner's posteriors line up with the audio. Chinese
readings of kanji do not.

## Goal

For any Japanese span (wherever it occurs), romanize it with a
Japanese-aware engine that picks contextually-correct readings, feed that
romaji to the aligner, and **display the original kana/kanji** in the
per-word cells (romaji is internal-only).

Non-goals: changing CTC model selection; changing Chinese-track handling;
re-romanizing English (status quo lowercased-normalized display for English
is unchanged).

## Approach

Convert Japanese spans to `(original-surface, romaji)` morpheme pairs
**before** the text reaches `preprocess_text`. Feed the romaji to the
aligner; carry the original surface through to display. The third-party
`preprocess_text` is left untouched; by the time it runs, the text is
already Latin and uroman no-ops.

This replaces the originally-floated "secondary-language detection pass":
Japanese is detected **per span** by Unicode block (kana → Japanese;
ambiguous kanji-only spans reuse the existing `cjk_lang` ja/zh resolver in
`_detect_language_from_text`, so Chinese tracks are never fed to a Japanese
romanizer). No detection model needed.

### Engine

- `cutlet` (romaji frontend) on `fugashi` (MeCab binding) with `unidic-lite`
  (~50 MB, bundled, no build-time download). `fugashi` is a native
  extension, so the Docker image needs a build toolchain (to verify during
  planning; likely already present).
- Lazy-imported, matching the existing lazy-import discipline in
  `lyrics_align.py`. A module-level singleton holds the loaded `Cutlet`.

## Components

### New: `transcriber/app/pipeline/jp_romaji.py`

Single responsibility, tokenize a line, romanizing Japanese:

```python
@dataclass
class JpToken:
    surface: str       # original text, for display (e.g. "君と")
    romaji: str        # latin, space-free, for alignment (e.g. "kimi")
    is_japanese: bool

def tokenize(text: str, *, treat_kanji_as_japanese: bool) -> list[JpToken]
```

Algorithm:

1. NFKC-normalize the input. (Critical: this folds full-width Latin/digits, `Ｌｏｖｅ`, `１２３`, into ASCII *before* classification, so they take the
   English path instead of being mistaken for non-Latin.)
2. Classify each char by Unicode block; group into maximal Japanese vs
   non-Japanese runs.
3. Non-Japanese runs: whitespace-split into tokens (`surface == romaji`,
   `is_japanese=False`).
4. Japanese runs: tokenize with fugashi morpheme-by-morpheme. Per morpheme,
   `surface = node.surface`, `romaji = cutlet reading`, `is_japanese=True`.
   Drop punctuation-only morphemes; collapse any internal space in a
   morpheme's romaji to keep it one space-free word.
5. A Japanese run is romanized only when `treat_kanji_as_japanese` is true
   *or* the run contains kana. (Kana presence is itself a definitive
   Japanese signal.)

### Changed: `transcriber/app/pipeline/lyrics_align.py`

- `realign_text` per-line loop: replace `units = t.split()` with
  `jp_romaji.tokenize(...)`. Feed each token's **`romaji`** to
  `preprocess_text(romanize=True, language="eng")` (already Latin → uroman
  no-ops, and avoids the char-level branch). Build a parallel
  `display_surfaces` list alongside `line_word_counts`.
- `treat_kanji_as_japanese` is driven by the existing `cjk_lang` resolver. If
  the track resolves to Chinese, **skip `jp_romaji` entirely** and keep
  today's `chi` char-level path untouched (cutlet is Japanese-only).
- The old `per_word = iso3 not in {"jpn","chi"}` special-case: Japanese now
  becomes romaji-word-level (one morpheme = one word = one `<star>` slot
  between morphemes), which is a strict improvement over today's char-level
  Japanese handling. Char-level remains only for Chinese.
- `_stitch_lines` takes `display_surfaces` and sets `LyricWord.text =
  surface` instead of the romaji.

### Changed: wire + frontend

- `LyricWord` gains one **optional** `romaji?: string` field
  (`lyrics_align.py` serialization + `src/lyrics/lrc.ts`) for the debug
  tooltip only.
- **No rendering change required:** the frontend already renders `word.text`
  (lyrics_row.tsx:410). Setting `LyricWord.text` to the original surface on
  the backend delivers display option B by construction.

## Load-bearing invariant

Each emitted alignment word ↔ exactly one display surface ↔ one fugashi
morpheme (or one Latin word). `line_word_counts` and `display_surfaces` stay
length-locked, so `_partition_words_by_line`'s existing hard-fail guard
catches any drift and degrades to line-level output gracefully.

Two edges the implementation must pin with tests:
- Punctuation-only morphemes (、。「」), dropped from both alignment and
  display, consistent with current `preprocess_text` stripping.
- A morpheme whose romaji contains an internal space, collapsed to one
  space-free word so the 1:1 mapping holds.

## Model routing, unchanged

`_detect_language_from_text` → `_pick_alignment_model` stays as-is
(ja→MMS-300m, en→wav2vec2-large-robust). Romaji is Latin, which both heads
align fine. This work changes *what text the aligner sees*, not *which
model*.

## Error handling

- cutlet/fugashi import fails → log once, set a flag, fall back to passing
  original Japanese through `preprocess_text` (today's uroman-as-Chinese
  behavior). Degraded, never crashes.
- Per-line tokenize exception → fall back to `t.split()` for that line
  (current behavior).

## Testing

`jp_romaji` unit tests:
- pure-English, pure-kana, kana+kanji, mixed En+Ja
- full-width Latin (`Ｌｏｖｅ` → English path), full-width digits
- punctuation handling
- kanji-only-no-kana under both `treat_kanji_as_japanese` values

Integration (extend `transcriber/tests/test_lyrics_align.py`):
- mixed line → correct `line_word_counts`
- display surfaces are the original characters
- romaji is what reaches the aligner
- partition stays 1:1
- cutlet-missing fallback (monkeypatch import to fail)

## Dependencies

Add to `transcriber/pyproject.toml`: `cutlet`, `fugashi`, `unidic-lite`.
No Dockerfile change needed: the image installs project deps via
`pip install -e .` (Dockerfile:128), so the three are picked up
automatically, and `build-essential` is already present (Dockerfile:59,
for madmom) to satisfy `fugashi`'s native extension if a wheel isn't
used. User drives the local `.venv` install (uv; install ordering
matters).
