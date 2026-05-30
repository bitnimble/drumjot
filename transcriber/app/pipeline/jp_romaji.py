"""Japanese-aware romanization for lyrics forced-alignment.

uroman (invoked inside `ctc_forced_aligner.preprocess_text`) reads
kanji as Chinese. This module pre-romanizes Japanese spans with cutlet
(on fugashi/MeCab + unidic-lite) so the aligner sees romaji that
matches the sung audio, while the original kana/kanji are preserved for
display.

The module is two layers:

  - Pure Unicode logic (`_normalize`, `_split_runs`): NFKC-normalize and
    partition a line into maximal Japanese vs non-Japanese runs. No
    third-party dependency; deterministic and unit-tested directly.
  - cutlet/fugashi morpheme romanization (`tokenize`): lazy-imported, so
    importing this module is cheap and a box without the Japanese stack
    can still import (and the caller degrades gracefully).
"""

from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

# Japanese POS tags (unidic `pos1`) whose morphemes carry no spoken
# content - punctuation, symbols, whitespace. Dropped from both
# alignment and display, matching how `preprocess_text` strips
# punctuation on the English path.
_DROP_POS1 = frozenset({"補助記号", "空白", "記号"})


def _normalize(text: str) -> str:
    """NFKC-normalize `text`.

    Folds full-width Latin / digits (`Ｌｏｖｅ`, `１２３`) to their ASCII
    forms *before* any script classification, so they take the
    non-Japanese path instead of being mistaken for a non-Latin run.
    """
    return unicodedata.normalize("NFKC", text)


def _is_kana(cp: int) -> bool:
    """Hiragana + katakana (incl. the prolonged sound mark ー at 0x30FC,
    which lives in the katakana block) and the katakana phonetic
    extensions. Kana is a definitive Japanese signal."""
    return (0x3040 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF)


def _is_han(cp: int) -> bool:
    """CJK ideographs (kanji): the main block, Extension A, the
    compatibility block, and the iteration mark 々 (0x3005, e.g. 人々).
    Glyph-ambiguous between Japanese and Chinese on its own - the
    caller's `treat_kanji_as_japanese` flag decides a kanji-only run."""
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0xF900 <= cp <= 0xFAFF)
        or cp == 0x3005
    )


@dataclass
class JpToken:
    """One alignment unit produced by {@link tokenize}.

    `surface` is the original text for display (e.g. "君と" -> two tokens
    "君", "と"); `romaji` is the Latin form fed to the aligner (lowercase,
    ASCII, space-free). For non-Japanese tokens the two are identical.
    """

    surface: str
    romaji: str
    is_japanese: bool


# Lazy cutlet singleton. Constructed on first `tokenize` call (loading
# the unidic-lite dictionary is not free); held for the process
# lifetime. `use_foreign_spelling=False` keeps loanword readings
# phonetic (カツカレー -> "katsu karee", not "Cutlet curry") so the
# romaji matches what is actually sung rather than the source-language
# spelling.
_cutlet_instance: Any | None = None
_cutlet_lock = threading.Lock()


def _cutlet() -> Any:
    global _cutlet_instance
    if _cutlet_instance is None:
        with _cutlet_lock:
            if _cutlet_instance is None:
                import cutlet  # type: ignore[import-not-found]  # lazy + optional

                _cutlet_instance = cutlet.Cutlet(use_foreign_spelling=False)
    return _cutlet_instance


def tokenize(text: str, *, treat_kanji_as_japanese: bool) -> list[JpToken]:
    """Tokenize a line into ordered {@link JpToken}s, romanizing Japanese.

    Japanese runs (per {@link _split_runs}) are segmented into morphemes
    by fugashi; each morpheme keeps its original surface for display and
    carries a contextual Japanese reading as `romaji`. Punctuation-only
    morphemes are dropped. Non-Japanese runs are whitespace-split and
    passed through verbatim (`romaji == surface`), so English is left
    for the existing aligner path untouched.

    Requires the cutlet/fugashi stack; callers that may run without it
    must guard the import (see `lyrics_align`'s graceful fallback).
    """
    tokens: list[JpToken] = []
    for run_text, is_japanese in _split_runs(
        text, treat_kanji_as_japanese=treat_kanji_as_japanese
    ):
        if is_japanese:
            for node in _cutlet().tagger(run_text):
                if node.feature.pos1 in _DROP_POS1:
                    continue
                romaji = "".join(_cutlet().romaji_word(node).split()).lower()
                if not romaji:
                    continue
                tokens.append(
                    JpToken(surface=node.surface, romaji=romaji, is_japanese=True)
                )
        else:
            for word in run_text.split():
                # Drop punctuation-only splits (e.g. a stray 、 or "-"
                # that fell into a non-Japanese run); the existing
                # English path strips these too via preprocess_text.
                if not any(c.isalnum() for c in word):
                    continue
                tokens.append(JpToken(surface=word, romaji=word, is_japanese=False))
    return tokens


def _split_runs(text: str, *, treat_kanji_as_japanese: bool) -> list[tuple[str, bool]]:
    """NFKC-normalize then partition `text` into maximal runs, each
    tagged `is_japanese`.

    Grouping is by script: contiguous kana/kanji characters form one
    CJK run, everything else (Latin, digits, punctuation, spaces) forms
    non-CJK runs. A CJK run is Japanese when it contains any kana, or
    when it is kanji-only and `treat_kanji_as_japanese` is set; this is
    why a kanji adjacent to kana (僕の心) stays one Japanese run while a
    bare kanji stretch (心) is gated by the flag. Non-CJK runs are never
    Japanese.
    """
    normalized = _normalize(text)
    runs: list[tuple[str, bool]] = []
    buf: list[str] = []
    buf_is_cjk: bool | None = None
    buf_has_kana = False

    def flush() -> None:
        if not buf:
            return
        is_japanese = (buf_has_kana or treat_kanji_as_japanese) if buf_is_cjk else False
        runs.append(("".join(buf), is_japanese))

    for ch in normalized:
        cp = ord(ch)
        is_kana = _is_kana(cp)
        is_cjk = is_kana or _is_han(cp)
        if buf_is_cjk is None or is_cjk == buf_is_cjk:
            buf.append(ch)
            buf_is_cjk = is_cjk
            buf_has_kana = buf_has_kana or is_kana
        else:
            flush()
            buf = [ch]
            buf_is_cjk = is_cjk
            buf_has_kana = is_kana
    flush()
    return runs
