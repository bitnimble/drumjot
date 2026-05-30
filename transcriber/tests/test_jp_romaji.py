"""Tests for `app.pipeline.jp_romaji`.

Split in two: the pure Unicode logic (`_normalize`, `_split_runs`) has
no third-party dependency and is tested directly; the cutlet/fugashi-
backed `tokenize` morpheme path is tested behind an importorskip so the
suite still collects on a box without the Japanese stack installed.
"""

from __future__ import annotations

import pytest

from app.pipeline.jp_romaji import _normalize, _split_runs


def test_normalize_folds_fullwidth_latin_and_digits():
    """NFKC must fold full-width Latin / digits to ASCII *before* any
    script classification, so `Ｌｏｖｅ` is recognised as English rather
    than mistaken for a non-Latin run."""
    assert _normalize("Ｌｏｖｅ１２３") == "Love123"


def test_split_runs_pure_english_is_one_non_japanese_run():
    assert _split_runs("been together", treat_kanji_as_japanese=True) == [
        ("been together", False),
    ]


def test_split_runs_pure_kana_is_one_japanese_run():
    """Kana is a definitive Japanese signal regardless of the flag."""
    assert _split_runs("はじめまして", treat_kanji_as_japanese=False) == [
        ("はじめまして", True),
    ]


def test_split_runs_mixed_alternates_runs_in_order():
    """A mixed line splits into ordered runs, English untouched."""
    assert _split_runs("君と dance", treat_kanji_as_japanese=True) == [
        ("君と", True),
        (" dance", False),
    ]


def test_split_runs_kanji_only_follows_flag():
    """Kanji with no kana is ambiguous (could be Chinese). The flag
    decides: ja-resolved tracks romanize it, otherwise it passes through
    as a non-Japanese run."""
    assert _split_runs("心", treat_kanji_as_japanese=True) == [("心", True)]
    assert _split_runs("心", treat_kanji_as_japanese=False) == [("心", False)]


def test_split_runs_kana_run_keeps_kanji_when_flag_off():
    """Even with the flag off, a run containing kana stays Japanese as a
    whole (the kanji rides with its kana neighbours within the morpheme
    run); only a kanji-only stretch is gated by the flag."""
    assert _split_runs("僕の心", treat_kanji_as_japanese=False) == [
        ("僕の心", True),
    ]


def test_split_runs_normalizes_fullwidth_before_classifying():
    """Full-width Latin folds to ASCII and lands on the English path."""
    assert _split_runs("Ｌｏｖｅ", treat_kanji_as_japanese=True) == [
        ("Love", False),
    ]


# --- cutlet/fugashi-backed morpheme romanization -------------------
# Skips cleanly on a box without the Japanese stack so the pure-logic
# tests above still run.

pytest.importorskip("cutlet")

from app.pipeline.jp_romaji import tokenize  # noqa: E402


def test_tokenize_english_only_is_passthrough():
    toks = tokenize("been together", treat_kanji_as_japanese=True)
    assert [(t.surface, t.romaji, t.is_japanese) for t in toks] == [
        ("been", "been", False),
        ("together", "together", False),
    ]


def test_tokenize_japanese_surfaces_preserve_original_characters():
    """Display option B: per-token surface is the original kana/kanji;
    romaji is internal, lowercase ASCII, and space-free."""
    toks = tokenize("ねこ", treat_kanji_as_japanese=True)
    assert "".join(t.surface for t in toks) == "ねこ"
    assert all(t.is_japanese for t in toks)
    for t in toks:
        assert t.romaji.isascii() and t.romaji.islower() and " " not in t.romaji


def test_tokenize_kanji_uses_japanese_reading_not_chinese():
    """The whole point: 心 reads as kokoro (Japanese), never xin
    (uroman's Chinese reading)."""
    toks = tokenize("心", treat_kanji_as_japanese=True)
    assert [t.surface for t in toks] == ["心"]
    assert toks[0].romaji == "kokoro"


def test_tokenize_mixed_keeps_english_verbatim_and_romanizes_japanese():
    toks = tokenize("君と dance", treat_kanji_as_japanese=True)
    eng = [t for t in toks if not t.is_japanese]
    assert any(t.surface == "dance" and t.romaji == "dance" for t in eng)
    ja = [t for t in toks if t.is_japanese]
    assert "".join(t.surface for t in ja) == "君と"
    assert all(t.romaji.isascii() and " " not in t.romaji for t in ja)


def test_tokenize_drops_japanese_punctuation_morphemes():
    """Punctuation-only morphemes (、) are dropped from both alignment
    and display, consistent with preprocess_text's stripping."""
    toks = tokenize("好き、本当", treat_kanji_as_japanese=True)
    assert "、" not in [t.surface for t in toks]
    assert all(t.romaji for t in toks)


def test_tokenize_kanji_only_passthrough_when_flag_off():
    """Flag off -> a kanji-only run is left for the existing
    (uroman) path: surface unchanged, not marked Japanese, romaji is
    the original text (we don't romanize it ourselves)."""
    toks = tokenize("心", treat_kanji_as_japanese=False)
    assert [(t.surface, t.romaji, t.is_japanese) for t in toks] == [
        ("心", "心", False),
    ]
