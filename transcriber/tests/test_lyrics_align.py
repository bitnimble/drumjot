"""Tests for `_detect_language_from_text`.

Guards the realign-path language pick. The hard regression we guard
against (one per-letter "word" instead of word-level) fires when an
English LRC was routed through a no-space-language aligner because
audio-based detection mis-classified the first 30 s of the vocals
stem. Tests pin the SCRIPT-based decision tree end-to-end. Imports
are kept narrow so the test module doesn't transitively pull in
whisperx / torch.
"""

from __future__ import annotations

from app.pipeline.lyrics_align import InputLine, _detect_language_from_text


def _lines(*texts: str) -> list[InputLine]:
    """Build a list of InputLines from raw text strings."""
    return [InputLine(start_sec=float(i), text=t) for i, t in enumerate(texts)]


def test_detect_language_plain_english():
    assert _detect_language_from_text(_lines("been together forever")) == "en"


def test_detect_language_japanese_pure_kana():
    """Hiragana / katakana are definitive Japanese signals."""
    assert _detect_language_from_text(_lines("はじめまして")) == "ja"
    assert _detect_language_from_text(_lines("カタカナ")) == "ja"


def test_detect_language_japanese_kanji_with_kana():
    """A typical J-pop line - kanji + kana - must resolve to ja."""
    assert _detect_language_from_text(_lines("愛してる Forever")) == "ja"


def test_detect_language_kanji_leading_japanese():
    """Regression: the two-pass detector must look past leading kanji
    and still tag as ja when kana appears later in the string. The
    single-pass first-script-wins detector returned zh on this input."""
    assert _detect_language_from_text(_lines("恋人達 ステレオ")) == "ja"


def test_detect_language_ambiguous_cjk_defaults_to_japanese():
    """Kanji-only text (no kana, no Chinese-only markers) is genuinely
    ambiguous between Japanese and Traditional Chinese. The library
    bias is toward Japanese music, so the default is ja."""
    assert _detect_language_from_text(_lines("漢字")) == "ja"
    assert _detect_language_from_text(_lines("夜明け前")) == "ja"
    assert _detect_language_from_text(_lines("Hello, 世界")) == "ja"


def test_detect_language_chinese_via_simplified_marker():
    """Presence of any simplified-Chinese-only glyph routes to zh -
    overrides the ja default for ambiguous CJK."""
    assert _detect_language_from_text(_lines("我爱你")) == "zh"
    assert _detect_language_from_text(_lines("我们这里")) == "zh"
    assert _detect_language_from_text(_lines("听说")) == "zh"


def test_detect_language_chinese_marker_in_one_of_many_lines():
    """A single line with a simplified marker should be enough to flip
    the whole input to zh; the detector concatenates lines before
    scanning so the marker doesn't have to be on every line.

    Fixture is kanji-only (no kana) on the first line to isolate the
    marker check - any kana would correctly short-circuit to ja and
    defeat the test's intent."""
    lines = _lines("詩歌", "我爱你", "more text")
    assert _detect_language_from_text(lines) == "zh"


def test_detect_language_korean():
    assert _detect_language_from_text(_lines("안녕하세요")) == "ko"


def test_detect_language_thai():
    assert _detect_language_from_text(_lines("สวัสดี")) == "th"


def test_detect_language_empty_returns_none():
    """Empty / whitespace / non-alphabetic input returns None so the
    caller falls back to audio-based detection."""
    assert _detect_language_from_text([]) is None
    assert _detect_language_from_text(_lines("")) is None
    assert _detect_language_from_text(_lines("   \n  ")) is None
    assert _detect_language_from_text(_lines("!!! ???")) is None
    assert _detect_language_from_text(_lines("1234")) is None


def test_detect_language_concatenates_lines():
    """Detection scans all lines, not just the first - a leading
    instrumental marker shouldn't shadow the actual lyric content."""
    lines = _lines("", "  ", "Hello world")
    assert _detect_language_from_text(lines) == "en"


def test_detect_language_kana_wins_over_chinese_marker():
    """Codeswitched J-pop that happens to contain a glyph also used as a
    simplified-Chinese marker should still resolve as ja - any kana is
    a definitive Japanese signal that beats the Chinese-marker check."""
    # 时 is in the simplified-Chinese marker set, but the kana に here
    # is conclusive evidence the text is Japanese (not Chinese).
    assert _detect_language_from_text(_lines("時に 时")) == "ja"
