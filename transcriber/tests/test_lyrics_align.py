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

from app.pipeline.lyrics_align import (
    InputLine,
    _detect_language_from_text,
    _extract_words,
    lines_to_json,
    LyricLine,
    LyricWord,
)


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


# ---------- _extract_words end-time fallback chain --------------------


def _seg(words: list[dict[str, object]], *, end: float | None = 1.0) -> dict[str, object]:
    """Synthesise a whisperx-shaped segment dict for the extractor."""
    seg: dict[str, object] = {"words": words}
    if end is not None:
        seg["end"] = end
    return seg


def test_extract_words_passes_explicit_end_through():
    seg = _seg(
        [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
        ]
    )
    out = _extract_words(seg, segment_start=0.0)
    assert [(w.start_sec, w.end_sec, w.text) for w in out] == [
        (0.0, 0.4, "hello"),
        (0.5, 0.9, "world"),
    ]


def test_extract_words_fills_missing_end_from_next_words_start():
    """Step 1 of the fallback: a missing `end` borrows the next
    surviving word's `start`. The aligner sometimes omits `end` on
    short tokens; rather than collapse the cell, the frontend should
    see a duration that reaches the next aligned word."""
    seg = _seg(
        [
            {"word": "hi", "start": 0.0},  # missing end
            {"word": "there", "start": 0.3, "end": 0.6},
        ]
    )
    out = _extract_words(seg, segment_start=0.0)
    assert out[0].end_sec == 0.3
    assert out[1].end_sec == 0.6


def test_extract_words_falls_back_to_segment_end_for_last_word():
    """Step 2: when the missing-end word is also the last word, the
    next-word fallback can't fire, so the segment's `end` is used."""
    seg = _seg(
        [
            {"word": "final", "start": 0.5},  # missing end, no neighbor
        ],
        end=1.2,
    )
    out = _extract_words(seg, segment_start=0.0)
    assert out[0].end_sec == 1.2


def test_extract_words_falls_back_to_start_plus_epsilon_as_last_resort():
    """Step 3: when neither a next word nor a segment `end` is
    available, the cell still gets a non-zero width via a small
    epsilon. Guards the frontend's `width = max(0, ...)` math from
    ever computing a collapsed cell."""
    seg = _seg(
        [
            {"word": "alone", "start": 0.5},  # missing end
        ],
        end=None,
    )
    out = _extract_words(seg, segment_start=0.0)
    assert out[0].end_sec > out[0].start_sec
    assert out[0].end_sec == 0.5 + 0.05


def test_extract_words_clamps_inverted_end():
    """Pathological case: whisperx emits an `end` <= `start`. The
    extractor must clamp so downstream cell-width math (right edge =
    start + width) never wraps negative."""
    seg = _seg(
        [
            {"word": "weird", "start": 1.0, "end": 0.8},
        ]
    )
    out = _extract_words(seg, segment_start=0.0)
    assert out[0].end_sec == 1.0 + 0.05


def test_extract_words_skips_empty_tokens_in_neighbor_fallback():
    """Empty-text entries are filtered out before the fallback walks
    for a neighbor's `start`, so a missing-end word looks past blanks
    to the next real word for its end-time."""
    seg = _seg(
        [
            {"word": "first", "start": 0.0},  # missing end
            {"word": "  ", "start": 0.2, "end": 0.3},  # filtered out
            {"word": "third", "start": 0.4, "end": 0.6},
        ]
    )
    out = _extract_words(seg, segment_start=0.0)
    # The neighbor fallback skips the empty middle entry and lands on
    # `third`'s start at 0.4 - if it had picked the empty entry, the
    # extractor would have crashed on the missing text anyway.
    assert len(out) == 2
    assert out[0].end_sec == 0.4


def test_lines_to_json_emits_end_sec_per_word():
    """The wire format includes both start + end per word so the
    frontend can size each word's cell. `end_sec` rides alongside
    `start_sec` in camelCase."""
    lines = [
        LyricLine(
            start_sec=0.0,
            text="hello world",
            words=[
                LyricWord(start_sec=0.0, end_sec=0.4, text="hello"),
                LyricWord(start_sec=0.5, end_sec=0.9, text="world"),
            ],
        ),
    ]
    out = lines_to_json(lines)
    assert out == [
        {
            "startSec": 0.0,
            "text": "hello world",
            "words": [
                {"startSec": 0.0, "endSec": 0.4, "text": "hello"},
                {"startSec": 0.5, "endSec": 0.9, "text": "world"},
            ],
        },
    ]


def test_lines_to_json_omits_words_when_alignment_failed():
    """Whisper alignment can degrade to "transcription-only" when the
    detected language has no aligner; in that case `words` is None on
    the dataclass and the JSON drops the key entirely (not `null`)."""
    lines = [LyricLine(start_sec=1.0, text="just text", words=None)]
    out = lines_to_json(lines)
    assert out == [{"startSec": 1.0, "text": "just text"}]
