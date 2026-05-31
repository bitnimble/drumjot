"""Tests for `_detect_language_from_text`.

Guards the realign-path language pick. The hard regression we guard
against (one per-letter "word" instead of word-level) fires when an
English LRC was routed through a no-space-language aligner because
audio-based detection mis-classified the first 30 s of the vocals
stem. Tests pin the **majority-by-character-count** routing rule
end-to-end. Imports are kept narrow so the test module doesn't
transitively pull in ctc-forced-aligner / torch.
"""

from __future__ import annotations

import pytest

from app.pipeline.lyrics_align import (
    InputLine,
    LyricLine,
    LyricWord,
    _detect_language_from_text,
    _iso1_to_iso3,
    _partition_words_by_line,
    _preprocess_lines,
    _resolve_cjk_lang,
    _resolve_cyrillic_lang,
    _stitch_lines,
    lines_to_json,
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


def test_detect_language_japanese_majority_over_english_sprinkle():
    """A line whose Japanese characters outnumber its Latin characters
    resolves to ja under majority routing. Pure Japanese phrases with
    only a short English ad-lib stay on the multilingual aligner."""
    # 9 kanji/kana ja chars vs 4 latin chars -> ja wins.
    assert _detect_language_from_text(_lines("僕の心に住む君よ love")) == "ja"


def test_detect_language_english_majority_over_japanese_sprinkle():
    """Symmetric to the above and the case the majority rule was
    designed for: an English-dominant line with a few Japanese
    ad-libs routes to en so it picks up the English-specialized
    aligner instead of being dragged onto MMS by a stray kana."""
    # 7 latin chars vs 3 kana -> en wins.
    assert _detect_language_from_text(_lines("Forever 愛してる")) == "en"


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


def test_detect_language_latin_majority_beats_cjk_sprinkle():
    """Under majority routing, a single CJK glyph no longer flips the
    whole line. `Hello, 世界` is 5 latin + 2 CJK characters; en wins
    on count. This is the case the routing change was designed for -
    an English-dominant line with a tiny non-Latin sprinkle picks up
    the English aligner instead of being dragged onto MMS."""
    assert _detect_language_from_text(_lines("Hello, 世界")) == "en"


def test_detect_language_chinese_via_simplified_marker():
    """Presence of any simplified-Chinese-only glyph routes to zh -
    overrides the ja default for ambiguous CJK."""
    assert _detect_language_from_text(_lines("我爱你")) == "zh"
    assert _detect_language_from_text(_lines("我们这里")) == "zh"
    assert _detect_language_from_text(_lines("听说")) == "zh"


def test_detect_language_chinese_marker_in_one_of_many_lines():
    """A single line containing a simplified-Chinese-only glyph
    disambiguates all the ambiguous CJK characters in the same input
    to zh; the detector does the marker scan once over the full
    concatenated text before counting. The marker doesn't have to be
    on every line.

    Fixture is kanji-only (no kana, no Latin) so the marker is the
    only signal in play and ALL the ambiguous CJK chars get counted
    as zh, giving zh the majority. Any kana would correctly
    short-circuit to ja and defeat the test's intent; any Latin chars
    in a count exceeding the CJK count would flip the majority to en
    (see `test_detect_language_latin_majority_beats_cjk_sprinkle`)."""
    lines = _lines("詩歌", "我爱你", "回家")
    assert _detect_language_from_text(lines) == "zh"


def test_detect_language_korean():
    assert _detect_language_from_text(_lines("안녕하세요")) == "ko"


def test_detect_language_thai():
    assert _detect_language_from_text(_lines("สวัสดี")) == "th"


def test_detect_language_cyrillic_ukrainian():
    """Ukrainian-specific letters (і ї є ґ) route to uk."""
    assert _detect_language_from_text(_lines("Соловей співає і")) == "uk"
    assert _detect_language_from_text(_lines("їхав козак")) == "uk"


def test_detect_language_cyrillic_russian():
    """Russian-specific letters (ы э ё) route to ru; bare Cyrillic with
    no distinctive marker also defaults to ru."""
    assert _detect_language_from_text(_lines("ты моё солнце")) == "ru"
    assert _detect_language_from_text(_lines("Калинка малинка")) == "ru"


def test_detect_language_cyrillic_bulgarian():
    """ъ as a full vowel, without the Russian markers, routes to bg."""
    assert _detect_language_from_text(_lines("България е тъжна")) == "bg"


def test_detect_language_cyrillic_serbian():
    assert _detect_language_from_text(_lines("Ђурђевдан")) == "sr"


def test_detect_language_cyrillic_macedonian():
    assert _detect_language_from_text(_lines("Ѓоко ќе дојде")) == "mk"


def test_detect_language_cyrillic_majority_over_latin_sprinkle():
    """Mostly-Cyrillic with a Latin ad-lib still routes to the Cyrillic
    language (majority by character count) - the case that previously
    fell through to the audio detector."""
    assert _detect_language_from_text(_lines("Стефанія мамо oh")) == "uk"


def test_detect_language_other_script_routes_to_mms_not_none():
    """A script we don't classify into a language (Greek, Hebrew, ...)
    must NOT return None - it routes to the multilingual MMS + uroman
    path via the `und` tag, so we never drop onto the audio detector."""
    assert _detect_language_from_text(_lines("Καλημέρα κόσμε")) == "und"  # Greek
    assert _detect_language_from_text(_lines("שלום עולם")) == "und"  # Hebrew


def test_resolve_cyrillic_lang_defaults_to_russian():
    """No distinctive marker -> ru (most common Cyrillic lyric language)."""
    assert _resolve_cyrillic_lang("привет мир") == "ru"
    assert _resolve_cyrillic_lang("їхали") == "uk"
    assert _resolve_cyrillic_lang("тъга") == "bg"


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


# ---------- _iso1_to_iso3 -----------------------------------------------


def test_iso1_to_iso3_maps_text_detector_outputs():
    """Every code that `_detect_language_from_text` can emit must round-
    trip to a real ISO-639-3 value, otherwise `preprocess_text` would
    raise on the alignment path. This pins the contract so a future
    text-detector addition doesn't silently degrade to `eng`.

    Note: `zh -> chi` (not `cmn`) is load-bearing. The aligner's
    `preprocess_text` checks `language in ["jpn", "chi"]` to switch
    to char-level tokenisation; routing Chinese through `cmn` would
    fall through to word-split and shatter against the no-space input.
    """
    assert _iso1_to_iso3("en") == "eng"
    assert _iso1_to_iso3("ja") == "jpn"
    assert _iso1_to_iso3("ko") == "kor"
    assert _iso1_to_iso3("zh") == "chi"
    assert _iso1_to_iso3("th") == "tha"
    # Cyrillic-script languages emitted by `_resolve_cyrillic_lang`.
    assert _iso1_to_iso3("ru") == "rus"
    assert _iso1_to_iso3("uk") == "ukr"
    assert _iso1_to_iso3("bg") == "bul"
    assert _iso1_to_iso3("sr") == "srp"
    assert _iso1_to_iso3("mk") == "mkd"
    # The `_OTHER_SCRIPT_LANG` catch-all (`und`) intentionally rides the
    # unknown-code default to `eng`: uroman romanizes the actual
    # (non-Latin) script by codepoint, `eng` is a code the aligner
    # accepts, and the non-`en` tag still routes to MMS, not the English
    # head. This is correct behaviour, not the degradation the test below
    # guards against.
    assert _iso1_to_iso3("und") == "eng"


def test_iso1_to_iso3_falls_back_for_unknown_codes():
    """Unknown / typo'd codes degrade to `eng` rather than raising;
    MMS-300m still aligns Latin-script text reasonably under English
    romanization, so a misdetected language stays recoverable."""
    assert _iso1_to_iso3("xx") == "eng"
    assert _iso1_to_iso3("") == "eng"


def test_iso1_to_iso3_is_case_insensitive():
    """Whisper's language head sometimes emits uppercase codes."""
    assert _iso1_to_iso3("EN") == "eng"
    assert _iso1_to_iso3("Ja") == "jpn"


# ---------- _partition_words_by_line ------------------------------------


def _word(text: str, start: float, end: float) -> dict[str, object]:
    """Synthesise a ctc-forced-aligner-shaped word dict."""
    return {"text": text, "start": start, "end": end, "score": 1.0}


def test_partition_words_by_line_splits_by_counts():
    words = [
        _word("a", 0.0, 0.1),
        _word("b", 0.1, 0.2),
        _word("c", 0.2, 0.3),
        _word("d", 0.3, 0.4),
    ]
    out = _partition_words_by_line(words, [2, 1, 1])
    assert out is not None
    assert [len(group) for group in out] == [2, 1, 1]
    assert [w["text"] for w in out[0]] == ["a", "b"]
    assert [w["text"] for w in out[1]] == ["c"]
    assert [w["text"] for w in out[2]] == ["d"]


def test_partition_words_by_line_returns_none_on_count_mismatch():
    """A mismatch means our `text.split()` view of the input diverges
    from the aligner's tokenisation (typically a non-Latin script where
    romanisation reshapes word boundaries). We return None rather than
    guess at boundaries; the caller degrades to line-level output."""
    words = [_word("a", 0.0, 0.1), _word("b", 0.1, 0.2)]
    assert _partition_words_by_line(words, [3]) is None
    assert _partition_words_by_line(words, [1]) is None


def test_partition_words_by_line_handles_zero_word_lines():
    """An entry with count=0 produces an empty group at its slot; the
    sum of counts still has to equal the total word count for the
    partition to succeed. Defensive - `_build_concat_text` filters
    these out so the realign path won't actually pass zero-count
    entries today, but the helper stays robust if a caller does."""
    words = [_word("a", 0.0, 0.1)]
    out = _partition_words_by_line(words, [0, 1, 0])
    assert out is not None
    assert out[0] == []
    assert [w["text"] for w in out[1]] == ["a"]
    assert out[2] == []


# ---------- _stitch_lines -----------------------------------------------


def test_stitch_lines_preserves_empty_text_lines():
    """Empty-text input lines pass through with `words=None` even when
    surrounded by aligned lines, so the response mirrors the original
    line count + ordering 1:1."""
    input_lines = [
        InputLine(start_sec=0.0, text="hello world"),
        InputLine(start_sec=5.0, text=""),
        InputLine(start_sec=10.0, text="goodbye"),
    ]
    non_empty_indices = [0, 2]
    partitioned = [
        [_word("hello", 0.1, 0.4), _word("world", 0.5, 0.9)],
        [_word("goodbye", 10.2, 10.8)],
    ]
    out = _stitch_lines(input_lines, non_empty_indices, partitioned)
    assert len(out) == 3
    assert out[0].text == "hello world"
    assert out[0].words is not None
    assert [(w.start_sec, w.end_sec, w.text) for w in out[0].words] == [
        (0.1, 0.4, "hello"),
        (0.5, 0.9, "world"),
    ]
    assert out[1].text == ""
    assert out[1].words is None
    assert out[1].start_sec == 5.0
    assert out[2].words is not None
    assert out[2].words[0].text == "goodbye"


def test_stitch_lines_refines_line_start_from_first_word():
    """A non-empty line's `start_sec` is overwritten with the first
    aligned word's start - that's the whole point of running the
    aligner; the caller's input timestamp was a rough estimate."""
    input_lines = [InputLine(start_sec=0.0, text="hi")]
    out = _stitch_lines(input_lines, [0], [[_word("hi", 7.3, 7.5)]])
    assert out[0].start_sec == 7.3


def test_stitch_lines_clamps_inverted_word():
    """Defensive clamp for the rare case CTC alignment emits end<=start
    (a held vowel that wav2vec2 absorbed into a neighbour, etc.). The
    end gets bumped by an epsilon and the fallback marker is set so the
    UI debug tooltip can flag it - vocabulary is backend-agnostic so
    the frontend doesn't have to know which aligner produced the data."""
    input_lines = [InputLine(start_sec=0.0, text="weird")]
    out = _stitch_lines(input_lines, [0], [[_word("weird", 1.0, 0.8)]])
    assert out[0].words is not None
    w = out[0].words[0]
    assert w.start_sec == 1.0
    assert w.end_sec == 1.05
    assert w.end_fallback == "inverted-clamp"


def test_stitch_lines_drops_empty_aligned_text():
    """If the aligner somehow emits a word entry with empty text (the
    `<star>` filter in postprocess_results is occasionally leaky), we
    skip it rather than emit a zero-width invisible cell."""
    input_lines = [InputLine(start_sec=0.0, text="hi there")]
    partitioned = [[_word("hi", 0.1, 0.2), _word("", 0.3, 0.4)]]
    out = _stitch_lines(input_lines, [0], partitioned)
    assert out[0].words is not None
    assert [w.text for w in out[0].words] == ["hi"]


def test_lines_to_json_emits_end_sec_per_word():
    """The wire format includes both start + end per word so the
    frontend can size each word's cell. `end_sec` rides alongside
    `start_sec` in camelCase. Debug fields are omitted when None so
    the common-case payload stays tight."""
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


def test_lines_to_json_emits_debug_fields_when_present():
    """Raw model values + fallback marker ride along on the wire when
    the extractor set them, so the UI tooltip can show "model said X,
    we render Y". The frontend reads them as optional fields."""
    lines = [
        LyricLine(
            start_sec=0.0,
            text="hi there",
            words=[
                LyricWord(
                    start_sec=0.0,
                    end_sec=0.3,
                    text="hi",
                    raw_start_sec=0.0,
                    raw_end_sec=None,
                    end_fallback="next-start",
                ),
                LyricWord(
                    start_sec=0.3,
                    end_sec=0.6,
                    text="there",
                    raw_start_sec=0.3,
                    raw_end_sec=0.6,
                    end_fallback=None,
                ),
            ],
        ),
    ]
    out = lines_to_json(lines)
    words = out[0]["words"]
    # Substituted-end word: rawEndSec omitted, endFallback present.
    assert words[0] == {
        "startSec": 0.0,
        "endSec": 0.3,
        "text": "hi",
        "rawStartSec": 0.0,
        "endFallback": "next-start",
    }
    # Model-clean word: raw values mirror the final, no endFallback.
    assert words[1] == {
        "startSec": 0.3,
        "endSec": 0.6,
        "text": "there",
        "rawStartSec": 0.3,
        "rawEndSec": 0.6,
    }


def test_lines_to_json_omits_words_when_alignment_failed():
    """Whisper alignment can degrade to "transcription-only" when the
    detected language has no aligner; in that case `words` is None on
    the dataclass and the JSON drops the key entirely (not `null`)."""
    lines = [LyricLine(start_sec=1.0, text="just text", words=None)]
    out = lines_to_json(lines)
    assert out == [{"startSec": 1.0, "text": "just text"}]


# --------------------------------------------------------------------
# Japanese-aware romanization wiring
# --------------------------------------------------------------------


def test_resolve_cjk_lang_kana_is_ja():
    assert _resolve_cjk_lang("はじめまして") == "ja"


def test_resolve_cjk_lang_simplified_marker_is_zh():
    assert _resolve_cjk_lang("这是") == "zh"


def test_resolve_cjk_lang_kanji_only_defaults_ja():
    """Glyph-ambiguous kanji with no kana / markers defaults to the
    J-pop bias."""
    assert _resolve_cjk_lang("心") == "ja"


def test_stitch_lines_uses_original_surface_and_records_romaji():
    """With per-word display surfaces, the cell text is the original
    kana/kanji and the aligned romaji rides along in the debug field."""
    inputs = _lines("君と")
    partitioned = [
        [
            {"text": "kimi", "start": 0.0, "end": 0.5, "score": -0.2},
            {"text": "to", "start": 0.5, "end": 0.8, "score": -0.3},
        ]
    ]
    surfaces = [["君", "と"]]
    out = _stitch_lines(inputs, [0], partitioned, surfaces)
    words = out[0].words
    assert [w.text for w in words] == ["君", "と"]
    assert [w.romaji for w in words] == ["kimi", "to"]


def test_stitch_lines_without_surfaces_uses_aligned_text():
    """English / Chinese path (no surface override): cell text stays the
    aligned text and romaji is absent - status quo behavior."""
    inputs = _lines("hello world")
    partitioned = [
        [
            {"text": "hello", "start": 0.0, "end": 0.5, "score": -0.1},
            {"text": "world", "start": 0.5, "end": 1.0, "score": -0.1},
        ]
    ]
    out = _stitch_lines(inputs, [0], partitioned)
    words = out[0].words
    assert [w.text for w in words] == ["hello", "world"]
    assert all(w.romaji is None for w in words)


class _RecordingPreprocess:
    """Stand-in for ctc_forced_aligner.preprocess_text. Returns one
    alignment word per call (wrapped in <star> on each side, mirroring
    the real output shape) and records (unit, language) calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, unit: str, romanize: bool, language: str):
        self.calls.append((unit, language))
        return (["<star>", unit, "<star>"], ["<star>", unit, "<star>"])


pytest.importorskip("cutlet")


def test_preprocess_lines_mixed_threads_surfaces_and_counts():
    """The load-bearing invariant: one morpheme -> one alignment word ->
    one display surface, in order. Japanese tokens carry their original
    surface; English tokens carry None (fall back to aligned text)."""
    pp = _RecordingPreprocess()
    all_tokens, all_text, surfaces, counts, non_empty = _preprocess_lines(
        _lines("君と dance"),
        use_jp_romaji=True,
        treat_kanji_as_japanese=True,
        iso3="jpn",
        preprocess_text=pp,
    )
    assert counts == [3]
    assert non_empty == [0]
    assert surfaces == ["君", "と", None]


def test_preprocess_lines_feeds_romaji_as_english_not_charlevel():
    """Romaji must go through preprocess_text as language='eng' (word-
    level), never 'jpn' (which would char-split 'kimi' into k/i/m/i)."""
    pp = _RecordingPreprocess()
    _preprocess_lines(
        _lines("君"),
        use_jp_romaji=True,
        treat_kanji_as_japanese=True,
        iso3="jpn",
        preprocess_text=pp,
    )
    assert pp.calls == [("kimi", "eng")]


def test_preprocess_lines_falls_back_when_tokenize_raises(monkeypatch):
    """If the Japanese romanizer blows up on a line, that line degrades
    to the char-level path (whole line, language=iso3) instead of
    failing the whole request."""
    import app.pipeline.jp_romaji as jp

    def _boom(*args, **kwargs):
        raise RuntimeError("mecab exploded")

    monkeypatch.setattr(jp, "tokenize", _boom)
    pp = _RecordingPreprocess()
    _, _, surfaces, counts, _ = _preprocess_lines(
        _lines("君と"),
        use_jp_romaji=True,
        treat_kanji_as_japanese=True,
        iso3="jpn",
        preprocess_text=pp,
    )
    assert pp.calls == [("君と", "jpn")]  # whole line, original text, iso3
    assert all(s is None for s in surfaces)
    assert counts == [1]


def test_preprocess_lines_chinese_path_keeps_charlevel_and_no_surfaces():
    """Chinese is left on the existing char-level path: the whole line is
    one preprocess unit at language=iso3, and no surface overrides."""
    pp = _RecordingPreprocess()
    _, _, surfaces, _, _ = _preprocess_lines(
        _lines("这是"),
        use_jp_romaji=False,
        treat_kanji_as_japanese=False,
        iso3="chi",
        preprocess_text=pp,
    )
    assert pp.calls == [("这是", "chi")]
    assert all(s is None for s in surfaces)
