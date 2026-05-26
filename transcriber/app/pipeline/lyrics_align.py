"""CTC forced-alignment for lyrics.

Loads a vocals stem and produces line + word level time-aligned lyrics
using `ctc-forced-aligner` (MahmoudAshraf97/ctc-forced-aligner), which
runs the MMS-300m multilingual CTC model over the FULL audio in one
pass (with internal chunking + posterior stitching) and a single
global Viterbi alignment of all the caller's lyric text against the
resulting posteriors. The benefit over the previous whisperx per-line
approach is that wav2vec2 picks each word's actual audio position
instead of being constrained to a `[line.start_sec, next_line.start_sec]`
search window - which broke down for plain-text inputs (synthesized
timestamps) and for LRCLIB matches against a different cut.

Whisperx is still pulled in for the language-detect fallback (a
faster-whisper inference over the first 30 s of audio when our script-
based text detector can't decide); we don't use whisperx's wav2vec2
aligner anymore.

Models are loaded **lazily** on the first /lyrics/align request rather
than eagerly at startup; the existing separator stack already eats most
of the GPU's wake-up budget, and lyrics alignment is an optional, on-
demand feature. The aligner keeps the loaded models around for
subsequent requests so warm-call latency drops to inference time only.

Memory budget on a 6 GB consumer GPU (e.g. GTX 1660 Super, see
config.py::whisper_compute_type rationale): the separator pipeline
unloads its model between stages, so when alignment runs the GPU has
effectively ~5 GB free. MMS-300m at fp16 peaks at ~600 MB; faster-
whisper `medium` int8_float16 (for language detection) peaks at
~700 MB and unloads after the detection pass.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class LyricWord:
    """One word within a {@link LyricLine}'s `words` array.

    `end_sec` is wav2vec2's phoneme-release time for the word - the
    moment after which the next aligned word can begin. Frontend uses
    `start_sec`..`end_sec` as the word's visual cell on the bars row so
    sustained notes read as held.

    The `raw_*` and `end_fallback` fields preserve the model's
    pre-substitution view of the world for UI debug tooltips, so the
    user can see "what wav2vec2 actually said" alongside "what we use
    for layout". Distinct fields rather than diff-against-final so the
    consumer doesn't have to know our substitution rules to reconstruct
    the model output."""

    start_sec: float
    end_sec: float
    text: str
    # Raw MMS-300m outputs before any clamping. With the ctc-forced-
    # aligner path these mirror the final `start_sec` / `end_sec` in
    # the common case because the aligner always emits both edges; the
    # fields are still optional on the wire because earlier whisperx
    # output could omit them (and to leave room for future aligners
    # that don't emit both). Kept so the UI debug tooltip can show
    # "what the model said" vs "what we render" without the consumer
    # having to know our substitution rules.
    raw_start_sec: float | None = None
    raw_end_sec: float | None = None
    # Marker for when our code adjusted `end_sec` away from what the
    # model emitted. None means the rendered value matches the raw
    # value. With ctc-forced-aligner the only path that fires today is
    # `inverted-clamp`; the `next-start` / `segment-end` / `epsilon`
    # values are reserved for older whisperx-style outputs (the wire
    # vocabulary stays stable so the frontend doesn't have to know
    # which aligner produced the data):
    #   - "inverted-clamp": model emitted end <= start; bumped to
    #                       start + 0.05s
    #   - "next-start"    : (legacy) end borrowed from next word's start
    #   - "segment-end"   : (legacy) end clamped to segment boundary
    #   - "epsilon"       : (legacy) last-ditch start + 0.05s
    end_fallback: str | None = None


@dataclass
class LyricLine:
    """One line of synced lyrics. `words` is the sub-line breakdown the
    wav2vec2 aligner produced; present whenever alignment succeeded;
    a transcription-only fallback (no alignment model for the detected
    language) returns lines with `words=None`."""

    start_sec: float
    text: str
    words: list[LyricWord] | None


@dataclass
class InputLine:
    """Caller-provided lyric line for the forced-alignment path. Mirrors
    the LRC subset of {@link LyricLine}: line text + the caller's best
    guess at when the line begins. Used only as a starting estimate;
    `realign_text` recomputes both line and word timings from the audio
    via wav2vec2 forced alignment."""

    start_sec: float
    text: str

# Audio sample rate the CTC aligner + whisperx language detector both
# normalise every input to. Used here to slice the first 30 s of audio
# for language detection without having to read the original audio's
# sample rate.
_AUDIO_SAMPLE_RATE = 16000
_LANGUAGE_DETECT_SECONDS = 30
# Batch size handed to ctc_forced_aligner.generate_emissions: how many
# chunks of audio are pushed through the model in parallel. Each chunk
# is ~30 s, so batch_size=4 keeps peak VRAM bounded while still being
# faster than serial. Tune up if VRAM headroom exists.
_CTC_BATCH_SIZE = 4

# Simplified-Chinese-only characters whose Japanese equivalent uses a
# visibly different glyph (e.g. simplified 爱 vs Japanese / traditional
# 愛). Presence of any of these in otherwise-ambiguous CJK-only text
# routes language detection to `zh`; otherwise we default to `ja`
# because Japanese songs dominate the typical drumjot library and a
# false-positive `zh` on a J-pop lyric shatters every kanji into its
# own alignment unit (the wav2vec2 ZH aligner tokenises by character
# the same way JA does, but pulls from a Chinese phoneme map - Whisper
# loads the wrong model and the timings come out garbage).
#
# Conservative on purpose: this set MUST stay entries that are
# vanishingly rare in Japanese text. Characters that exist in both
# scripts (国, 学, 来, 会, 着, 没, etc.) are EXCLUDED on purpose - they
# don't disambiguate. Curate additions; don't paste a "simplified
# Chinese top-N" list wholesale.
_SIMPLIFIED_CHINESE_MARKERS = frozenset(
    "爱们这时长个听见说话让给风马鸟鱼谁谢发实还对么"
)


class LyricsAligner:
    """Lazy-loaded forced-alignment wrapper.

    Two model groups are loaded and held in memory:

      - `align_model` + `align_tokenizer`: MMS-300m multilingual CTC
        aligner via `ctc-forced-aligner`. ONE model handles every
        language; the per-language wav2vec2 pinning that whisperx
        required is gone.
      - `transcribe_model`: faster-whisper (CTranslate2) Whisper sized
        per `settings.whisper_model`. Used ONLY for the language-detect
        fallback when our script-based text detector returns None
        (audio-only LRCs etc.). Loaded on first detect call, kept warm.

    Thread safety: callers may invoke `realign_text` concurrently, but
    GPU model inference is single-threaded; we serialise requests on
    `_lock` so a burst doesn't blow up VRAM by trying to load two
    copies of the same model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._transcribe_model: Any | None = None
        self._align_model: Any | None = None
        self._align_tokenizer: Any | None = None
        self._device: str | None = None
        self._compute_type: str | None = None

    def _resolve_device(self) -> str:
        """Pick the device the alignment + detection models run on.
        `auto` ≡ `cuda` if available, else `cpu`. faster-whisper
        doesn't support MPS today, so an `mps` setting silently
        downgrades to CPU."""
        if self._device is not None:
            return self._device
        configured = settings.device.lower()
        if configured in {"cuda", "cpu"}:
            self._device = configured
        elif configured == "mps":
            log.warning(
                "lyrics: device=mps not supported by faster-whisper; "
                "falling back to CPU"
            )
            self._device = "cpu"
        else:  # auto
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        return self._device

    def _resolve_compute_type(self, device: str) -> str:
        """CTranslate2 compute type for the language-detect Whisper
        model. int8_float16 is invalid on CPU (CT2 has no int8 CPU
        kernel); fall back to plain int8 there."""
        if self._compute_type is not None:
            return self._compute_type
        if device == "cpu":
            self._compute_type = "int8"
        else:
            self._compute_type = settings.whisper_compute_type
        return self._compute_type

    def _load_transcribe(self) -> Any:
        if self._transcribe_model is not None:
            return self._transcribe_model
        # Imported lazily so `import app.pipeline.lyrics_align` doesn't
        # pull in 300+ MB of CT2 / pyannote when the endpoint isn't used.
        import whisperx  # type: ignore[import-not-found]

        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)
        download_root = settings.models_dir / "whisperx"
        download_root.mkdir(parents=True, exist_ok=True)
        log.info(
            "lyrics: loading language-detect model %s "
            "(device=%s, compute_type=%s)",
            settings.whisper_model, device, compute_type,
        )
        model = whisperx.load_model(
            settings.whisper_model,
            device,
            compute_type=compute_type,
            download_root=str(download_root),
        )
        self._transcribe_model = model
        return model

    def _load_ctc_aligner(self) -> tuple[Any, Any]:
        """Load (or return cached) the MMS-300m forced-alignment model +
        its tokenizer via `ctc_forced_aligner.load_alignment_model`.
        Weights are pulled from HuggingFace
        (`MahmoudAshraf/mms-300m-1130-forced-aligner`, ~1.2 GB) on first
        use and cached under `HF_HOME` (the Dockerfile points that at
        the `models_dir` volume, so the download survives container
        restarts).

        Picks fp16 on CUDA and fp32 on CPU - fp16 halves activation
        memory and roughly doubles throughput on consumer GPUs with no
        observable hit to word-alignment accuracy on our inputs; on CPU
        fp16 has no kernel coverage so we'd silently fall back to fp32
        anyway, and the explicit dtype here keeps the model load
        deterministic. The returned model has `.dtype` / `.device`
        attributes that `load_audio` reads to materialise the waveform
        on the same device + precision as the model."""
        if self._align_model is not None:
            return self._align_model, self._align_tokenizer
        # Lazy import so a process that never touches /lyrics/align
        # doesn't pull in `transformers` + the alignment package's
        # ~1.2 GB model on boot.
        import torch
        from ctc_forced_aligner import (  # type: ignore[import-not-found]
            load_alignment_model,
        )

        device = self._resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        log.info(
            "lyrics: loading CTC aligner (device=%s, dtype=%s)",
            device, dtype,
        )
        model, tokenizer = load_alignment_model(device, dtype=dtype)
        self._align_model = model
        self._align_tokenizer = tokenizer
        return model, tokenizer

    def realign_text(
        self,
        audio_path: Path,
        input_lines: list[InputLine],
        language: str | None = None,
    ) -> list[LyricLine]:
        """Forced-align caller-provided lyric text to `audio_path`.

        Treats the caller's text as ground truth and recomputes
        timings from scratch via CTC forced alignment. The full audio
        is aligned in ONE call - no per-line `[start, next_start]`
        windows - so each word lands at the audio position MMS picked,
        not clamped to the caller's rough timestamps. This is the key
        win over the previous whisperx-per-segment approach, which
        broke down hard for plain-text inputs (where caller timestamps
        are evenly synthesised) and for LRCLIB matches against a
        different cut of the same song.

        `language` overrides automatic detection; pass it whenever the
        caller knows (e.g. lifted from a lyrics-file metadata tag). When
        omitted, we try script-based text detection first (cheap,
        deterministic) and fall back to a one-off faster-whisper pass
        over the first 30 s of audio.

        Empty-text input lines (LRC instrumental markers) are passed
        through to the output untouched - the aligner has nothing to
        place on them - so the line count and ordering are preserved
        1:1.

        On unrecoverable failure (aligner raised, word-count mismatch
        we can't partition cleanly) we degrade to returning the input
        lines unchanged (`words=None`) so the frontend can still
        render the user's text. Only model-load failures propagate.
        """
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio not found: {audio_path}")
        if not input_lines:
            return []

        # Lazy imports inside the lock so the per-process model singletons
        # initialise once even under concurrent calls.
        from ctc_forced_aligner import (  # type: ignore[import-not-found]
            generate_emissions,
            get_alignments,
            get_spans,
            load_audio,
            postprocess_results,
            preprocess_text,
        )

        with self._lock:
            model, tokenizer = self._load_ctc_aligner()
            # load_audio materialises the waveform on the same device +
            # dtype as the model so generate_emissions can run without
            # an extra copy / cast inside its inner loop.
            audio_waveform = load_audio(str(audio_path), model.dtype, model.device)

            language_code = (
                language
                or settings.whisper_language
                or _detect_language_from_text(input_lines)
                or self._detect_language_via_audio(audio_path)
            )
            iso3 = _iso1_to_iso3(language_code)

            # Per-line preprocess so we know exactly how many post-
            # process_results entries each input line will receive. We
            # CAN'T just call text.split() and use that count because:
            #   - For jpn/chi the package switches to char-level
            #     tokenisation; a 5-char line returns 5 entries, not 1.
            #   - text_normalize strips punctuation, brackets with
            #     digits, etc.; the count can drift from a naive split.
            # Doing the per-line call upfront pins the boundary count
            # to whatever the aligner is about to do, so the partition
            # downstream is always exact.
            all_tokens: list[str] = []
            all_text: list[str] = []
            line_word_counts: list[int] = []
            non_empty_indices: list[int] = []
            for idx, line in enumerate(input_lines):
                t = line.text.strip()
                if not t:
                    continue
                try:
                    tokens_starred, text_starred = preprocess_text(
                        t, romanize=True, language=iso3,
                    )
                except Exception as exc:
                    log.warning(
                        "lyrics: preprocess_text failed for line %d (%r): %s",
                        idx, t, exc,
                    )
                    continue
                real = sum(1 for s in text_starred if s != "<star>")
                if real == 0:
                    continue
                all_tokens.extend(tokens_starred)
                all_text.extend(text_starred)
                line_word_counts.append(real)
                non_empty_indices.append(idx)

            if not all_tokens:
                # Every line was empty or unprocessable; echo back so
                # the response still mirrors the input line count.
                return [
                    LyricLine(start_sec=line.start_sec, text=line.text, words=None)
                    for line in input_lines
                ]

            log.info(
                "lyrics: aligning %d non-empty lines against %s "
                "(language=%s/%s, total_tokens=%d)",
                len(non_empty_indices), audio_path.name,
                language_code, iso3, sum(line_word_counts),
            )

            try:
                emissions, stride = generate_emissions(
                    model, audio_waveform, batch_size=_CTC_BATCH_SIZE,
                )
                segments, scores, blank_token = get_alignments(
                    emissions, all_tokens, tokenizer,
                )
                spans = get_spans(all_tokens, segments, blank_token)
                word_timestamps = postprocess_results(
                    all_text, spans, stride, scores,
                )
            except Exception as exc:
                log.warning(
                    "lyrics: CTC forced alignment failed (language=%s), "
                    "returning caller lines unchanged: %s",
                    language_code, exc,
                )
                return [
                    LyricLine(start_sec=line.start_sec, text=line.text, words=None)
                    for line in input_lines
                ]

            partitioned = _partition_words_by_line(
                word_timestamps, line_word_counts,
            )
            if partitioned is None:
                # Aligner emitted a token count we couldn't map to lines.
                # Should be impossible given the per-line preprocess
                # above, but the partition guard stays so a future API
                # drift fails loud instead of misaligning words to the
                # wrong lines.
                log.warning(
                    "lyrics: aligner returned %d words; expected %d. "
                    "Returning lines without word-level timings.",
                    len(word_timestamps), sum(line_word_counts),
                )
                return [
                    LyricLine(start_sec=line.start_sec, text=line.text, words=None)
                    for line in input_lines
                ]

            return _stitch_lines(input_lines, non_empty_indices, partitioned)

    def _detect_language_via_audio(self, audio_path: Path) -> str:
        """Fallback language detection: run a single Whisper pass on
        the leading 30 s of audio purely to harvest the detected
        language. Returns `'en'` when the detector emits nothing
        (silent / instrumental clip)."""
        import whisperx  # type: ignore[import-not-found]

        transcribe = self._load_transcribe()
        audio = whisperx.load_audio(str(audio_path))
        slice_len = _LANGUAGE_DETECT_SECONDS * _AUDIO_SAMPLE_RATE
        detect_slice = audio[:slice_len] if len(audio) > slice_len else audio
        result = transcribe.transcribe(
            detect_slice, batch_size=1, language=None
        )
        return result.get("language") or "en"


def _detect_language_from_text(input_lines: list[InputLine]) -> str | None:
    """Cheap, deterministic language detection from caller-provided
    lyric text. Used by `realign_text` so we don't have to trust
    Whisper's audio-based detector - which mis-classifies the first
    30 s of a vocals stem if the intro is silent / instrumental /
    background-vocal-only, and which on noise has been observed to
    return a `LANGUAGES_WITHOUT_SPACES` code even for English audio.
    That mis-classification then shatters every word into a per-letter
    "word" because whisperx tokenises no-space languages character-by-
    character (each char becomes its own alignment unit and its own
    output entry).

    Returns an ISO-639-1 code, or `None` when the text doesn't contain
    enough alphabetic characters to decide; the caller then falls back
    to the audio-based detector. Heuristic is script-based with a
    Japanese bias on ambiguous CJK (see `_SIMPLIFIED_CHINESE_MARKERS`
    for the override path). For mixed bilingual lyrics (J-pop with
    English chorus, etc.) we pick the dominant non-Latin script, since
    the Latin chunks aligned against e.g. the Japanese aligner will
    fragment anyway - the correct long-term fix is per-line language
    detection and per-line aligner load, which we don't do today.
    """
    text = "".join(line.text for line in input_lines)
    if not text.strip():
        return None
    # Two-pass: collect script presence first, then prioritise. Single
    # pass with early-return mis-tagged kanji-leading Japanese as
    # Chinese, because the kana (the actual ja-distinguishing signal)
    # appears later in the string than the leading kanji.
    has_kana = False
    has_hangul = False
    has_thai = False
    has_cjk = False
    has_latin = False
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            has_kana = True
        elif 0xAC00 <= cp <= 0xD7AF:
            has_hangul = True
        elif 0x0E00 <= cp <= 0x0E7F:
            has_thai = True
        elif 0x4E00 <= cp <= 0x9FFF:
            has_cjk = True
        elif ch.isalpha() and cp < 0x250:
            has_latin = True
    # Priority order:
    #   1. kana       -> ja (any kana is a definitive Japanese signal)
    #   2. hangul     -> ko
    #   3. thai       -> th
    #   4. simplified-Chinese-only glyph -> zh
    #   5. CJK without any of the above -> ja by default
    #
    # The kanji-only case (#5) is genuinely ambiguous - 漢字 is
    # identical glyph-for-glyph in Japanese and Traditional Chinese -
    # so we lean on the user's stated library bias (mostly Japanese
    # music) and default to ja. Mis-routing a Traditional-Chinese lyric
    # is the trade-off; the caller can pin `language="zh"` to override.
    if has_kana:
        return "ja"
    if has_hangul:
        return "ko"
    if has_thai:
        return "th"
    if has_cjk:
        if any(ch in _SIMPLIFIED_CHINESE_MARKERS for ch in text):
            return "zh"
        return "ja"
    # No non-Latin script seen; if there's any Latin letter, treat as
    # English. The wav2vec2 EN aligner copes well with Latin-script
    # romance / germanic languages (mis-tagging French or Spanish as
    # English still yields WORD-level timings, vs the character-soup
    # failure mode of routing space-separated text through a no-space
    # aligner). For genuine non-English Latin text the caller can pass
    # `language` explicitly.
    if has_latin:
        return "en"
    return None


# --------------------------------------------------------------------
# Helpers for the ctc-forced-aligner pipeline. Kept module-private so
# tests can drive each transformation in isolation.
# --------------------------------------------------------------------

# ISO-639-1 (Whisper / our text detector) -> ISO-639-3 (the language
# code expected by ctc_forced_aligner.preprocess_text, which feeds MMS
# romanization). Covers the codes _detect_language_from_text emits
# (en/ja/ko/zh/th) plus a few common Latin-script tags so callers can
# pin specific Romance / Germanic languages through `settings.
# whisper_language` or the request's `language` field. Anything else
# falls back to `eng` because MMS handles unspecified Latin text fine
# but barfs on an unknown ISO-639-3 code.
_ISO639_1_TO_3 = {
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
    # `chi` (not `cmn`): ctc-forced-aligner's `preprocess_text` checks
    # `language in ["jpn", "chi"]` to switch to char-level tokenization,
    # which is the only thing that gives sensible per-character cells
    # for languages without whitespace word boundaries.
    "zh": "chi",
    "th": "tha",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "sv": "swe",
    "no": "nor",
    "da": "dan",
    "fi": "fin",
    "pl": "pol",
    "ru": "rus",
    "vi": "vie",
    "id": "ind",
    "ms": "msa",
    "tr": "tur",
    "ar": "ara",
}


def _iso1_to_iso3(code: str) -> str:
    """Map an ISO-639-1 code (Whisper's output / our text detector's
    output) to ISO-639-3 (what ctc-forced-aligner's `preprocess_text`
    wants). Unknown codes degrade to `eng` rather than raising, so a
    misdetected language still produces output (just with English
    romanization, which is wrong but recoverable)."""
    return _ISO639_1_TO_3.get(code.lower(), "eng")


def _partition_words_by_line(
    word_timestamps: list[dict[str, Any]],
    line_word_counts: list[int],
) -> list[list[dict[str, Any]]] | None:
    """Slice the aligner's flat word list back into per-line groups.

    Returns one list of word dicts per entry in `line_word_counts`, or
    `None` when the total counts don't match. A mismatch is a
    deliberate hard-fail signal: it means our `text.split()` view of
    the input disagrees with whatever the aligner's tokeniser produced
    (typically a non-Latin script where romanisation introduced extra
    or merged tokens). Rather than guess at boundaries we let the
    caller degrade to line-level output - which is what the realign
    path's catch-all already does on alignment exceptions.
    """
    expected = sum(line_word_counts)
    if len(word_timestamps) != expected:
        return None
    out: list[list[dict[str, Any]]] = []
    cursor = 0
    for count in line_word_counts:
        out.append(word_timestamps[cursor : cursor + count])
        cursor += count
    return out


def _stitch_lines(
    input_lines: list[InputLine],
    non_empty_indices: list[int],
    partitioned: list[list[dict[str, Any]]],
) -> list[LyricLine]:
    """Build the final {@link LyricLine} list, slotting word-level
    timings back into the non-empty positions and passing empty-text
    lines through with `words=None`.

    Each word dict in `partitioned` is the shape ctc-forced-aligner's
    `postprocess_results` emits: `{"text": str, "start": float,
    "end": float, "score": float}`. The frontend's LyricWord type
    additionally carries `raw_*` debug fields; ctc-forced-aligner
    never substitutes start/end so those mirror the final values and
    `end_fallback` stays None.
    """
    by_input_idx = dict(zip(non_empty_indices, partitioned, strict=True))
    out: list[LyricLine] = []
    for idx, line in enumerate(input_lines):
        words = by_input_idx.get(idx)
        if not words:
            out.append(LyricLine(start_sec=line.start_sec, text=line.text, words=None))
            continue
        lyric_words: list[LyricWord] = []
        for w in words:
            start_sec = float(w.get("start", 0.0))
            raw_end = float(w.get("end", start_sec + 0.05))
            text = str(w.get("text", "")).strip()
            if not text:
                continue
            # CTC alignment is occasionally degenerate on syllables it
            # can't place (a held vowel that wav2vec2 absorbs into the
            # neighbouring word, etc.); clamp so the cell never inverts
            # downstream. Preserve the model's raw end in `raw_end_sec`
            # so the UI tooltip can show what the aligner emitted vs
            # what we use. Marker vocabulary stays stable across
            # backends so the frontend doesn't have to know whether
            # ctc-forced-aligner or whisperx produced the data.
            if raw_end <= start_sec:
                end_sec = start_sec + 0.05
                fallback: str | None = "inverted-clamp"
            else:
                end_sec = raw_end
                fallback = None
            lyric_words.append(
                LyricWord(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=text,
                    raw_start_sec=start_sec,
                    raw_end_sec=raw_end,
                    end_fallback=fallback,
                )
            )
        refined_start = (
            lyric_words[0].start_sec if lyric_words else line.start_sec
        )
        out.append(
            LyricLine(
                start_sec=refined_start,
                text=line.text,
                words=lyric_words if lyric_words else None,
            )
        )
    return out


# Process-wide singleton; the heavy MMS-300m aligner lives here so the
# second request reuses the warm model. Imported by main.py and
# initialised on first use; never auto-loaded at startup.
_aligner_singleton: LyricsAligner | None = None
_singleton_lock = threading.Lock()


def get_aligner() -> LyricsAligner:
    """Return the process-wide {@link LyricsAligner}, constructing on
    first call. The aligner itself defers model loading until
    `realign_text` is invoked, so this is cheap."""
    global _aligner_singleton
    with _singleton_lock:
        if _aligner_singleton is None:
            _aligner_singleton = LyricsAligner()
        return _aligner_singleton


def _word_to_json(w: LyricWord) -> dict[str, Any]:
    """Per-word wire shape. Required fields (`startSec`, `endSec`,
    `text`) always present; debug fields (`rawStartSec`, `rawEndSec`,
    `endFallback`) only when set, so the response payload stays small
    on the common case where the model emitted complete timings."""
    entry: dict[str, Any] = {
        "startSec": w.start_sec,
        "endSec": w.end_sec,
        "text": w.text,
    }
    if w.raw_start_sec is not None:
        entry["rawStartSec"] = w.raw_start_sec
    if w.raw_end_sec is not None:
        entry["rawEndSec"] = w.raw_end_sec
    if w.end_fallback is not None:
        entry["endFallback"] = w.end_fallback
    return entry


def lines_to_json(lines: list[LyricLine]) -> list[dict[str, Any]]:
    """Serialize `LyricLine`s into the frontend's wire shape.

    Mirrors `src/lyrics/lrc.ts::LyricLine` exactly: camelCase keys,
    `words` omitted (rather than `null`) when alignment didn't succeed.
    The endpoint wraps this into `{lines: [...]}`.
    """
    out: list[dict[str, Any]] = []
    for line in lines:
        entry: dict[str, Any] = {"startSec": line.start_sec, "text": line.text}
        if line.words is not None:
            entry["words"] = [_word_to_json(w) for w in line.words]
        out.append(entry)
    return out
