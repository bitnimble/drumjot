"""Whisper-based lyrics alignment.

Loads a vocals stem and produces line + word level time-aligned lyrics
using whisperx (faster-whisper transcription + wav2vec2 forced
alignment for sub-word timestamps).

Models are loaded **lazily** on the first /lyrics/align request rather
than eagerly at startup; the existing separator stack already eats most
of the GPU's wake-up budget, and lyrics alignment is an optional, on-
demand feature. The aligner keeps the loaded models around for
subsequent requests so warm-call latency drops to inference time only.

Memory budget on a 6 GB consumer GPU (e.g. GTX 1660 Super, see
config.py::whisper_compute_type rationale): the separator pipeline
unloads its model between stages, so when whisper inference runs the
GPU has effectively ~5 GB free. `medium` + `int8_float16` peaks at
~700 MB; `large-v3` int8 peaks at ~1.5 GB.
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
    sustained notes read as held."""

    start_sec: float
    end_sec: float
    text: str


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

# Audio sample rate whisperx normalises every input to (see
# `whisperx.load_audio`). Used here to compute the audio's true length
# in seconds and to slice the first 30 s for language detection without
# having to read the original audio's sample rate.
_WHISPER_SAMPLE_RATE = 16000
_LANGUAGE_DETECT_SECONDS = 30
# Trailing time appended to the last line's `end` when we don't have a
# next-line anchor and the audio also doesn't extend much past the
# line's start. wav2vec2 needs a non-zero search window; 5 s comfortably
# covers a long held final note without bleeding into a fade-out tail
# the LRC didn't include.
_LAST_LINE_TAIL_SEC = 5.0

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


class WhisperAligner:
    """Lazy-loaded transcribe + align wrapper.

    Two whisperx components are loaded and held in memory:

      - `transcribe_model`: a faster-whisper (CTranslate2) Whisper model
        sized per `settings.whisper_model`. Produces segment-level
        transcripts. Loaded on first use.
      - `align_models`: per-language wav2vec2 forced aligners, cached by
        detected language. Each language reuses its cached model on
        subsequent calls.

    Thread safety: callers may invoke `realign_text` concurrently, but the
    inference call inside whisperx is single-threaded; we serialise
    requests on `_lock` so a burst doesn't blow up VRAM by trying to load
    two copies of the model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._transcribe_model: Any | None = None
        self._align_models: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._device: str | None = None
        self._compute_type: str | None = None

    def _resolve_device(self) -> str:
        """Pick the device whisperx should run on. `auto` ≡ `cuda` if
        available, else `cpu`. faster-whisper doesn't support MPS today,
        so an `mps` setting silently downgrades to CPU."""
        if self._device is not None:
            return self._device
        configured = settings.device.lower()
        if configured in {"cuda", "cpu"}:
            self._device = configured
        elif configured == "mps":
            log.warning(
                "whisper: device=mps not supported by faster-whisper; "
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
        """CTranslate2 compute type. int8_float16 is invalid on CPU
        (CT2 has no int8 CPU kernel); fall back to plain int8 there."""
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
            "whisper: loading transcribe model %s (device=%s, compute_type=%s)",
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

    def _load_aligner(self, language_code: str) -> tuple[Any, dict[str, Any]]:
        cached = self._align_models.get(language_code)
        if cached is not None:
            return cached
        import whisperx  # type: ignore[import-not-found]

        device = self._resolve_device()
        log.info("whisper: loading aligner for language=%s", language_code)
        model, metadata = whisperx.load_align_model(
            language_code=language_code, device=device
        )
        self._align_models[language_code] = (model, metadata)
        return model, metadata

    def realign_text(
        self,
        audio_path: Path,
        input_lines: list[InputLine],
        language: str | None = None,
    ) -> list[LyricLine]:
        """Forced-align caller-provided lyric text to `audio_path`.

        Skips Whisper transcription entirely: the text is treated as
        ground truth and only the timings are recomputed. The intended
        use case is "I already have an accurate LRC but its timings are
        wrong / I want word-level timings" - especially valuable for
        code-switched lyrics, where Whisper's single-language decode
        would otherwise mangle the foreign segments into katakana-style
        phonetic transcriptions.

        `language` overrides automatic detection; pass it whenever the
        caller knows (e.g. lifted from a lyrics-file metadata tag). When
        omitted, we run a one-off Whisper inference on the first 30 s
        of audio purely to get the language code - cheaper than full
        transcription, still loud enough for the language head.

        Empty-text input lines (LRC instrumental markers) are passed
        through to the output untouched - wav2vec2 has nothing to align
        on them - so the line count and ordering are preserved 1:1.

        On unrecoverable failure (no aligner for the detected language,
        wav2vec2 raised, audio missing words) we degrade to returning
        the input lines unchanged (`words=None`) so the frontend can
        still render the user's text. Only model-load failures
        propagate.
        """
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio not found: {audio_path}")
        if not input_lines:
            return []
        import whisperx  # type: ignore[import-not-found]

        with self._lock:
            device = self._resolve_device()
            audio = whisperx.load_audio(str(audio_path))
            audio_duration_sec = (
                len(audio) / _WHISPER_SAMPLE_RATE if len(audio) else 0.0
            )

            language_code = (
                language
                or settings.whisper_language
                or _detect_language_from_text(input_lines)
                or self._detect_language(audio)
            )
            log.info(
                "whisper: realigning %d caller-provided lines against %s "
                "(language=%s)",
                len(input_lines), audio_path.name, language_code,
            )

            segments, segment_indices = _build_align_segments(
                input_lines, audio_duration_sec
            )
            if not segments:
                # All input lines were empty-text. Nothing to align;
                # echo them back so the response still mirrors the input.
                return [
                    LyricLine(start_sec=line.start_sec, text=line.text, words=None)
                    for line in input_lines
                ]

            try:
                align_model, metadata = self._load_aligner(language_code)
                aligned = whisperx.align(
                    segments,
                    align_model,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                aligned_segments = aligned.get("segments") or []
            except Exception as exc:
                log.warning(
                    "whisper: realign failed (language=%s), returning "
                    "caller lines unchanged: %s",
                    language_code, exc,
                )
                return [
                    LyricLine(start_sec=line.start_sec, text=line.text, words=None)
                    for line in input_lines
                ]

            # Re-stitch: walk the original input order and, for each
            # non-empty line, pop its aligned counterpart off the
            # whisperx output. Empty-text lines pass straight through.
            aligned_by_input_idx: dict[int, Any] = {}
            for aligned_seg, input_idx in zip(
                aligned_segments, segment_indices, strict=False,
            ):
                aligned_by_input_idx[input_idx] = aligned_seg

            out: list[LyricLine] = []
            for idx, line in enumerate(input_lines):
                aligned_seg = aligned_by_input_idx.get(idx)
                if aligned_seg is None:
                    out.append(
                        LyricLine(
                            start_sec=line.start_sec, text=line.text, words=None
                        )
                    )
                    continue
                # Preserve the CALLER'S exact text - whisperx may have
                # normalised punctuation / casing during alignment, and
                # the whole point of the realign path is that the input
                # text is authoritative.
                words = _extract_words(
                    aligned_seg, segment_start=line.start_sec
                )
                refined_start = (
                    words[0].start_sec if words else line.start_sec
                )
                out.append(
                    LyricLine(
                        start_sec=refined_start,
                        text=line.text,
                        words=words if words else None,
                    )
                )
            return out

    def _detect_language(self, audio: Any) -> str:
        """Run a single Whisper pass on the leading 30 s of audio purely
        to harvest the detected language. Falls back to English if the
        detector returns nothing (silent / instrumental clip).
        """
        transcribe = self._load_transcribe()
        slice_len = _LANGUAGE_DETECT_SECONDS * _WHISPER_SAMPLE_RATE
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


def _build_align_segments(
    input_lines: list[InputLine], audio_duration_sec: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Convert {@link InputLine}s into whisperx.align's segment shape.

    Each output segment carries `text` + `start` + `end`. `end` is set
    to the next non-empty line's start (a wav2vec2 search window is
    capped at the next line's beginning so words can't leak forward);
    the last segment's `end` clamps to `audio_duration_sec`, or to
    `start + tail` if the audio is shorter / unknown.

    Returns `(segments, indices)` where `indices[i]` is the position of
    `segments[i]` in the original `input_lines` list, so empty-text
    lines (skipped here) can be merged back in by the caller.
    """
    out_segments: list[dict[str, Any]] = []
    out_indices: list[int] = []
    # Pre-compute the next non-empty line's start_sec for each index so
    # `end` doesn't span a stretch the LRC marked as instrumental.
    next_start: list[float | None] = [None] * len(input_lines)
    seen_after: float | None = None
    for i in range(len(input_lines) - 1, -1, -1):
        next_start[i] = seen_after
        if input_lines[i].text.strip():
            seen_after = input_lines[i].start_sec

    for idx, line in enumerate(input_lines):
        text = line.text.strip()
        if not text:
            continue
        if next_start[idx] is not None:
            end_sec = next_start[idx]
        elif audio_duration_sec > line.start_sec:
            end_sec = audio_duration_sec
        else:
            end_sec = line.start_sec + _LAST_LINE_TAIL_SEC
        # whisperx requires end > start; guard against malformed LRCs
        # where two lines share a timestamp or arrive out of order.
        if end_sec <= line.start_sec:
            end_sec = line.start_sec + _LAST_LINE_TAIL_SEC
        out_segments.append(
            {"text": text, "start": line.start_sec, "end": end_sec}
        )
        out_indices.append(idx)
    return out_segments, out_indices


def _extract_words(segment: Any, *, segment_start: float) -> list[LyricWord]:
    """Pull the per-word entries out of a whisperx segment dict,
    stripping empty tokens and filling in missing timings.

    `start` falls back to `segment_start` when whisperx couldn't align a
    token (rare, very short tokens). `end` walks a fallback chain so the
    frontend always sees a numeric width:

        word's own `end` -> next surviving word's `start`
                         -> segment's declared `end`
                         -> `start + 0.05` (last-ditch epsilon).

    The two-pass shape (collect first, backfill ends second) is needed
    because the "next word's start" fallback can only be resolved once
    we know which raw entries survived the empty-token filter."""
    raw_words = segment.get("words") or []
    seg_end_raw = segment.get("end")
    seg_end = float(seg_end_raw) if seg_end_raw is not None else None

    # Pass 1: collect surviving (start, end_or_none, text) triples,
    # carrying `end` through as-is so the backfill in pass 2 can see
    # which entries actually need filling.
    raw: list[tuple[float, float | None, str]] = []
    for w in raw_words:
        word_text = (w.get("word") or "").strip()
        if not word_text:
            continue
        word_start = w.get("start")
        start_sec = float(word_start) if word_start is not None else segment_start
        word_end = w.get("end")
        end_sec = float(word_end) if word_end is not None else None
        raw.append((start_sec, end_sec, word_text))

    # Pass 2: backfill missing ends in left-to-right order so each gap
    # can borrow the next surviving entry's start. The last entry's
    # fallback walks past the next-word step into segment-end / epsilon.
    out: list[LyricWord] = []
    for i, (start_sec, end_sec, text) in enumerate(raw):
        if end_sec is None:
            next_start: float | None = None
            for j in range(i + 1, len(raw)):
                next_start = raw[j][0]
                break
            if next_start is not None:
                end_sec = next_start
            elif seg_end is not None:
                end_sec = seg_end
            else:
                end_sec = start_sec + 0.05
        # Guard pathological cases (next word starts before current, or
        # equal): clamp so the frontend's cell width stays non-negative.
        if end_sec <= start_sec:
            end_sec = start_sec + 0.05
        out.append(LyricWord(start_sec=start_sec, end_sec=end_sec, text=text))
    return out


# Process-wide singleton; the heavy `WhisperModel` lives here so the
# second request reuses the warm model. Imported by main.py and
# initialised on first use; never auto-loaded at startup.
_aligner_singleton: WhisperAligner | None = None
_singleton_lock = threading.Lock()


def get_aligner() -> WhisperAligner:
    """Return the process-wide {@link WhisperAligner}, constructing on
    first call. The aligner itself defers model loading until
    `realign_text` is invoked, so this is cheap."""
    global _aligner_singleton
    with _singleton_lock:
        if _aligner_singleton is None:
            _aligner_singleton = WhisperAligner()
        return _aligner_singleton


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
            entry["words"] = [
                {"startSec": w.start_sec, "endSec": w.end_sec, "text": w.text}
                for w in line.words
            ]
        out.append(entry)
    return out
