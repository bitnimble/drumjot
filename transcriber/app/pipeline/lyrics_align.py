"""CTC forced-alignment for lyrics.

Loads a vocals stem and produces line + word level time-aligned lyrics
using `ctc-forced-aligner` (MahmoudAshraf97/ctc-forced-aligner). The
package runs a wav2vec2-family CTC model over the FULL audio in one
pass (with internal chunking + posterior stitching) and a single
global Viterbi alignment of all the caller's lyric text against the
resulting posteriors. wav2vec2 picks each word's actual audio position
across the whole song, instead of being constrained to a
`[line.start_sec, next_line.start_sec]` search window - which breaks
down for plain-text inputs (synthesized timestamps) and for LRCLIB
matches against a different cut.

The CTC checkpoint is dispatched per-language by `_pick_alignment_model`:

  - English uses `facebook/wav2vec2-large-robust-ft-libri-960h`
    (Apache 2.0, ~317M, multi-domain robust pretraining + LS-960
    fine-tune). Tighter posteriors on English vocals than the MMS
    multilingual head because the param budget isn't shared with 1100+
    other languages, and commercial-licence-clean.
  - Everything else falls back to MMS-300m via the package's default
    (`MahmoudAshraf/mms-300m-1130-forced-aligner`, CC-BY-NC 4.0). The
    CC-BY-NC licence is a known commercial blocker for non-English
    songs and is tracked as a follow-up; OWSM-CTC v4 1B (CC-BY-4.0)
    is the current preferred replacement candidate for ja/ko/zh.

Both checkpoints share the same `<star>` wildcard mechanism (appended
at runtime by `generate_emissions`, model-agnostic) and the same
`preprocess_text(romanize=True, language=iso3)` output. Vocabs are
compatible: MMS uses uroman-romanized Latin chars and
wav2vec2-large-robust uses lowercase Latin + space + apostrophe.

Language is resolved entirely from the caller's lyric text by
`_detect_language_from_text` (script ranges + distinctive-letter
heuristics), so there is no audio-based language detector: every
script with letters routes to the MMS + uroman path, and only
genuinely letter-free text falls back to a plain `en` default.

Models are loaded **lazily** on the first /lyrics/align request rather
than eagerly at startup; the existing separator stack already eats most
of the GPU's wake-up budget, and lyrics alignment is an optional, on-
demand feature. The aligner keeps the loaded models around for
subsequent requests so warm-call latency drops to inference time only.

Memory budget on a 6 GB consumer GPU (e.g. GTX 1660 Super): the
separator pipeline unloads its model between stages, so when alignment
runs the GPU has effectively ~5 GB free. MMS-300m at fp16 peaks at
~600 MB and the English wav2vec2-large-robust head at ~650 MB, both
comfortably within that budget.
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
    # fields stay optional on the wire to leave room for future
    # aligners that may not emit both. Kept so the UI debug tooltip can
    # show "what the model said" vs "what we render" without the
    # consumer having to know our substitution rules.
    raw_start_sec: float | None = None
    raw_end_sec: float | None = None
    # Marker for when our code adjusted `end_sec` away from what the
    # model emitted. None means the rendered value matches the raw
    # value. With ctc-forced-aligner the only path that fires today is
    # `inverted-clamp`; the `next-start` / `segment-end` / `epsilon`
    # values are reserved for future aligners whose outputs may need
    # similar fix-ups (the wire vocabulary stays stable so the
    # frontend doesn't have to know which aligner produced the data):
    #   - "inverted-clamp": model emitted end <= start; bumped to
    #                       start + 0.05s
    #   - "next-start"    : (legacy) end borrowed from next word's start
    #   - "segment-end"   : (legacy) end clamped to segment boundary
    #   - "epsilon"       : (legacy) last-ditch start + 0.05s
    end_fallback: str | None = None
    # The Latin romaji actually fed to the aligner, present only when
    # `text` is a non-Latin display surface that differs from what was
    # aligned (i.e. Japanese tokens romanized via `jp_romaji`). Lets the
    # UI debug tooltip show "displayed 君 / aligned kimi". None for
    # English / Chinese words, where `text` already is the aligned form.
    romaji: str | None = None


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

# Audio sample rate the CTC aligner normalises every input to. Used to
# convert frame counts to seconds in the audio-stats log without having
# to read the original audio's sample rate.
_AUDIO_SAMPLE_RATE = 16000
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

# The CTC aligner HF ids are build settings (config.py "Model asset sources"),
# read at call time (`settings.lyrics_align_model_english` / `..._default`) so a
# build/env override applies. English (Apache-2.0 wav2vec2-large-robust) handles
# any request detected as English; every other language uses the multilingual MMS
# aligner (adapters pre-merged, romanized).


def _lyrics_onnx_enabled() -> bool:
    """Default ON: run the CTC acoustic model through onnxruntime (torch-free
    inference). Opt out with DRUMJOT_LYRICS_ONNX in {0,false,no,off,torch}
    (mirrors DRUMJOT_SEP_ONNX / DRUMJOT_ONSET_ONNX / DRUMJOT_BEAT_ONNX)."""
    import os

    return os.environ.get("DRUMJOT_LYRICS_ONNX", "1").strip().lower() not in (
        "0", "false", "no", "off", "torch",
    )


def _onnx_providers():
    """onnxruntime providers from settings.device (no torch import): CPU-pinned
    when CPU/MPS is forced, else the available set (+ CPU fallback in the loader)."""
    dev = (settings.device or "auto").lower()
    return ["CPUExecutionProvider"] if dev in ("cpu", "mps") else None


def _is_torch_tensor(x: Any) -> bool:
    """True for a torch tensor, False for a numpy array (duck-typed, no import).
    Used to skip the torch-only emission diagnostics on the numpy/ONNX path."""
    return hasattr(x, "is_cpu")


def _all_finite(emissions: Any) -> bool:
    if _is_torch_tensor(emissions):
        import torch

        return bool(torch.isfinite(emissions).all())
    import numpy as np

    return bool(np.isfinite(emissions).all())


class LyricsAligner:
    """Lazy-loaded forced-alignment wrapper.

    The `align_model` + `align_tokenizer` pair (MMS-300m multilingual
    CTC aligner via `ctc-forced-aligner`, plus the English-specialised
    wav2vec2 head) is loaded and held in memory. ONE multilingual model
    handles every language with no per-language wav2vec2 pinning, and
    language is resolved from the caller's text alone (no audio-based
    detector).

    Thread safety: callers may invoke `realign_text` concurrently, but
    GPU model inference is single-threaded; we serialise requests on
    `_lock` so a burst doesn't blow up VRAM by trying to load two
    copies of the same model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Lazy-loaded CTC aligners keyed by HuggingFace model path
        # (`None` selects the package's built-in default = MMS-300m).
        # Each entry is built on first request that routes to it; both
        # English (wav2vec2-large-robust) and the multilingual MMS-300m
        # can be resident simultaneously, ~1-1.2 GB each at fp16 / fp32
        # respectively, well within the alignment-stage VRAM budget.
        self._align_models: dict[str | None, tuple[Any, Any]] = {}
        # Torch-free ONNX aligners (OnnxCtcAligner), keyed the same way.
        self._onnx_aligners: dict[str | None, Any] = {}
        self._device: str | None = None

    def _resolve_device(self) -> str:
        """Pick the device the CTC alignment models run on. `auto` ≡
        `cuda` if available, else `cpu`. MPS isn't exercised for the
        aligner today, so an `mps` setting silently downgrades to CPU."""
        if self._device is not None:
            return self._device
        configured = settings.device.lower()
        if configured in {"cuda", "cpu"}:
            self._device = configured
        elif configured == "mps":
            log.warning(
                "lyrics: device=mps not supported; falling back to CPU"
            )
            self._device = "cpu"
        else:  # auto
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        return self._device

    def _load_ctc_aligner(
        self, model_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Load (or return cached) a forced-alignment model + tokenizer.

        `model_path=None` -> the package's default
        `MahmoudAshraf/mms-300m-1130-forced-aligner` (CC-BY-NC 4.0,
        multilingual, ~1.2 GB at fp32). Any other value is a
        HuggingFace model id passed straight through to
        `load_alignment_model`; the package accepts any wav2vec2-family
        CTC checkpoint there. Used today by `_pick_alignment_model` to
        route English requests to `settings.lyrics_align_model_english`
        (wav2vec2-large-robust-ft-libri-960h, Apache 2.0, ~317M).

        Caching: per-model-path; each unique path loads once and stays
        warm for the process lifetime. The same lock that gates
        `realign_text` serialises model loads, so concurrent first-hit
        requests for the same model path can't race.

        Precision is per-checkpoint, not per-device. The English head
        (`wav2vec2-large-robust`) runs fp16 on CUDA - it's numerically
        stable there and fp16 halves activation memory + roughly doubles
        throughput with no observable hit to word-alignment accuracy on
        our inputs. MMS-300m (the multilingual default, `model_path is
        None`) runs fp32 everywhere: at fp16 it overflows in the feature-
        encoder LayerNorm/GELU on loud vocal frames, poisoning the whole
        emissions tensor with inf -> NaN so Viterbi collapses every frame
        to `<blank>` (observed as `emissions_stats: nan=~all mean=nan`
        and a downstream `get_spans` divergence). The overflow sits right
        at the fp16 threshold, so it's flaky run-to-run (cuDNN picks
        different conv algorithms) and was masked while the only
        exercised path was the stable English head; the new Japanese path
        routes through MMS-300m and trips it reliably. CPU always gets
        fp32 (no fp16 kernel coverage). The gpu_park machinery parks the
        drum + vocals models off-GPU before alignment, so MMS-300m's
        ~1.2 GB fp32 footprint fits the 6 GB budget. The returned model
        has `.dtype` / `.device` attributes that `load_audio` reads to
        materialise the waveform on the same device + precision as the
        model."""
        cached = self._align_models.get(model_path)
        if cached is not None:
            return cached
        # Lazy import so a process that never touches /lyrics/align
        # doesn't pull in `transformers` + the alignment package's
        # ~1.2 GB model on boot.
        import torch
        from ctc_forced_aligner import (  # type: ignore[import-not-found]
            load_alignment_model,
        )

        device = self._resolve_device()
        # fp16 only for the stable English head on CUDA; MMS-300m (and CPU)
        # must stay fp32 or fp16 NaN-poisons its emissions (see docstring).
        use_fp16 = device == "cuda" and model_path == settings.lyrics_align_model_english
        dtype = torch.float16 if use_fp16 else torch.float32
        log.info(
            "lyrics: loading CTC aligner (model=%s, device=%s, dtype=%s)",
            model_path or "<package default (MMS-300m)>", device, dtype,
        )
        if model_path is None:
            model, tokenizer = load_alignment_model(device, dtype=dtype)
        else:
            model, tokenizer = load_alignment_model(
                device, model_path=model_path, dtype=dtype,
            )
        self._align_models[model_path] = (model, tokenizer)
        return model, tokenizer

    def _load_ctc_onnx(self, model_path: str | None):
        """Torch-free `OnnxCtcAligner` for `model_path` (None -> the MMS default).
        Exports the model's `.onnx` once (cached in settings.models_dir)."""
        cached = self._onnx_aligners.get(model_path)
        if cached is not None:
            return cached
        from app.pipeline.lyrics_onnx import load_onnx_aligner

        resolved = model_path or settings.lyrics_align_model_default
        log.info("lyrics: loading CTC aligner (ONNX, model=%s)", resolved)
        aligner = load_onnx_aligner(resolved, settings.models_dir, providers=_onnx_providers())
        self._onnx_aligners[model_path] = aligner
        return aligner

    def _pick_alignment_model(self, language_code: str) -> str | None:
        """Route a detected ISO-639-1 language code to a CTC checkpoint.

        Returns the HuggingFace model id to load, or `None` to use the
        package's default MMS-300m head. English routes to a
        purpose-built English wav2vec2-large-robust for tighter
        posteriors and a commercial-friendly Apache 2.0 licence;
        everything else falls back to MMS-300m's multilingual head
        (CC-BY-NC, replacement candidates tracked separately).

        Latin-script languages mis-detected as English by
        `_detect_language_from_text` (which returns "en" for any
        Latin-only text it can't tag more specifically) also land on
        the English aligner. That's a deliberate trade: wav2vec2-EN
        copes well with Romance / Germanic Latin-script text per the
        upstream model card, and the dominant case for mis-detection
        is "mostly English with some non-English" anyway."""
        if language_code == "en":
            return settings.lyrics_align_model_english
        return None

    def park(self) -> None:
        """Move every loaded CTC aligner to CPU. Idempotent and a
        no-op when no aligner has been loaded yet.

        Callers must hold the process-wide GPU lock so an in-flight
        `realign_text` isn't mid-`generate_emissions` on the model
        being moved. See `app.pipeline.gpu_park.park_for_transcribe`."""
        from app.pipeline.gpu_park import park_module

        for key, entry in list(self._align_models.items()):
            label = f"ctc_align[{key or 'mms-default'}]"
            park_module(entry[0], label)
        # ONNX aligners hold their VRAM in the ORT session; drop them (they reload
        # lazily from the cached .onnx) so the swap frees it.
        self._onnx_aligners.clear()

    def unpark(self) -> None:
        """Move every loaded CTC aligner back to CUDA. Idempotent."""
        from app.pipeline.gpu_park import unpark_module

        for key, entry in list(self._align_models.items()):
            label = f"ctc_align[{key or 'mms-default'}]"
            unpark_module(entry[0], label)

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
        not clamped to the caller's rough timestamps. Critical for
        plain-text inputs (where caller timestamps are evenly
        synthesised) and for LRCLIB matches against a different cut of
        the same song; both would otherwise mis-window the aligner.

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

        with self._lock:
            # Resolve language first so the CTC checkpoint can be picked
            # accordingly. Detection is text-only (`input_lines`); the
            # final `"en"` default only fires for letter-free text, where
            # any choice romanizes to nothing anyway.
            language_code = (
                language
                or settings.whisper_language
                or _detect_language_from_text(input_lines)
                or "en"
            )
            iso3 = _iso1_to_iso3(language_code)

            model_path = self._pick_alignment_model(language_code)
            # Default: torch-free ONNX (acoustic model on onnxruntime, the Viterbi
            # + everything else numpy); opt out with DRUMJOT_LYRICS_ONNX=0. Each
            # branch binds `alignments_fn(emissions, tokens, tokenizer)`; the
            # emissions call itself is inside the try (below) so a failure degrades
            # gracefully. get_spans / postprocess_results / _repair are shared.
            # `preprocess_text` / `get_spans` / `postprocess_results` are pure
            # numpy/python. On the ONNX path they come from `lyrics_onnx` (which
            # loads the package's torch-free pieces WITHOUT its torch-importing
            # __init__), keeping the whole path torch-free; the torch path imports
            # them (and generate_emissions/get_alignments/load_audio) from the
            # package as before.
            use_onnx = _lyrics_onnx_enabled()
            aligner = model = None
            if use_onnx:
                from app.pipeline.lyrics_onnx import (
                    get_alignments_np,
                    get_spans,
                    load_audio_np,
                    postprocess_results,
                    preprocess_text,
                )

                aligner = self._load_ctc_onnx(model_path)
                tokenizer = aligner.tokenizer
                audio_waveform = load_audio_np(str(audio_path))
                alignments_fn = get_alignments_np
            else:
                from ctc_forced_aligner import (  # type: ignore[import-not-found]
                    generate_emissions,
                    get_alignments,
                    get_spans,
                    load_audio,
                    postprocess_results,
                    preprocess_text,
                )

                model, tokenizer = self._load_ctc_aligner(model_path)
                # load_audio materialises the waveform on the same device +
                # dtype as the model so generate_emissions can run without
                # an extra copy / cast inside its inner loop.
                audio_waveform = load_audio(str(audio_path), model.dtype, model.device)
                alignments_fn = get_alignments

            _log_audio_stats(audio_waveform, audio_path)

            # Decide Japanese-aware romanization. cutlet/fugashi reads
            # kanji with Japanese readings (vs uroman's Chinese inside
            # preprocess_text); `jp_romaji` rewrites Japanese spans to
            # romaji before they reach the aligner and carries the
            # original kana/kanji for display. Skipped for Chinese tracks
            # (the char-level path stays) and when the optional stack
            # isn't installed. `treat_kanji_as_japanese` resolves
            # glyph-ambiguous kanji per the kana/marker heuristic, so a
            # Chinese sprinkle inside an English-dominant track isn't
            # romanized as Japanese.
            use_jp_romaji = iso3 != "chi" and _jp_romaji_available()
            full_text = "".join(line.text for line in input_lines)
            treat_kanji_as_japanese = _resolve_cjk_lang(full_text) == "ja"

            # Per-line preprocess pins the exact per-line word count the
            # aligner will emit (text_normalize strips punctuation, the
            # jp path expands a span into morphemes, etc.), so the
            # downstream partition is always exact. `display_surfaces`
            # rides along, length-locked to the emitted words.
            (
                all_tokens,
                all_text,
                display_surfaces,
                line_word_counts,
                non_empty_indices,
            ) = _preprocess_lines(
                input_lines,
                use_jp_romaji=use_jp_romaji,
                treat_kanji_as_japanese=treat_kanji_as_japanese,
                iso3=iso3,
                preprocess_text=preprocess_text,
            )

            if not all_tokens:
                # Every line was empty or unprocessable; echo back so
                # the response still mirrors the input line count.
                return [
                    LyricLine(start_sec=line.start_sec, text=line.text, words=None)
                    for line in input_lines
                ]

            log.info(
                "lyrics: aligning %d non-empty lines against %s "
                "(language=%s/%s, jp_romaji=%s, total_tokens=%d)",
                len(non_empty_indices), audio_path.name,
                language_code, iso3, use_jp_romaji, sum(line_word_counts),
            )
            _log_token_sequence(all_tokens, all_text)

            try:
                if use_onnx:
                    emissions, stride = aligner.generate_emissions(
                        audio_waveform, batch_size=_CTC_BATCH_SIZE,
                    )
                else:
                    emissions, stride = generate_emissions(
                        model, audio_waveform, batch_size=_CTC_BATCH_SIZE,
                    )
                # `stride`'s unit is package-internal (seen as ms or samples
                # per frame depending on version) so it's not safe to use
                # for our diagnostic time axis. Derive sec/frame from the
                # known input duration instead - the audio waveform is at
                # exactly `_AUDIO_SAMPLE_RATE` because `load_audio` resampled
                # to that, and emissions frame count maps 1:1 to its time
                # axis. `postprocess_results` keeps using `stride` because
                # the package's own conversion handles whatever unit it
                # emits.
                audio_seconds = (
                    int(audio_waveform.shape[-1]) / _AUDIO_SAMPLE_RATE
                )
                _log_emissions_stats(emissions, tokenizer)
                # Fail fast + actionable on poisoned emissions. A non-finite
                # tensor (fp16 overflow on an unstable head, corrupt weights,
                # broken audio) otherwise flows into Viterbi, which collapses
                # to one all-`<blank>` segment and surfaces as a cryptic
                # `get_spans` AssertionError far from the real cause. The
                # `except Exception` below catches this and degrades to
                # returning the caller's lines unchanged.
                # Fail fast on poisoned emissions: a non-finite tensor (fp16
                # overflow on an unstable head, corrupt weights, broken audio)
                # otherwise flows into Viterbi, which collapses to one all-blank
                # segment and surfaces as a cryptic `get_spans` AssertionError far
                # from the cause. The `except Exception` below then degrades to
                # returning the caller's lines unchanged.
                if not _all_finite(emissions):
                    raise ValueError(
                        f"acoustic model emitted non-finite emissions; "
                        f"model={model_path or 'MMS-300m'} - likely numerical "
                        f"instability (load this head in fp32)"
                    )
                _log_emissions_windowed(emissions, audio_seconds, tokenizer)
                segments, scores, blank_token = alignments_fn(
                    emissions, all_tokens, tokenizer,
                )
                try:
                    spans = get_spans(all_tokens, segments, blank_token)
                except AssertionError as inner:
                    _diagnose_get_spans_failure(
                        all_tokens, segments, blank_token, inner,
                    )
                    raise
                word_timestamps = postprocess_results(
                    all_text, spans, stride, scores,
                )
                _log_word_score_diagnostics(
                    word_timestamps, emissions, audio_seconds, tokenizer,
                )
                word_timestamps = _repair_low_score_words(
                    word_timestamps,
                    emissions=emissions,
                    audio_seconds=audio_seconds,
                    all_tokens=all_tokens,
                    all_text=all_text,
                    tokenizer=tokenizer,
                    stride=stride,
                    get_alignments=alignments_fn,
                    get_spans=get_spans,
                    postprocess_results=postprocess_results,
                )
                _log_word_score_diagnostics(
                    word_timestamps, emissions, audio_seconds, tokenizer,
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

            # Partition `display_surfaces` with the same per-line counts.
            # Guaranteed length-locked to `word_timestamps` once the
            # partition above succeeded (both equal sum(line_word_counts)),
            # so words map to surfaces 1:1.
            partitioned_surfaces: list[list[str | None]] = []
            cursor = 0
            for count in line_word_counts:
                partitioned_surfaces.append(display_surfaces[cursor : cursor + count])
                cursor += count

            return _stitch_lines(
                input_lines, non_empty_indices, partitioned, partitioned_surfaces,
            )

def _resolve_cjk_lang(text: str) -> str:
    """Decide whether glyph-ambiguous CJK (kanji/hanzi, identical between
    Japanese and Traditional Chinese) in `text` should be read as
    Japanese or Chinese, using two presence signals:

      - any kana anywhere            -> "ja"
      - else any simplified-Chinese-only marker -> "zh"
      - else                         -> "ja" (J-pop library bias)

    Used both by `_detect_language_from_text` (to count ambiguous CJK
    toward the right language) and by `realign_text` (to decide whether
    to romanize kanji as Japanese via `jp_romaji`). See
    `_SIMPLIFIED_CHINESE_MARKERS` for the conservative marker set."""
    has_kana = False
    has_simplified_marker = False
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            has_kana = True
        if ch in _SIMPLIFIED_CHINESE_MARKERS:
            has_simplified_marker = True
        if has_kana and has_simplified_marker:
            break
    if has_kana:
        return "ja"
    if has_simplified_marker:
        return "zh"
    return "ja"


# Distinctive letters that disambiguate the major Cyrillic-script
# languages, mirroring `_SIMPLIFIED_CHINESE_MARKERS` for CJK. Each set
# holds characters that appear in one language's orthography but not the
# others we route, so a single occurrence tips the whole text. uroman
# romanizes Cyrillic by codepoint regardless, so this only refines the
# per-language hint (e.g. Ukrainian г -> "h" vs Russian г -> "g"); a miss
# degrades to a slightly different romanization, not a failure.
_CYRILLIC_UKRAINIAN_MARKERS = frozenset("іїєґ")
_CYRILLIC_RUSSIAN_MARKERS = frozenset("ыэё")
_CYRILLIC_SERBIAN_MARKERS = frozenset("ђћ")
_CYRILLIC_MACEDONIAN_MARKERS = frozenset("ѓќѕ")


def _resolve_cyrillic_lang(text: str) -> str:
    """Pick a specific Cyrillic-script language from `text` via
    distinctive-letter markers, for the uroman romanization hint on the
    MMS alignment path. Returns an ISO-639-1 code (`uk` / `ru` / `bg` /
    `sr` / `mk`), defaulting to `ru` (the most common Cyrillic lyric
    language) when no distinctive marker is present. Most-specific
    scripts are checked first; mixed/contradictory marker sets are rare
    in real lyrics and resolve to whichever is checked first."""
    chars = {ch.lower() for ch in text}
    if chars & _CYRILLIC_MACEDONIAN_MARKERS:
        return "mk"
    if chars & _CYRILLIC_SERBIAN_MARKERS:
        return "sr"
    if chars & _CYRILLIC_UKRAINIAN_MARKERS:
        return "uk"
    if chars & _CYRILLIC_RUSSIAN_MARKERS:
        return "ru"
    if "ъ" in chars:
        # Bulgarian leans on ъ as a full vowel and lacks the Russian
        # markers above (ы / э / ё), which would already have returned
        # "ru".
        return "bg"
    return "ru"


# Sentinel language tag for "alphabetic text in a script we don't
# classify into a specific language" (Greek, Hebrew, Indic, Armenian,
# Georgian, ...). It deliberately rides two existing defaults rather than
# naming a language: `_pick_alignment_model` routes any non-`en` tag to
# the MMS-300m multilingual head, and `_iso1_to_iso3` maps any unknown
# tag to `eng` - a valid ISO-639-3 that ctc-forced-aligner's uroman pass
# accepts. uroman romanizes by codepoint (the language hint only refines
# a few ambiguous letters), so an unclassified script still romanizes
# correctly and aligns against MMS. The point is purely to stop returning
# None - which would drop the request onto the audio language detector -
# whenever the lyric text has *any* letters.
_OTHER_SCRIPT_LANG = "und"


def _detect_language_from_text(input_lines: list[InputLine]) -> str | None:
    """Cheap, deterministic language detection from caller-provided
    lyric text. Used by `realign_text` so we don't have to trust
    an audio-based detector - which mis-classifies the first 30 s of
    a vocals stem if the intro is silent / instrumental / background-
    vocal-only, and which on noise has been observed to return a
    `LANGUAGES_WITHOUT_SPACES` code even for English audio. That
    mis-classification then shatters every word into a per-letter
    "word" because the CTC aligner tokenises no-space languages
    character-by-character (each char becomes its own alignment unit
    and its own output entry).

    Returns an ISO-639-1 code, or `None` only when the text contains no
    alphabetic characters at all (pure punctuation / digits / empty); the
    caller then falls back to the audio-based detector. Any script with
    letters resolves here instead - Latin -> `en`, the CJK / Korean /
    Thai blocks as before, Cyrillic to a specific language via
    `_resolve_cyrillic_lang`, and every other script (Greek, Hebrew,
    Indic, ...) to the `_OTHER_SCRIPT_LANG` catch-all. All of the
    non-`en` codes route to the MMS-300m + uroman pathway, so we no
    longer drop a non-Latin lyric onto the (slower, less reliable) audio
    detector just because its script wasn't counted.

    **Routing rule: majority by character count.** Each alphabetic
    character votes for one language; the language with the most votes
    wins. This handles bilingual code-switching by following the bulk
    of the text rather than letting any sprinkle of non-Latin script
    flip the whole song. An English-dominant lyric with a few kana
    ad-libs routes to `en` (so it picks up the English-specialized
    aligner); a J-pop song whose verses are mostly Japanese still
    routes to `ja` even if the chorus is English, because the
    verse-character count dwarfs the chorus character count over the
    whole song. The trade-off is a bilingual *line* whose English
    word-count exceeds its Japanese char-count routes to `en` - that's
    a deliberate acceptance, because (a) one bilingual line isn't
    representative of a whole song's character distribution and (b)
    routing to a non-English aligner for an English-majority body
    fragments worse than the inverse.

    For genuinely ambiguous CJK characters (kanji/hanzi, glyph-
    identical between Japanese and Traditional Chinese), the ja/zh
    mapping uses a two-signal heuristic *before* counting:
      - any kana anywhere -> CJK counts as ja
      - any simplified-Chinese-only marker, no kana -> CJK counts as zh
      - otherwise -> CJK counts as ja (J-pop bias for ambiguous text)

    Pure ties (rare in practice) are broken by `counts` dict iteration
    order (`en` first), because Latin is the most common script in
    pop music globally and wav2vec2-EN handles Romance / Germanic
    Latin-script text reasonably.
    """
    text = "".join(line.text for line in input_lines)
    if not text.strip():
        return None

    # Resolve glyph-ambiguous scripts to a single language once for the
    # whole text - one distinctive marker tips every ambiguous char in
    # the same direction (see _resolve_cjk_lang / _resolve_cyrillic_lang).
    cjk_lang = _resolve_cjk_lang(text)
    cyrillic_lang = _resolve_cyrillic_lang(text)

    # Second pass: count alphabetic characters by resolved language.
    # `en` is listed first so it wins pure ties; the non-Latin script
    # buckets are seeded after it so they never steal a tie from Latin
    # (see docstring).
    counts: dict[str, int] = {"en": 0, "ja": 0, "ko": 0, "th": 0, "zh": 0}
    counts.setdefault(cyrillic_lang, 0)
    counts.setdefault(_OTHER_SCRIPT_LANG, 0)
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            counts["ja"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["ko"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["th"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counts[cjk_lang] += 1
        elif 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
            counts[cyrillic_lang] += 1
        elif ch.isalpha() and cp < 0x250:
            counts["en"] += 1
        elif ch.isalpha():
            # Any other script with text (Greek, Hebrew, Indic, ...).
            # Routes through MMS + uroman, which romanizes by codepoint.
            counts[_OTHER_SCRIPT_LANG] += 1

    if sum(counts.values()) == 0:
        return None
    return max(counts, key=lambda k: counts[k])


# --------------------------------------------------------------------
# Helpers for the ctc-forced-aligner pipeline. Kept module-private so
# tests can drive each transformation in isolation.
# --------------------------------------------------------------------


def _log_audio_stats(audio_waveform: Any, audio_path: Path) -> None:
    """Summarise the vocals-stem waveform fed to the aligner so we can
    tell at a glance whether the upstream separator produced sane audio.
    NaN/Inf counts catch upstream numerical tainting; `near_silent`
    catches the case where the separator zeroed out quiet regions and
    the model is being asked to align text against a near-flat signal."""
    if not _is_torch_tensor(audio_waveform):
        return  # numpy/ONNX path: this torch-only diagnostic is skipped
    import torch  # type: ignore[import-not-found]

    numel = audio_waveform.numel()
    if numel == 0:
        log.warning("lyrics: audio_stats: %s is empty", audio_path.name)
        return
    nan = int(torch.isnan(audio_waveform).sum())
    inf = int(torch.isinf(audio_waveform).sum())
    finite = audio_waveform[torch.isfinite(audio_waveform)]
    if finite.numel() == 0:
        log.warning(
            "lyrics: audio_stats: %s shape=%s dtype=%s ALL NON-FINITE "
            "(nan=%d inf=%d)",
            audio_path.name, tuple(audio_waveform.shape), audio_waveform.dtype,
            nan, inf,
        )
        return
    finite_f32 = finite.float()
    abs_max = float(finite_f32.abs().max())
    rms = float(finite_f32.pow(2).mean().sqrt())
    near_silent = float((audio_waveform.float().abs() < 1e-3).float().mean())
    duration_sec = int(audio_waveform.shape[-1]) / _AUDIO_SAMPLE_RATE
    log.info(
        "lyrics: audio_stats: %s shape=%s dtype=%s duration=%.2fs "
        "nan=%d inf=%d abs_max=%.4f rms=%.4f near_silent_frac=%.3f",
        audio_path.name, tuple(audio_waveform.shape), audio_waveform.dtype,
        duration_sec, nan, inf, abs_max, rms, near_silent,
    )


def _repair_low_score_words(
    word_timestamps: list[dict[str, Any]],
    *,
    emissions: Any,
    audio_seconds: float,
    all_tokens: list[str],
    all_text: list[str],
    tokenizer: Any,
    stride: float,
    get_alignments: Any,
    get_spans: Any,
    postprocess_results: Any,
    score_threshold: float = -5.0,
    neighbor_count: int = 2,
) -> list[dict[str, Any]]:
    """Re-run forced alignment locally for every word scoring below
    `score_threshold`.

    The global Viterbi occasionally lands on degenerate solutions where
    one word's frame assignment spans many seconds of audio the model
    couldn't phoneme-match (held vowel after the word, breath, ad-lib,
    instrumental gap). The standard `<star>` absorber isn't usable on
    speech-active frames because real wav2vec2 outputs near-zero
    posterior to the `<star>` column at speech frames; this helper
    re-aligns each pathological word locally with a small window of
    well-aligned neighbors as anchors. Two reasons a local rerun can
    win where the global pass lost:
      - The package's internal 30 s chunking + posterior stitching
        introduces cross-chunk boundary effects; aligning a short
        window avoids them entirely.
      - The local Viterbi minimisation is over a much smaller trellis;
        a stretched-word path that was globally optimal can lose to a
        tighter alternative locally because the gain from "stretching
        the next word too" no longer applies.

    Conservative accept-gate: replace the local cluster's timings only
    when the targeted bad word's score strictly improves. A no-op
    rerun (same audio + same text + same package = same path) never
    makes things worse, and a regression on the targeted word means
    something about the local context is also broken and we don't want
    to propagate that into neighbours either.

    Default threshold `-5.0` is roughly the median score on a typical
    pop vocals stem aligned with MMS-300m; tune lower (e.g. `-10`) to
    target only the worst handful of words, higher to be more
    aggressive at the cost of compute on many no-op reruns."""
    if not word_timestamps:
        return word_timestamps

    n_words = len(word_timestamps)
    # Map word index (in word_timestamps, which has stars filtered out)
    # to its position in `all_text` (which retains stars). Used to locate
    # the surrounding `<star>` slots for the sub-token slice.
    word_to_all_text_idx = [i for i, t in enumerate(all_text) if t != "<star>"]
    if len(word_to_all_text_idx) != n_words:
        log.warning(
            "lyrics: repair: skipping (word_timestamps/all_text mismatch: %d vs %d)",
            n_words, len(word_to_all_text_idx),
        )
        return word_timestamps

    em_dim = int(emissions.ndim)  # .ndim works for both torch tensors and numpy
    total_frames = int(emissions.shape[1 if em_dim == 3 else 0])
    sec_per_frame = audio_seconds / max(total_frames, 1)

    # Every word scoring below the threshold becomes a repair target.
    # Walk in index order so cluster merging below stays left-to-right.
    bad_indices = [
        i for i, w in enumerate(word_timestamps)
        if float(w.get("score", 0.0)) < score_threshold
    ]

    # Cluster nearby bad words so their windows don't overlap. The
    # condition `next - last <= 2*neighbor_count` is the boundary at
    # which their N-neighbour windows touch; merging avoids re-aligning
    # the same audio twice with inconsistent boundaries.
    clusters: list[list[int]] = []
    for idx in bad_indices:
        if clusters and idx - clusters[-1][-1] <= 2 * neighbor_count:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    out = list(word_timestamps)
    accepted = 0
    for cluster in clusters:
        n_lo = max(0, cluster[0] - neighbor_count)
        n_hi = min(n_words - 1, cluster[-1] + neighbor_count)

        # Frame range spans from the left-neighbour start to the
        # right-neighbour end. Conversion uses our audio-derived
        # sec/frame, not the package's `stride` (whose unit we don't
        # trust per the diagnostic-bug investigation earlier).
        left_start = float(word_timestamps[n_lo].get("start", 0.0))
        right_end = float(word_timestamps[n_hi].get("end", audio_seconds))
        f_lo = max(0, int(left_start / max(sec_per_frame, 1e-9)))
        f_hi = min(total_frames, int(right_end / max(sec_per_frame, 1e-9)) + 1)
        if f_hi - f_lo < 4:
            continue

        # Token slice: from the `<star>` immediately before the leftmost
        # neighbour to the `<star>` immediately after the rightmost.
        # `preprocess_text` produces alternating star/word/star, so
        # `word_to_all_text_idx[n] ± 1` lands on the surrounding star.
        t_left = word_to_all_text_idx[n_lo]
        t_right = word_to_all_text_idx[n_hi]
        token_lo = max(0, t_left - 1)
        token_hi = min(len(all_tokens), t_right + 2)
        sub_tokens = all_tokens[token_lo:token_hi]
        sub_text = all_text[token_lo:token_hi]

        local_em = emissions[f_lo:f_hi] if em_dim == 2 else emissions[:, f_lo:f_hi]

        try:
            local_segments, local_scores, blank_token = get_alignments(
                local_em, sub_tokens, tokenizer,
            )
            local_spans = get_spans(sub_tokens, local_segments, blank_token)
            local_wt = postprocess_results(
                sub_text, local_spans, stride, local_scores,
            )
        except Exception as exc:
            log.warning(
                "lyrics: repair: cluster %s re-align failed: %s",
                cluster, exc,
            )
            continue

        expected = n_hi - n_lo + 1
        if len(local_wt) != expected:
            log.warning(
                "lyrics: repair: cluster %s word-count mismatch (got %d, expected %d)",
                cluster, len(local_wt), expected,
            )
            continue

        # Accept-gate: leftmost bad word in the cluster must improve.
        target = cluster[0]
        target_old_score = float(word_timestamps[target].get("score", float("-inf")))
        target_new_score = float(local_wt[target - n_lo].get("score", float("-inf")))
        if target_new_score <= target_old_score:
            log.info(
                "lyrics: repair: cluster %s SKIP (word %d %r score %.2f -> %.2f)",
                cluster, target, word_timestamps[target].get("text", ""),
                target_old_score, target_new_score,
            )
            continue

        # Accepted: rewrite timings for all words in the window. Local
        # timestamps from `postprocess_results` are relative to the
        # slice start; shift by `offset_sec` to get global times.
        offset_sec = f_lo * sec_per_frame
        for local_i, global_i in enumerate(range(n_lo, n_hi + 1)):
            new_w = dict(local_wt[local_i])
            new_w["start"] = float(new_w.get("start", 0.0)) + offset_sec
            new_w["end"] = float(new_w.get("end", 0.0)) + offset_sec
            out[global_i] = new_w
        accepted += 1
        log.info(
            "lyrics: repair: cluster %s ACCEPT (word %d %r score %.2f -> %.2f, "
            "old span [%.3f, %.3f]s -> new span [%.3f, %.3f]s)",
            cluster, target, word_timestamps[target].get("text", ""),
            target_old_score, target_new_score,
            float(word_timestamps[target].get("start", 0.0)),
            float(word_timestamps[target].get("end", 0.0)),
            float(out[target].get("start", 0.0)),
            float(out[target].get("end", 0.0)),
        )

    log.info(
        "lyrics: repair: %d/%d cluster(s) accepted (threshold=%.2f, "
        "bad_words=%d, neighbor_count=%d)",
        accepted, len(clusters), score_threshold, len(bad_indices),
        neighbor_count,
    )
    return out


def _log_token_sequence(all_tokens: list[str], all_text: list[str]) -> None:
    """Print exactly what we hand to `get_alignments` so we can verify
    `<star>` token placement empirically.

    Three numbers tell the story:
      - `total` vs `non_star` distinguishes "did we add stars" from "are
        they being counted as words"; the gap is the absorber budget.
      - `consecutive_star_pairs` measures whether per-word concatenation
        actually produced extra stars or whether the package (or our
        own logic) collapsed them.
      - `lone_stars` (stars with non-star neighbours on both sides) is
        the number of in-line absorber slots Viterbi can use without
        sharing them with another star.

    Also dumps the first 40 token strings so a human can sanity-check
    the structure ("<star>, word, <star>, word, …" is what we want;
    "<star>, word, word, word, <star>" would mean stars were collapsed
    out somewhere)."""
    total = len(all_tokens)
    star_count = sum(1 for t in all_text if t == "<star>")
    non_star = total - star_count
    consecutive_pairs = 0
    lone_stars = 0
    for i, t in enumerate(all_text):
        if t != "<star>":
            continue
        prev_is_star = i > 0 and all_text[i - 1] == "<star>"
        next_is_star = i + 1 < total and all_text[i + 1] == "<star>"
        if prev_is_star:
            consecutive_pairs += 1
        if not prev_is_star and not next_is_star:
            lone_stars += 1
    log.info(
        "lyrics: token_sequence: total=%d non_star=%d star=%d "
        "consecutive_star_pairs=%d lone_stars=%d",
        total, non_star, star_count, consecutive_pairs, lone_stars,
    )
    head = all_text[:40]
    log.info("lyrics:   head[:40] = %r", head)


def _log_emissions_stats(emissions: Any, tokenizer: Any) -> None:
    """Summarise MMS-300m's per-frame log-probs before forced alignment.

    Three failure modes we're trying to catch:
      - `nan` / `inf` > 0 -> fp16 instability in the aligner itself
        (LayerNorm / softmax over fp16 activations); Viterbi degenerates
        to whatever the C++ kernel does with NaN log-probs.
      - `top_label_frac` near 1.0 -> the model is predicting the same
        class at every frame (the failure mode we already observed:
        every frame -> 'r', producing one giant segment). Indicates
        either fp16 NaN that collapsed softmax to a default class or a
        broken input waveform.
      - `<star>` argmax dominant + `star_margin` > 5 -> the appended
        wildcard column is winning by miles (>150x prob ratio over the
        runner-up). Distinguishes "model genuinely thinks audio is OOV"
        (margin 0-2 logp; possibly real instrumental input) from
        "<star> column was initialized to ~+inf or model load is
        corrupt" (margin large and uniform across all frames).

    Decodes the dominant argmax index via the tokenizer vocab so the
    log line reads "top=r 0.92" rather than just a bare integer. The
    `<star>` column appended by `generate_emissions` lives at index
    `vocab_size` and shows up as '<star-col>' since it isn't in the
    tokenizer's own vocab map."""
    if not _is_torch_tensor(emissions):
        return  # numpy/ONNX path: this torch-only diagnostic is skipped
    import torch  # type: ignore[import-not-found]

    nan = int(torch.isnan(emissions).sum())
    inf = int(torch.isinf(emissions).sum())
    em_f32 = emissions.float()
    mean = float(em_f32.mean())
    std = float(em_f32.std())

    argmax = em_f32.argmax(dim=-1).view(-1)
    total = argmax.numel()
    counts = torch.bincount(argmax)
    k = min(3, counts.numel())
    top_counts, top_indices = torch.topk(counts, k=k)
    vocab = tokenizer.get_vocab()
    vocab_inv = {v: k_ for k_, v in vocab.items()}
    star_col = len(vocab)  # `generate_emissions` appends this column

    # Global star-vs-runner-up margin. We always compute it, not just
    # when star dominates, so a non-dominant star with a large margin
    # over an obscure runner-up still surfaces in the log.
    if star_col < em_f32.shape[-1]:
        flat = em_f32.view(-1, em_f32.shape[-1])
        star_logp = float(flat[:, star_col].mean())
        mask = torch.ones(flat.shape[-1], dtype=torch.bool, device=flat.device)
        mask[star_col] = False
        runner_up_logp = float(flat[:, mask].max(dim=-1).values.mean())
        star_margin = star_logp - runner_up_logp
        star_summary = (
            f" star_logp={star_logp:.2f} "
            f"runner_up_logp={runner_up_logp:.2f} "
            f"star_margin={star_margin:+.2f}"
        )
    else:
        star_summary = ""

    def _label(idx: int) -> str:
        if idx == star_col:
            return "<star-col>"
        return vocab_inv.get(idx, f"?{idx}")

    top_summary = ", ".join(
        f"{_label(int(i))}={int(c)}({int(c) / total:.2f})"
        for i, c in zip(top_indices, top_counts, strict=True)
    )
    log.info(
        "lyrics: emissions_stats: shape=%s dtype=%s nan=%d inf=%d "
        "mean=%.3f std=%.3f top_argmax=[%s]%s",
        tuple(emissions.shape), emissions.dtype, nan, inf, mean, std,
        top_summary, star_summary,
    )


def _log_emissions_windowed(
    emissions: Any, audio_seconds: float, tokenizer: Any,
    window_seconds: float = 5.0,
) -> None:
    """Per-time-window summary of MMS-300m's posteriors so we can
    correlate a bad-alignment span to "the model was confident here"
    vs. "the model was mush here".

    Cascade triage matrix (read alongside `_log_word_score_diagnostics`):

      - low `max_phoneme_prob` + low word score in same window
            -> model lost the audio (separator artifact, off-mic vocal,
               or genuinely no speech). Fix is upstream: better vocals
               separator, or VAD-gating the posteriors to discourage
               Viterbi from placing words in dead frames.
      - high `max_phoneme_prob` + low word score in same window
            -> model heard a phoneme but it doesn't match the text token
               Viterbi was forced to consume. Fix is text-side: wrong
               language pick, missing `<star>` tokens around ad-libs /
               harmonies, or the LRC text disagrees with what's actually
               sung.
      - high `star_frac` over many consecutive windows
            -> the aligner found nothing it wanted to commit to; usually
               an instrumental section. Words placed in this span are
               cascade victims by definition; VAD-gating fixes it.

    Reports `max_phoneme_prob` (avg-over-frames of the max non-blank,
    non-`<star>` probability in linear space) rather than raw log-probs
    so the numbers read as "0.84 = confident, 0.05 = mush" instead of
    requiring an exp() in your head. Top-3 argmax classes ride the same
    `<star-col>` label vocabulary as `_log_emissions_stats` for
    cross-line greppability."""
    if not _is_torch_tensor(emissions):
        return  # numpy/ONNX path: this torch-only diagnostic is skipped
    import torch  # type: ignore[import-not-found]

    em = emissions.float()
    if em.dim() == 3:
        # `generate_emissions` returns shape (1, T, V+1) in some versions;
        # flatten the leading batch dim so the rest of this function can
        # treat emissions as a 2D (T, V+1) matrix unconditionally.
        em = em[0]
    total_frames = em.shape[0]
    # Seconds per emission frame derived from the known input duration so
    # we don't have to know `generate_emissions`'s `stride` unit (it's
    # been observed in ms in this version, samples in others). Robust
    # to package version drift.
    sec_per_frame = audio_seconds / max(total_frames, 1)
    frames_per_window = max(1, int(round(window_seconds / max(sec_per_frame, 1e-9))))
    vocab = tokenizer.get_vocab()
    vocab_inv = {v: k_ for k_, v in vocab.items()}
    star_col = len(vocab)
    # CTC blank for wav2vec2-style models is conventionally index 0;
    # excluding it from "phoneme confidence" keeps the metric focused on
    # whether the model is committing to *any* real phoneme in the window.
    # Mis-identifying blank only slightly skews `max_phoneme_prob` (the
    # max over a long axis is dominated by the actual peak phoneme), so
    # diagnostics stay informative even if a future model uses a
    # different blank index.
    blank_col = 0
    real_phoneme_mask = torch.ones(em.shape[1], dtype=torch.bool, device=em.device)
    real_phoneme_mask[blank_col] = False
    if star_col < em.shape[1]:
        real_phoneme_mask[star_col] = False

    log.info(
        "lyrics: emissions_windowed: T=%d audio=%.2fs sec/frame=%.4f "
        "window=%.1fs frames/win=%d",
        total_frames, audio_seconds, sec_per_frame, window_seconds,
        frames_per_window,
    )
    for w_start in range(0, total_frames, frames_per_window):
        w_end = min(total_frames, w_start + frames_per_window)
        window = em[w_start:w_end]
        t_start = w_start * sec_per_frame
        t_end = w_end * sec_per_frame
        # Per-frame max log-prob over real phoneme classes; mean across
        # frames in the window, then exp -> linear avg "how confident
        # was the model about *some* phoneme each frame".
        non_special = window[:, real_phoneme_mask]
        max_phoneme_logp = non_special.max(dim=-1).values
        max_phoneme_prob = float(max_phoneme_logp.exp().mean())
        # Argmax-only stats: fraction of frames where star / blank won,
        # and top-3 classes by argmax count. Together these distinguish
        # "model picked a phoneme but the wrong one" from "model couldn't
        # commit to anything but blank/star".
        argmax = window.argmax(dim=-1)
        argmax_n = argmax.numel()
        star_frac = float((argmax == star_col).float().mean())
        blank_frac = float((argmax == blank_col).float().mean())
        counts = torch.bincount(argmax, minlength=star_col + 1)
        top_k = min(3, int((counts > 0).sum()))
        if top_k > 0:
            top_counts, top_indices = torch.topk(counts, k=top_k)
            top_summary = ", ".join(
                _label_argmax_class(int(i), vocab_inv, star_col)
                + f"={int(c) / argmax_n:.2f}"
                for i, c in zip(top_indices, top_counts, strict=True)
            )
        else:
            top_summary = "(empty)"
        # When <star> dominates, also report HOW dominant. A small margin
        # (1-2 logp) over the runner-up means the model is genuinely
        # uncertain and slightly preferring star ("I don't know what
        # this audio is"). A huge margin (>5 logp ~= >150x prob ratio)
        # means star is winning for non-modelling reasons (corrupt
        # weights, mis-loaded model, broken audio preprocessing) and
        # the model never actually evaluated this window.
        star_extra = ""
        if star_col < em.shape[1] and star_frac > 0.5:
            star_logp = float(window[:, star_col].mean())
            runner_up_logp = float(
                window[:, real_phoneme_mask].max(dim=-1).values.mean()
            )
            star_extra = (
                f" star_logp={star_logp:.2f} "
                f"runner_up_logp={runner_up_logp:.2f} "
                f"star_margin={star_logp - runner_up_logp:+.2f}"
            )
        log.info(
            "lyrics:   t=[%6.2f,%6.2f]s max_phoneme_prob=%.3f "
            "blank_frac=%.2f star_frac=%.2f top=[%s]%s",
            t_start, t_end, max_phoneme_prob, blank_frac, star_frac,
            top_summary, star_extra,
        )


def _log_word_score_diagnostics(
    word_timestamps: list[dict[str, Any]],
    emissions: Any,
    audio_seconds: float,
    tokenizer: Any,
) -> None:
    """Distribution + worst-N report for per-word alignment scores.

    `score` from `postprocess_results` is the mean log-prob along the
    Viterbi path for that word's frames. Sharply negative scores mean
    Viterbi traversed low-probability frames to land the word there -
    the canonical signature of a *forced* placement: either the word
    sits in an instrumental section (no phoneme matched), or it was
    shifted by an upstream cascade and ended up on the wrong phonemes.

    For each of the worst-N words we also re-read the emissions inside
    that word's frame range and report `max_phoneme_prob` there. This
    is the cross-correlation the per-window logger above lets you do by
    eye, baked in:

      - low score + low max_phoneme_prob
            -> word placed in a dead-audio span (cascade victim or
               instrumental). VAD-gating / better separator fixes it.
      - low score + high max_phoneme_prob
            -> model heard *something* there but it disagreed with the
               text token Viterbi was forced to consume. Wrong language,
               missing `<star>` for ad-libs, or LRC text mismatch.
    """
    import torch  # type: ignore[import-not-found]

    if not word_timestamps:
        log.info("lyrics: word_scores: no words (skipping)")
        return
    if not _is_torch_tensor(emissions):
        return  # numpy/ONNX path: this torch-only diagnostic is skipped
    em = emissions.float()
    if em.dim() == 3:
        em = em[0]
    total_frames = em.shape[0]
    # Same audio-derived sec/frame as `_log_emissions_windowed`; package
    # `stride` units differ across versions so we don't trust it here.
    sec_per_frame = audio_seconds / max(total_frames, 1)
    vocab = tokenizer.get_vocab()
    star_col = len(vocab)
    blank_col = 0
    real_phoneme_mask = torch.ones(em.shape[1], dtype=torch.bool, device=em.device)
    real_phoneme_mask[blank_col] = False
    if star_col < em.shape[1]:
        real_phoneme_mask[star_col] = False

    scores = [float(w.get("score", 0.0)) for w in word_timestamps]
    scores_sorted = sorted(scores)
    n = len(scores)

    def _percentile(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return scores_sorted[idx]

    # Threshold of -1.5 is a coarse heuristic: forced-aligner scores on
    # cleanly-recognised words usually sit between -0.5 and 0; below
    # ~-1.5 the path is averaging blank / wrong-phoneme frames. Re-tune
    # once we have a few real songs' baselines logged.
    threshold = -1.5
    below = sum(1 for s in scores if s < threshold)
    log.info(
        "lyrics: word_scores: n=%d min=%.2f p10=%.2f median=%.2f p90=%.2f max=%.2f below_%.1f=%d",
        n, scores_sorted[0], _percentile(0.10), _percentile(0.50),
        _percentile(0.90), scores_sorted[-1], threshold, below,
    )

    worst = sorted(
        ((float(w.get("score", 0.0)), w) for w in word_timestamps),
        key=lambda t: t[0],
    )[:10]
    log.info("lyrics:   worst 10 words by alignment score:")
    for score, w in worst:
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))
        f_start = max(0, min(total_frames - 1, int(start / max(sec_per_frame, 1e-9))))
        f_end = max(f_start + 1, min(total_frames, int(end / max(sec_per_frame, 1e-9))))
        window = em[f_start:f_end]
        if window.numel() > 0:
            non_special = window[:, real_phoneme_mask]
            max_phoneme_prob = float(non_special.max(dim=-1).values.exp().mean())
        else:
            max_phoneme_prob = float("nan")
        log.info(
            "lyrics:     t=[%7.3f,%7.3f]s score=%6.2f max_phoneme_prob=%.3f text=%r",
            start, end, score, max_phoneme_prob, w.get("text", ""),
        )


def _label_argmax_class(idx: int, vocab_inv: dict[int, str], star_col: int) -> str:
    """Decode an argmax index to the same label vocabulary
    `_log_emissions_stats` uses (`<star-col>` for the appended wildcard,
    `?N` for genuinely unknown indices)."""
    if idx == star_col:
        return "<star-col>"
    return vocab_inv.get(idx, f"?{idx}")


def _diagnose_get_spans_failure(
    all_tokens: list[str],
    segments: list[Any],
    blank: str,
    error: AssertionError,
) -> None:
    """Re-walk `get_spans`'s loop to pinpoint where it diverged, then log
    a snapshot. Strictly diagnostic; the original AssertionError is still
    raised by the caller. Mirrors `get_spans`'s state machine exactly so
    the reproduced position matches the package's failure site."""
    tokens_idx = 0
    ltr_idx = 0
    non_blank_count = 0
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(all_tokens):
            break
        if seg.label == blank:
            continue
        cur_token = all_tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label != ltr:
            ctx_lo = max(0, tokens_idx - 2)
            ctx_hi = min(len(all_tokens), tokens_idx + 3)
            seg_lo = max(0, seg_idx - 5)
            seg_hi = min(len(segments), seg_idx + 6)
            log.warning(
                "lyrics: get_spans diverged at seg_idx=%d (label=%r) vs "
                "tokens_idx=%d ltr_idx=%d (token=%r expected_ltr=%r). "
                "AssertionError: %s",
                seg_idx, seg.label, tokens_idx, ltr_idx,
                all_tokens[tokens_idx], ltr, error,
            )
            log.warning(
                "lyrics:   token context [%d:%d] = %r",
                ctx_lo, ctx_hi, all_tokens[ctx_lo:ctx_hi],
            )
            log.warning(
                "lyrics:   segment context [%d:%d] = %r",
                seg_lo, seg_hi,
                [(s.label, s.start, s.end) for s in segments[seg_lo:seg_hi]],
            )
            log.warning(
                "lyrics:   totals: tokens=%d segments=%d non_blank_segs_so_far=%d",
                len(all_tokens), len(segments), non_blank_count,
            )
            return
        non_blank_count += 1
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            while tokens_idx < len(all_tokens) and len(all_tokens[tokens_idx]) == 0:
                tokens_idx += 1
        else:
            ltr_idx += 1
    log.warning(
        "lyrics: get_spans diagnostic walked the entire loop without "
        "reproducing the divergence; AssertionError was: %s",
        error,
    )




# ISO-639-1 (Whisper / our text detector) -> ISO-639-3 (the language
# code expected by ctc_forced_aligner.preprocess_text, which feeds MMS
# romanization). Covers the codes _detect_language_from_text emits
# (en/ja/ko/zh/th plus the Cyrillic set uk/ru/bg/sr/mk) plus a few common
# Latin-script tags so callers can pin specific Romance / Germanic
# languages through `settings.whisper_language` or the request's
# `language` field. Anything else - including the `_OTHER_SCRIPT_LANG`
# (`und`) catch-all - falls back to `eng`, which is deliberate: MMS
# handles unspecified text fine and uroman romanizes the real script by
# codepoint, but the aligner barfs on an *unknown* ISO-639-3 code, so an
# unrecognized tag must resolve to a valid one rather than pass through.
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
    # Cyrillic-script languages emitted by `_resolve_cyrillic_lang`.
    "uk": "ukr",
    "bg": "bul",
    "sr": "srp",
    "mk": "mkd",
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


_jp_romaji_ok: bool | None = None


def _jp_romaji_available() -> bool:
    """True when the cutlet/fugashi stack can be imported. Probed once
    and cached. When false, callers leave Japanese kanji on the existing
    uroman path (romanized as Chinese) rather than crashing - a graceful
    degrade for environments where the optional Japanese deps aren't
    installed."""
    global _jp_romaji_ok
    if _jp_romaji_ok is None:
        try:
            import cutlet  # type: ignore[import-not-found]  # noqa: F401

            _jp_romaji_ok = True
        except Exception as exc:
            log.warning(
                "lyrics: cutlet/fugashi unavailable; Japanese kanji will be "
                "romanized as Chinese (uroman fallback): %s",
                exc,
            )
            _jp_romaji_ok = False
    return _jp_romaji_ok


def _preprocess_lines(
    input_lines: list[InputLine],
    *,
    use_jp_romaji: bool,
    treat_kanji_as_japanese: bool,
    iso3: str,
    preprocess_text: Any,
) -> tuple[list[str], list[str], list[str | None], list[int], list[int]]:
    """Build the flat aligner inputs from caller lyric lines.

    Returns `(all_tokens, all_text, display_surfaces, line_word_counts,
    non_empty_indices)`:

      - `all_tokens` / `all_text`: concatenated `preprocess_text` output
        over every non-empty line (`all_text` keeps `<star>` markers).
      - `display_surfaces`: flat list parallel to the NON-star words (one
        entry per emitted alignment word, in order). A Japanese token
        contributes its original surface (e.g. "君"); every other token
        contributes `None`, meaning "display the aligned text as-is".
      - `line_word_counts` / `non_empty_indices`: per-line non-star word
        counts and their indices into `input_lines`, for partitioning.

    When `use_jp_romaji` is set, each line is tokenized by `jp_romaji`
    and every token's romaji is preprocessed individually at
    `language='eng'` (word-level: one `<star>` slot between every pair of
    tokens, the same absorber budget the English path already relies on).
    Otherwise (Chinese, or the stack unavailable) the line takes the
    original whole-line path at `language=iso3` - char-level for `chi`.

    The load-bearing invariant is that `display_surfaces` stays
    length-locked to the emitted word list: each token extends it by
    exactly the number of non-star words that token produced, so the
    downstream partition maps words to surfaces 1:1."""
    all_tokens: list[str] = []
    all_text: list[str] = []
    display_surfaces: list[str | None] = []
    line_word_counts: list[int] = []
    non_empty_indices: list[int] = []

    for idx, line in enumerate(input_lines):
        t = line.text.strip()
        if not t:
            continue

        # `units` is a list of (text_for_aligner, display_surface, lang).
        # display_surface is None for anything we want rendered as the
        # aligned text (English, Chinese); the original kana/kanji for
        # Japanese tokens.
        units: list[tuple[str, str | None, str]]
        if use_jp_romaji:
            try:
                from app.pipeline.jp_romaji import tokenize

                units = [
                    (tok.romaji, tok.surface if tok.is_japanese else None, "eng")
                    for tok in tokenize(
                        t, treat_kanji_as_japanese=treat_kanji_as_japanese
                    )
                ]
            except Exception as exc:
                log.warning(
                    "lyrics: jp_romaji.tokenize failed for line %d (%r); "
                    "falling back to char-level path: %s",
                    idx, t, exc,
                )
                units = [(t, None, iso3)]
        else:
            units = [(t, None, iso3)]

        line_tokens: list[str] = []
        line_text: list[str] = []
        line_surfaces: list[str | None] = []
        line_real = 0
        preprocess_failed = False
        for unit_in, surface, lang in units:
            try:
                unit_tokens, unit_text = preprocess_text(
                    unit_in, romanize=True, language=lang,
                )
            except Exception as exc:
                log.warning(
                    "lyrics: preprocess_text failed for unit %r in line %d "
                    "(%r): %s",
                    unit_in, idx, t, exc,
                )
                preprocess_failed = True
                break
            unit_real = sum(1 for s in unit_text if s != "<star>")
            line_tokens.extend(unit_tokens)
            line_text.extend(unit_text)
            line_surfaces.extend([surface] * unit_real)
            line_real += unit_real

        if preprocess_failed or line_real == 0:
            continue
        all_tokens.extend(line_tokens)
        all_text.extend(line_text)
        display_surfaces.extend(line_surfaces)
        line_word_counts.append(line_real)
        non_empty_indices.append(idx)

    return all_tokens, all_text, display_surfaces, line_word_counts, non_empty_indices


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
    partitioned_surfaces: list[list[str | None]] | None = None,
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

    `partitioned_surfaces`, when given, is parallel to `partitioned`
    (same per-line word lists) and carries the display surface for each
    word: the original kana/kanji for Japanese tokens, or None to render
    the aligned text directly. When a surface is present the rendered
    `text` is the surface and the aligned romaji is preserved on
    `romaji` for the debug tooltip; when absent (English / Chinese), the
    aligned text is rendered as-is and `romaji` stays None - the
    pre-existing behavior.
    """
    by_input_idx = dict(zip(non_empty_indices, partitioned, strict=True))
    surf_by_idx: dict[int, list[str | None]] = {}
    if partitioned_surfaces is not None:
        surf_by_idx = dict(
            zip(non_empty_indices, partitioned_surfaces, strict=True)
        )
    out: list[LyricLine] = []
    for idx, line in enumerate(input_lines):
        words = by_input_idx.get(idx)
        if not words:
            out.append(LyricLine(start_sec=line.start_sec, text=line.text, words=None))
            continue
        surfaces = surf_by_idx.get(idx)
        lyric_words: list[LyricWord] = []
        for w_i, w in enumerate(words):
            start_sec = float(w.get("start", 0.0))
            raw_end = float(w.get("end", start_sec + 0.05))
            aligned_text = str(w.get("text", "")).strip()
            surface = surfaces[w_i] if surfaces is not None else None
            if surface is not None:
                # Display the original kana/kanji; keep the aligned romaji
                # for the debug tooltip.
                text = surface
                romaji: str | None = aligned_text or None
            else:
                text = aligned_text
                romaji = None
            if not text:
                continue
            # CTC alignment is occasionally degenerate on syllables it
            # can't place (a held vowel that wav2vec2 absorbs into the
            # neighbouring word, etc.); clamp so the cell never inverts
            # downstream. Preserve the model's raw end in `raw_end_sec`
            # so the UI tooltip can show what the aligner emitted vs
            # what we use. Marker vocabulary stays stable across any
            # future aligner backend so the frontend doesn't have to
            # know which one produced the data.
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
                    romaji=romaji,
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
    `endFallback`, `romaji`) only when set, so the response payload stays
    small on the common case where the model emitted complete timings."""
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
    if w.romaji is not None:
        entry["romaji"] = w.romaji
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
