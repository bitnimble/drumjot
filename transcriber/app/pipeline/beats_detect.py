"""Beat + downbeat detection and the `analyze_beats` orchestration.

Runs Beat This! (torch-free ONNX by default; `File2Beats` under
`DRUMJOT_BEAT_ONNX=0`) to get raw beat/downbeat times, converts them through
meter recovery + smoothing into a `BeatStructure`, and `analyze_beats` chains
the whole pipeline: detect -> (optional) align -> tempo finalize -> pad. Owns
the lazy model cache (`_beat_this_model` + `park_model`/`unpark_model`) and the
librosa fallback.

This is the top of the beats module graph: it imports the meter, structure,
tempo and alignment modules.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.pipeline.beats_align import align_beats_to_envelope, align_beats_to_onsets
from app.pipeline.beats_meter import (
    _beats_downbeats_to_raw,
    _recover_bar_length_if_incoherent,
)
from app.pipeline.beats_structure import _finalize_bar, _raw_to_structure, _summarize
from app.pipeline.beats_tempo import _finalize_bar_tempos, _pad_trailing_bars
from app.pipeline.beats_types import BarInfo, BeatStructure, BeatTick

log = logging.getLogger(__name__)


# ---------- Detection ----------

def analyze_beats(
    audio_path: Path,
    duration_seconds: float | None = None,
    align_onsets: list[tuple[float, float]] | None = None,
) -> BeatStructure:
    """Run beat + downbeat detection on the audio at `audio_path`.

    Uses Beat This! (DBN-free, meter-agnostic) for the actual ML; falls
    back to a librosa-based heuristic if it isn't importable, so the rest
    of the pipeline degrades gracefully rather than failing outright.

    When `duration_seconds` is supplied, the returned structure is padded
    with synthetic bars after the last detected beat so that every
    timestamp within the audio falls inside some bar. This stops onsets
    that occur after the final detected beat from piling up at an
    out-of-range `beat_in_bar` value on the last real bar (see
    `position`).

    When `align_onsets` is supplied (list of `(time, strength)` tuples),
    each detected beat is snapped to the strongest nearby drum onset
    before bars are padded. Neural beat trackers (BT especially) tend
    to report beat times ~50 ms past the actual transient because the
    activation peak lags the strike; snapping to onsets removes that
    systematic lag without changing bar phase or beat count.
    """
    try:
        structure = _beat_this_beats(audio_path)
    except Exception as exc:
        log.warning(
            "Beat This! beat tracking failed (%s); falling back to librosa "
            "(no downbeat classification - assuming 4/4 throughout)",
            exc,
        )
        structure = _librosa_fallback(audio_path)
    if align_onsets:
        # Coarse envelope phase-align first (wide ±half-bar search) so the
        # fine onset snap below isn't starved by a multi-slot phase error
        # outside its ±50 ms window, then remove the residual lag.
        align_beats_to_envelope(structure, audio_path)
        align_beats_to_onsets(structure, align_onsets)
    _finalize_bar_tempos(structure)
    if duration_seconds is not None and duration_seconds > 0:
        _pad_trailing_bars(structure, duration_seconds)
    return structure


_BEAT_THIS_MODEL = None


def _beat_onnx_enabled() -> bool:
    """Default ON: run Beat This! through onnxruntime (torch-free inference).
    Opt out with DRUMJOT_BEAT_ONNX in {0,false,no,off,torch} for the torch path
    (mirrors DRUMJOT_SEP_ONNX / DRUMJOT_ONSET_ONNX)."""
    import os

    return os.environ.get("DRUMJOT_BEAT_ONNX", "1").strip().lower() not in (
        "0", "false", "no", "off", "torch",
    )


def _onnx_providers():
    """onnxruntime providers from settings.device (no torch import): CPU-pinned
    only when CPU is forced, else the available set (+ CPU fallback in the loader)
    -- which is CUDA on Linux/Windows and CoreML on macOS. `mps`/`coreml` are NOT
    CPU-pinned: ORT's Apple EP is CoreML, so they take the available set too."""
    from app.config import settings

    dev = (settings.device or "auto").lower()
    return ["CPUExecutionProvider"] if dev == "cpu" else None


def _beat_this_model():
    """Lazily build the cached Beat This! inference wrapper.

    Beat This! (Foscarin et al., ISMIR 2024) is DBN-free, so it tracks
    beats + downbeats jointly with no fixed beats-per-bar prior, which is
    why it handles odd/compound meters where the old madmom-DBN path
    (madmom RNN and Beat Transformer both fed it) collapsed. Weights
    auto-download to the torch hub cache on first use. Lazy so an import/
    download hiccup doesn't block service startup.

    Default: the torch-free ONNX wrapper (`OnnxBeatThis`, the transformer on
    onnxruntime + numpy frontend/chunking/postproc); opt out with
    DRUMJOT_BEAT_ONNX=0 for the torch `File2Beats`. Both are callables
    `audio_path -> (beats, downbeats)`.
    """
    global _BEAT_THIS_MODEL
    if _BEAT_THIS_MODEL is None:
        if _beat_onnx_enabled():
            from app.config import settings
            from app.pipeline.beat_onnx import load_beat_session

            log.info("Loading Beat This! (ONNX, final0)")
            _BEAT_THIS_MODEL = load_beat_session(settings.models_dir, providers=_onnx_providers())
        else:
            import torch
            from beat_this.inference import File2Beats

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading Beat This! (final0) onto %s", device)
            _BEAT_THIS_MODEL = File2Beats(device=device, dbn=False)
    return _BEAT_THIS_MODEL


def park_model() -> None:
    """Drop the cached Beat This! model so /lyrics/align gets a clean GPU.

    Reloaded lazily on the next transcribe; the model is tiny (~78 MB) and
    its weights are disk-cached, so the reload cost is negligible. Mirrors
    the `park_model`/`unpark_model` pair on `adtof_onsets`."""
    global _BEAT_THIS_MODEL
    _BEAT_THIS_MODEL = None


def unpark_model() -> None:
    """No-op: `_beat_this_model` reloads lazily on next use."""


def beat_engine_name() -> str:
    """Which Beat This! backend is currently loaded: 'onnx' | 'torch' | 'none'.
    'none' before the first detection (or after `park_model`). Lets a caller
    (e.g. the desktop e2e) assert ONNX inference actually ran."""
    if _BEAT_THIS_MODEL is None:
        return "none"
    return "onnx" if type(_BEAT_THIS_MODEL).__name__ == "OnnxBeatThis" else "torch"


def _beat_this_beats(audio_path: Path) -> BeatStructure:
    """Beat This! beats + downbeats -> typed BeatStructure.

    Runs on the full mix (Beat This!'s training distribution; the
    `beat_input=full_mix` default). The downstream grid (alignment,
    tempo finalisation, padding) is unchanged: we convert the native
    (beats, downbeats) into the same Nx2 `(time, beat_pos_in_bar)` shape
    the old DBN emitted and reuse `_raw_to_structure`.
    """
    log.info("Beat This!: tracking beats/downbeats in %s", audio_path.name)
    beats, downbeats = _beat_this_model()(str(audio_path))
    downbeats = _recover_bar_length_if_incoherent(beats, downbeats, audio_path)
    raw = _beats_downbeats_to_raw(beats, downbeats)
    if raw.size == 0:
        log.warning("Beat This! returned no beats for %s", audio_path.name)
        return BeatStructure()
    return _raw_to_structure(raw)


def _librosa_fallback(audio_path: Path) -> BeatStructure:
    """Plain librosa beat tracking, no downbeat classification.

    Used only when Beat This! is unavailable. Produces a 4/4 structure with
    constant time signature; tempo follows whatever librosa returned.
    """
    import librosa

    log.info("librosa fallback: tracking beats on %s", audio_path.name)
    audio, sr = librosa.load(str(audio_path), sr=44100, mono=True)
    tempo, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0]) if np.isfinite(tempo) else 120.0
    if not isinstance(beat_times, np.ndarray) or beat_times.size == 0:
        return BeatStructure(initial_tempo=tempo)

    # Group every 4 beats into a 4/4 bar.
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    cur_bar_beats: list[BeatTick] = []
    bar_index = 0
    for i, t in enumerate(beat_times):
        beat_in_bar = (i % 4) + 1
        if beat_in_bar == 1 and cur_bar_beats:
            bars.append(_finalize_bar(bar_index, cur_bar_beats))
            bar_index += 1
            cur_bar_beats = []
        tick = BeatTick(time=float(t), beat_in_bar=beat_in_bar, bar_index=bar_index)
        beats.append(tick)
        cur_bar_beats.append(tick)
    if cur_bar_beats:
        bars.append(_finalize_bar(bar_index, cur_bar_beats))

    return _summarize(beats, bars)
