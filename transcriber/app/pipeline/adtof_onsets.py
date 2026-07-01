"""ADTOF onset backend — per-stem CRNN drum-onset detection.

The sole per-stem onset detector. (The legacy librosa spectral-flux
detector was removed in May 2026; see
`transcriber/docs/ai-midi-to-jot-notes.md` for the techniques captured
from the previous pathway.) We read only the stem's matching class
lane; per-stem identity still comes from MDX23C separation (we do
NOT use ADTOF to classify).

Input source per lane:

* every lane (kick / snare / toms / hihat / cymbals) reads ADTOF off
  its own isolated stem. The drum-stem-substitution path still exists
  (`_DRUM_STEM_INFERENCE_PITCHES`, `drum_stem_path`) but is currently
  empty: hi-hat was moved back to its isolated stem to recover quiet
  hats the full-mix HH lane was missing (a louder coincident kick /
  snare masks a soft hat in the mixed signal; on the isolated stem the
  hat is the dominant transient). The known downside is that open-hat
  sizzle/rattle can read as a phantom hit-train on the isolated stem, the noisy-lane gates (adaptive threshold + prominence + decay-reset)
  exist to suppress exactly that. Identity is unaffected either way: it
  comes from the separated stem; ADTOF only supplies the lane's timing.

Design notes / deliberate constraints:

* This uses the PyTorch port of ADTOF (xavriley/ADTOF-pytorch), not the
  original TensorFlow/Keras ADTOF. It's torch-only and bundles its
  pretrained Frame_RNN weights inside the package, so it's a core dep
  baked into the default image (no build arg, no Zenodo download).

* ADTOF was trained on full mixes; we run it on isolated stems, which
  is out-of-distribution. Failures (missing package, missing weights,
  per-stem inference error) raise — there is no fallback detector
  anymore. The Dockerfile bakes the package + its weights in at build
  time, so this should never fire in production.

* We take ADTOF's dense per-frame activations and run our OWN
  deterministic peak-pick (height + min-distance + prominence) so the
  output is reproducible run-to-run and tuned for the "high-recall,
  the LLM prunes" contract. We deliberately do NOT use the package's
  own `PeakPicker` / `transcribe_to_midi` for this reason.

* After find_peaks gives us activation-domain peak frames, we refine
  each peak's TIME against the audio's onset-strength envelope inside
  a ±N ms window. The NN's activation peak doesn't necessarily sit on
  the actual audio transient (OOD on isolated stems smears the rising
  edge by tens to hundreds of ms; the BiGRU contributes its own
  smearing). Refining against the audio's onset envelope inside a
  tight window pins each onset to the actual transient regardless of
  how messy the activation shape is. The earlier
  `librosa.onset.onset_backtrack` approach was removed because on
  smooth rising edges it would walk all the way back to the previous
  trough (no strict local minima exist in a monotone rise), reporting
  times 150-250 ms before the true transient.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from drumjot_dsp import peakpick  # shared peak-pick algorithm (single source of truth)
from scipy.signal import find_peaks

from app.config import settings
from app.models import OnsetCandidate

log = logging.getLogger(__name__)

# MDX23C stem pitch -> index into ADTOF's 5-class activation output.
# adtof_pytorch's LABELS_5 is [35, 38, 47, 42, 49] (BD, SD, TT, HH,
# CY+RD), so the lane indices are: 0=BD 1=SD 2=TT 3=HH 4=CY+RD. Ride
# (`d`) and crash (`c`) both read the merged CY+RD lane — acceptable
# because identity comes from the stem; ADTOF only contributes
# timing/confidence there. Mapping by lane index (not by parsing the
# MIDI pitch) sidesteps the HH(42)/TT(47) ordering quirk.
_LANE_FOR_PITCH: dict[str, int] = {
    "k": 0,  # kick  -> BD
    "s": 1,  # snare -> SD
    "t": 2,  # toms  -> TT
    "h": 3,  # hihat -> HH
    "d": 4,  # ride  -> CY+RD (merged)
    "c": 4,  # crash -> CY+RD (merged)
}

# The hi-hat energy floor (drop near-silent phantoms; see detect_onsets_adtof)
# normalizes to the median detected-onset amplitude, which is only stable with
# a handful of onsets. Below this count the floor is skipped (keep everything).
_MIN_ONSETS_FOR_AMPLITUDE_FLOOR = 8

# Cymbal lanes (the merged ride/crash `c`, and `d` post-split) measure
# amplitude over a forward "bloom" window instead of the ±20ms attack: a
# crash's loudness is in the wash that peaks ~100ms AFTER the strike, so
# the attack window scores full-volume crashes as quiet. See
# `_bloom_amplitude`. Hats/kick/snare/toms peak AT the strike and keep the
# attack window.
_BLOOM_LANE_PITCHES: frozenset[str] = frozenset({"c", "d"})
_BLOOM_PRE_S = 0.02   # look this far before the onset (catch the attack)
_BLOOM_POST_S = 0.25  # ...and this far after (catch the bloom), capped at
#                       the next onset so it can't leak into the next hit.

# Crash-shadow filter (see _crash_shadow_filter + config
# adtof_cymbal_shadow_louder_mult): drop a cymbal onset that rides the
# decay of a recent much-louder hit without injecting fresh energy.
_SHADOW_WINDOW_S = 1.5       # look back this far for a louder hit
_SHADOW_INJECT_MAX = 0.85    # drop only if RMS isn't rising (ratio < this)
_SHADOW_RMS_HOP = 256        # hop for the injection RMS envelope (~6ms)
# Energy-injection windows (seconds relative to the onset): peak RMS just
# after vs the floor just before. A fresh strike jumps up (>1); a
# re-trigger on a decay does not (<1).
_SHADOW_INJ_POST_S = 0.060
_SHADOW_INJ_PRE_LO_S = 0.070
_SHADOW_INJ_PRE_HI_S = 0.015

# adtof_pytorch's Frame_RNN emits activations on a fixed 100 fps grid
# (the package's hardcoded default; it exposes no per-model rate). The
# peak-pick converts frame index -> seconds with this constant.
_ADTOF_FPS = 100.0

# Lanes whose activation is OOD-compressed on an isolated stem: hihat
# and the merged ride/crash cymbal lane. These get the adaptive per-stem
# threshold + the wider min-distance / prominence / decay-reset post-
# filter. Kick (k) / snare (s) / toms (t) lanes stay on the fixed
# high-recall threshold; their transients are well-defined even on
# isolated stems, so the "detect hot, LLM prunes" contract still holds
# there. The median-of-non-silent input normalization is applied to ALL
# stems uniformly (see `_median_scale_factor`), not just these.
_NOISY_LANE_PITCHES: frozenset[str] = frozenset({"h", "d", "c"})

# Lanes whose ADTOF inference runs on the FULL drum mix
# (`drum_stem_path`) rather than the isolated per-instrument stem.
# Currently EMPTY: every lane reads its own isolated stem.
#
# History: hi-hat was routed through the drum mix because ADTOF is
# in-distribution on a full kit and the isolated hat stem's open-hat
# sizzle/decay tail can trigger phantom hit-trains. That was reverted
# because the full-mix HH lane was MISSING real hits, a louder
# coincident kick/snare masks a soft hat in the mixed signal, so quiet
# hats produced no HH activation at all (a recall loss no peak-pick
# parameter can recover, since the peak isn't there). On the isolated
# stem the hat is the dominant transient and those hits come back; the
# noisy-lane gates handle the open-hat over-trigger risk. Re-add "h"
# here to restore the drum-mix path if over-triggering returns.
#
# Cymbals (`c`) were also tried on the drum-mix path and pulled off:
# kick/snare bleed inside the drum mix routinely activated the cymbal
# lane at frames where the cymbal itself is silent (verified by
# checking `stem_c.mp3` plays nothing at the detected time); see the
# user-reported "phantom crash at bar 5" case.
_DRUM_STEM_INFERENCE_PITCHES: frozenset[str] = frozenset()


def _resolve_device() -> str:
    """Map `settings.device` onto adtof_pytorch's "cuda"/"cpu" choice.

    The package only accepts those two strings. `cpu`/`mps` -> "cpu"
    (there is no MPS path here); anything else asks for CUDA but falls
    back to CPU if torch reports no GPU, mirroring the package's own
    `transcribe_to_midi` behaviour.
    """
    import torch  # local: keep module import torch-free for the librosa default

    pref = (settings.device or "auto").lower()
    if pref in ("cpu", "mps"):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _onnx_enabled() -> bool:
    """Default ON: run the Frame_RNN through onnxruntime (torch-free inference).
    Opt out with DRUMJOT_ONSET_ONNX in {0,false,no,off,torch} (shared with the
    learned-onset backend, mirrors DRUMJOT_SEP_ONNX for separation)."""
    import os

    return os.environ.get("DRUMJOT_ONSET_ONNX", "1").strip().lower() not in (
        "0", "false", "no", "off", "torch",
    )


def _onnx_providers():
    """onnxruntime providers from settings.device (no torch import): CPU-pinned
    when CPU/MPS is forced, else the available set (+ CPU fallback in the loader)."""
    dev = (settings.device or "auto").lower()
    return ["CPUExecutionProvider"] if dev in ("cpu", "mps") else None


@lru_cache(maxsize=1)
def _load_adtof_session():
    """Cached ADTOF onnxruntime session; exports + caches the `.onnx` once."""
    from app.pipeline.adtof_onnx import load_adtof_session

    return load_adtof_session(settings.models_dir, providers=_onnx_providers())


@lru_cache(maxsize=1)
def _load_model():
    """Load the ADTOF model once per process and cache (lazy singleton).

    Like `beats._beat_this_model`: multi-second load, so we pay
    it once. All adtof_pytorch/torch imports happen HERE, never at
    module top, so the librosa-default path never imports them and a
    broken install degrades to librosa instead of failing this import.

    Returns `(model, device)`. Raises a clear error (caught by the
    caller, which then falls back to librosa) when adtof_pytorch is not
    installed or its bundled weights are absent.

    NOTE: the `adtof_pytorch` symbols used here are its documented
    public API (its `__all__`), pinned via the commit SHA in
    pyproject.toml. If a future port changes these signatures,
    `_load_model` raises and the caller falls back to librosa; verify
    against the pinned build at image-build time.
    """
    try:
        import torch
        from adtof_pytorch import (
            calculate_n_bins,
            create_frame_rnn_model,
            get_default_weights_path,
            load_pytorch_weights,
        )
    except ImportError as exc:  # adtof_pytorch / torch not importable
        raise RuntimeError(
            "ADTOF onset backend is the only detector but "
            "`adtof_pytorch` is not importable. It is a core dependency "
            "baked into the image; rebuild the image."
        ) from exc

    weights_path = get_default_weights_path()
    if not weights_path or not Path(weights_path).exists():
        raise FileNotFoundError(
            "adtof_pytorch is installed but its bundled Frame_RNN "
            f"weights are missing (get_default_weights_path()={weights_path!r}). "
            "The wheel should ship them; rebuild the image from the "
            "pinned commit."
        )

    device = _resolve_device()
    # Determinism: no dropout at inference + a fixed seed so the forward
    # pass is reproducible run-to-run for the A/B against librosa.
    torch.manual_seed(0)

    log.info("Loading adtof_pytorch Frame_RNN (device=%s) from %s", device, weights_path)
    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    model = load_pytorch_weights(model, str(weights_path), strict=False)
    model.to(device)
    return model, device


def park_model() -> None:
    """Free the ADTOF model's GPU memory for the /lyrics swap. Callers must hold
    the process-wide GPU lock; see `app.pipeline.gpu_park`.

    ONNX path: drop the onnxruntime session (its arena holds the VRAM); it
    reloads lazily from the cached `.onnx`. Torch path: move the module to CPU."""
    _load_adtof_session.cache_clear()  # ORT session VRAM; reloads from the cached .onnx
    if _load_model.cache_info().currsize == 0:
        return
    from app.pipeline.gpu_park import park_module

    model, _ = _load_model()
    park_module(model, "adtof")


def unpark_model() -> None:
    """Move the cached ADTOF Frame_RNN model back to CUDA. No-op when
    the lru_cache hasn't been hit yet."""
    if _load_model.cache_info().currsize == 0:
        return
    from app.pipeline.gpu_park import unpark_module

    model, _ = _load_model()
    unpark_module(model, "adtof")


@lru_cache(maxsize=1)
def _audio_processor():
    """Cached ADTOF `AudioProcessor` (filterbank built once per process).

    The package's `load_audio_for_model` constructs a fresh processor on
    every call, rebuilding the filterbank each time. Holding a singleton
    here saves that cost across stems and lets us call `compute_stft` /
    `apply_filterbank` directly on an in-memory array instead of going
    through the package's path-based loader (which would force us to
    write a temp file just to normalize amplitude).
    """
    from adtof_pytorch.audio import create_adtof_processor

    return create_adtof_processor()


def _load_mono_audio(audio_path: Path) -> np.ndarray:
    """Load `audio_path` as mono float32 at the ADTOF processor's rate."""
    proc = _audio_processor()
    y, _sr = librosa.load(str(audio_path), sr=proc.sample_rate, mono=True)
    return y.astype(np.float32, copy=False)


def _peak_amplitude(
    audio: np.ndarray, peak_time_sec: float, sample_rate: int,
    *, window_sec: float = 0.020,
) -> float:
    """Maximum |sample| in a ±`window_sec` window around `peak_time_sec`.

    Returns the raw audio amplitude (in [0, 1] sample units) at the
    onset, the loudness proxy that drives the per-pitch
    percentile-normalised velocity mapping in `onsets_midi.py`. Peak
    rather than RMS so the attack transient dominates the value
    (drum hits are mostly transient energy, and RMS would average
    in decay tail); ±20ms is long enough to catch the full attack
    without leaking into the next hit's onset (any reasonable drum
    spacing is >40ms apart at typical tempos).

    Used for lanes that peak AT the strike (hat / kick / snare / toms).
    Cymbal lanes bloom after the strike and use `_bloom_amplitude`.
    """
    half = int(round(window_sec * sample_rate))
    center = int(round(peak_time_sec * sample_rate))
    lo = max(0, center - half)
    hi = min(audio.size, center + half + 1)
    if hi <= lo:
        return 0.0
    return float(np.max(np.abs(audio[lo:hi])))


def _bloom_amplitude(
    audio: np.ndarray,
    peak_time_sec: float,
    next_time_sec: float,
    sample_rate: int,
    *,
    pre_sec: float = _BLOOM_PRE_S,
    post_sec: float = _BLOOM_POST_S,
) -> float:
    """Maximum |sample| over a forward window `[peak - pre, peak + post]`,
    capped at `next_time_sec` so it can't reach into the next hit.

    For cymbals this is the correct loudness proxy: a crash's energy is in
    the wash that blooms ~100ms after the strike, so the ±20ms attack
    window (`_peak_amplitude`) scores full-volume crashes as quiet. The
    forward window catches the bloom. Measured on itte: this lifts the real
    crashes that the attack window under-read (a 1:46 crash from 0.50x to
    1.42x of median) while leaving silent phantoms low (<=0.06x), restoring
    a clean energy-floor separation. Drives both the floor and velocity for
    cymbal lanes.
    """
    lo = max(0, int(round((peak_time_sec - pre_sec) * sample_rate)))
    end = min(peak_time_sec + post_sec, next_time_sec)
    hi = min(audio.size, int(round(end * sample_rate)))
    if hi <= lo:
        return 0.0
    return float(np.max(np.abs(audio[lo:hi])))


def _median_scale_factor(audio: np.ndarray) -> float:
    """Per-track amplitude scale mirroring the frontend waveform's
    `computeTrackAmpScale`: stride ~10k samples, take the median of
    |sample| above `silence_floor`, return `target / median` clamped to
    [0.25, 25].

    Robust to bleed spikes (median ignores the tail) and to separator
    output-gain variance, so the fixed peak threshold stays meaningful
    across stems. Returns 1.0 when the stem is silent or too short to
    take a median.
    """
    n = audio.size
    if n == 0:
        return 1.0
    floor = settings.adtof_median_silence_floor
    target = settings.adtof_median_target
    stride = max(1, n // 10000)
    sampled = np.abs(audio[::stride])
    nonsilent = sampled[sampled > floor]
    if nonsilent.size == 0:
        return 1.0
    median = float(np.median(nonsilent))
    if median <= 0.0:
        return 1.0
    return float(np.clip(target / median, 0.25, 25.0))


def _features(audio: np.ndarray) -> np.ndarray:
    """`[1, T, n_bins, 1]` log-filterbank features for an in-memory mono array (at
    the processor's rate). Bypasses `load_audio_for_model` so a median-scaled
    array feeds in without a temp wav. Torch-free (numpy STFT + filterbank)."""
    proc = _audio_processor()
    stft = proc.compute_stft(audio)
    filtered = proc.apply_filterbank(stft).T.astype(np.float32, copy=False)
    # AudioProcessor.process_audio adds a trailing channel dim of size 1 for the
    # mono case; the model's conv layers expect [batch, time, freq, channels].
    return filtered[np.newaxis, :, :, np.newaxis]


def _adtof_activations(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """ADTOF dense per-frame activations `(frames, 5)` + fps for `audio`.

    Runs the Frame_RNN through onnxruntime (default, torch-free) or torch
    (DRUMJOT_ONSET_ONNX=0). The frontend is numpy either way.
    """
    x = _features(audio)
    if _onnx_enabled():
        sess = _load_adtof_session()
        pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]  # [1, frames, classes]
    else:
        import torch

        model, device = _load_model()
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    acts = np.asarray(pred, dtype=np.float64)
    if acts.ndim == 3 and acts.shape[0] == 1:
        acts = acts[0]
    if acts.ndim != 2 or acts.shape[1] < 5:
        raise RuntimeError(
            f"Unexpected ADTOF activation shape {np.asarray(pred).shape} "
            f"(expected (1, frames, >=5))."
        )
    return acts, _ADTOF_FPS


def _resolve_threshold(pitch: str, activation: np.ndarray) -> float:
    """Pick the peak-pick height for `pitch`'s activation lane.

    Noisy lanes (`_NOISY_LANE_PITCHES`) use an adaptive threshold scaled
    to this stem's own activation distribution, `max(floor, k * pXX)`, so it self-calibrates to the lane's confidence range instead of
    assuming an absolute scale that doesn't hold OOD. All other lanes use
    the fixed global `adtof_peak_threshold`.

    Hi-hat uses a LOWER floor (`adtof_hihat_adaptive_threshold_floor`): the
    ~14 kHz band-limit starves ADTOF's HH activation, so the cymbal-tuned
    floor culls real hits. The adaptive `k * pXX` term is shared.
    """
    fixed = settings.adtof_peak_threshold
    if (
        pitch not in _NOISY_LANE_PITCHES
        or not settings.adtof_adaptive_threshold
        or activation.size == 0
    ):
        return fixed
    pxx = float(
        np.percentile(activation, settings.adtof_adaptive_threshold_pct)
    )
    floor = (
        settings.adtof_hihat_adaptive_threshold_floor
        if pitch == "h"
        else settings.adtof_adaptive_threshold_floor
    )
    return max(floor, settings.adtof_adaptive_threshold_k * pxx)


def _decay_reset_filter(
    activation: np.ndarray,
    peaks: np.ndarray,
    reset_frac: float,
    reset_floor: float,
) -> np.ndarray:
    """Drop peaks that re-trigger before the previous accepted peak's
    energy decayed — the open-hihat/cymbal "one sustained hit read as a
    stream of 16ths" case that height/prominence structurally can't
    catch (the ring's wobble has real prominence).

    A candidate is kept only if the activation, somewhere between it and
    the previous *accepted* peak, fell below
    `max(reset_floor, reset_frac * prev_peak_height)` — i.e. the prior
    ring actually came back down first. Continuous sustain never dips, so
    it collapses to one onset; genuinely separate hits (the activation
    plunges between them) all survive. `peaks` is assumed time-ordered
    (find_peaks returns ascending indices).
    """
    # Delegates to the shared implementation (drumjot_dsp.peakpick) so the
    # transcriber and the trainer can't drift; kept as a thin wrapper to
    # preserve the call sites + the docstring above.
    return peakpick.decay_reset_filter(activation, peaks, reset_frac, reset_floor)


def _refine_peak_times_audio(
    audio_path: Path,
    peak_times_sec: list[float],
    window_sec: float,
) -> list[float]:
    """Snap each ADTOF peak time to the nearest local maximum of the
    audio's onset-strength envelope within ±`window_sec`.

    Replaces the previous `librosa.onset.onset_backtrack` step. Backtrack
    chased activation-domain strict local minima; on a smooth rising
    edge no such minima exist, so it would jump all the way back to the
    previous onset's trough — reporting times 150-250 ms before the
    actual transient. Refining against the AUDIO's onset envelope in a
    tight window avoids that pathology: the audio's transient is where
    it is, regardless of how the NN activation got there.

    `audio_path` must be the same file ADTOF saw, so the envelope's
    frame times share a coordinate frame with the activation's. A
    librosa load / STFT failure falls back to returning the raw input
    times unchanged, so a broken refinement degrades cleanly instead of
    dropping onsets.
    """
    if not peak_times_sec or window_sec <= 0:
        return [float(t) for t in peak_times_sec]
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        log.warning(
            "Audio-domain refine: librosa.load(%s) failed (%s); "
            "using raw peak times.", audio_path.name, exc,
        )
        return [float(t) for t in peak_times_sec]
    if y.size == 0:
        return [float(t) for t in peak_times_sec]
    # Hop short enough that the per-peak window resolves to ~1 ms slots.
    # At sr=44100, hop=64 → ~1.45 ms/frame; tighter than the transient
    # itself, so the local-max-in-window snap is sample-honest.
    hop_length = 64
    try:
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length,
        )
    except Exception as exc:
        log.warning(
            "Audio-domain refine: onset_strength(%s) failed (%s); "
            "using raw peak times.", audio_path.name, exc,
        )
        return [float(t) for t in peak_times_sec]
    if onset_env.size == 0:
        return [float(t) for t in peak_times_sec]
    env_fps = float(sr) / float(hop_length)
    half_w = max(1, int(round(window_sec * env_fps)))
    n_frames = int(onset_env.size)
    refined: list[float] = []
    for t in peak_times_sec:
        center = int(round(float(t) * env_fps))
        lo = max(0, center - half_w)
        hi = min(n_frames, center + half_w + 1)
        if lo >= hi:
            refined.append(float(t))
            continue
        local_max_idx = lo + int(np.argmax(onset_env[lo:hi]))
        refined.append(float(local_max_idx) / env_fps)
    return refined


def _audio_onset_frames(
    audio_path: Path,
    fps: float,
    *,
    delta: float,
    wait_s: float,
    min_strength_mult: float,
) -> np.ndarray:
    """Detect onsets straight from the stem audio (librosa onset-strength
    peak-pick), returned as activation-frame indices at `fps`.

    For the hi-hat lane: the ~14 kHz band-limit starves ADTOF's HH
    activation, so it never peaks on many real hits, and no peak-pick
    threshold can recover a peak that isn't there. The isolated hat stem's
    own audio transients can.

    Open-hat sizzle would otherwise over-segment a single ring into a
    stream of phantom 16ths, so only peaks whose onset-strength value is
    >= `min_strength_mult` * the stem's median onset-strength are kept (a
    real strike spikes; sizzle ripples low). Returns an empty array on any
    load/STFT failure so the caller degrades cleanly to ADTOF-only.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        log.warning(
            "hihat audio supplement: load(%s) failed (%s); ADTOF-only.",
            audio_path.name, exc,
        )
        return np.empty(0, dtype=int)
    if y.size == 0:
        return np.empty(0, dtype=int)
    hop = 512
    try:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        wait = max(1, int(round(wait_s * sr / hop)))
        frames = librosa.util.peak_pick(
            env, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
            delta=delta, wait=wait,
        )
    except Exception as exc:
        log.warning(
            "hihat audio supplement: onset detect failed (%s); ADTOF-only.",
            exc,
        )
        return np.empty(0, dtype=int)
    frames = np.asarray(frames, dtype=int)
    if frames.size and min_strength_mult > 0.0:
        positive = env[env > 0.0]
        if positive.size:
            floor = min_strength_mult * float(np.median(positive))
            frames = frames[env[frames] >= floor]
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    return np.round(np.asarray(times) * fps).astype(int)


def _merge_audio_onsets(
    adtof_peaks: np.ndarray,
    n_frames: int,
    audio_frames: np.ndarray,
    dedup_frames: int,
) -> np.ndarray:
    """Union audio-derived onset frames into the ADTOF peak set.

    Drops audio frames within `dedup_frames` of an existing ADTOF peak
    (same physical hit) and any out-of-range frame. Returns sorted unique
    frame indices. Audio-only survivors carry no ADTOF confidence of their
    own; the caller reads `activation[frame]` for their `strength` (low
    where ADTOF was unsure, honest).
    """
    if audio_frames.size == 0:
        return adtof_peaks
    existing = np.sort(adtof_peaks)
    kept = list(int(p) for p in adtof_peaks)
    for raw in audio_frames:
        fr = int(raw)
        if fr < 0 or fr >= n_frames:
            continue
        if existing.size and int(np.min(np.abs(existing - fr))) <= dedup_frames:
            continue
        kept.append(fr)
    return np.array(sorted(set(kept)), dtype=int)


def _apply_amplitude_floor(
    candidates: list[OnsetCandidate],
    frac: float,
    min_onsets: int,
) -> tuple[list[OnsetCandidate], int]:
    """Drop onsets whose `amplitude` is below `frac` * the median onset
    amplitude, near-silent phantoms on the noise floor / a previous hit's
    decay, where a real strike sits ~1x the median. Returns
    `(kept, n_dropped)`.

    No-op when `frac <= 0`, when there are fewer than `min_onsets` (the
    median is unstable), or when no candidate carries an amplitude.
    Candidates with `amplitude is None` are never dropped (no signal to
    judge them on).
    """
    if frac <= 0.0 or len(candidates) < min_onsets:
        return candidates, 0
    amps = [c.amplitude for c in candidates if c.amplitude is not None]
    if not amps:
        return candidates, 0
    floor = frac * float(np.median(amps))
    kept = [
        c for c in candidates
        if c.amplitude is None or c.amplitude >= floor
    ]
    return kept, len(candidates) - len(kept)


def _energy_injection(rms: np.ndarray, rms_t: np.ndarray, t: float) -> float:
    """Ratio of peak RMS just after `t` to the RMS floor just before it.

    A fresh drum strike injects energy: the post-onset peak jumps above the
    pre-onset floor (ratio >> 1). An onset that merely rides a decaying tail
    (a crash sustain re-triggering the detector) has no fresh energy, so the
    RMS is flat or falling through it (ratio < 1). Returns `inf` for a
    rise out of silence and `0.0` when there is no energy at all.
    """
    post = (rms_t >= t - 0.005) & (rms_t <= t + _SHADOW_INJ_POST_S)
    pre = (rms_t >= t - _SHADOW_INJ_PRE_LO_S) & (rms_t <= t - _SHADOW_INJ_PRE_HI_S)
    peak = float(rms[post].max()) if np.any(post) else 0.0
    base = float(np.median(rms[pre])) if np.any(pre) else 0.0
    if base <= 1e-6:
        return float("inf") if peak > 1e-6 else 0.0
    return peak / base


def _crash_shadow_filter(
    candidates: list[OnsetCandidate],
    audio: np.ndarray,
    sample_rate: int,
    window_s: float,
    louder_mult: float,
    inject_max: float,
) -> tuple[list[OnsetCandidate], int]:
    """Drop onsets that ride the decay of a recent much-louder hit without
    injecting fresh energy: a crash's sustain re-triggering the detector.

    A candidate is dropped only when BOTH hold: an earlier onset within
    `window_s` is at least `louder_mult` times louder (by `amplitude`), AND
    its energy injection (`_energy_injection`) is below `inject_max` (the
    RMS isn't rising, so it is not a fresh strike). Requiring both spares a
    real soft hit (it injects energy) and a dense ride stream (its
    neighbours are the same loudness, so nothing casts a shadow). `amplitude`
    must be the bloom amplitude (cymbal lanes); candidates assumed
    time-ordered. Returns `(kept, n_dropped)`.

    No-op when disabled (`louder_mult <= 0`) or with fewer than 2 onsets.
    """
    if louder_mult <= 0.0 or len(candidates) < 2:
        return candidates, 0
    rms = librosa.feature.rms(y=audio, hop_length=_SHADOW_RMS_HOP)[0]
    rms_t = librosa.times_like(rms, sr=sample_rate, hop_length=_SHADOW_RMS_HOP)
    kept: list[OnsetCandidate] = []
    dropped = 0
    for i, c in enumerate(candidates):
        if c.amplitude is None:
            kept.append(c)
            continue
        if _energy_injection(rms, rms_t, float(c.time)) >= inject_max:
            kept.append(c)
            continue
        in_shadow = any(
            0.0 < c.time - p.time <= window_s
            and p.amplitude is not None
            and p.amplitude >= louder_mult * c.amplitude
            for p in candidates[:i]
        )
        if in_shadow:
            dropped += 1
        else:
            kept.append(c)
    return kept, dropped


def detect_onsets_adtof(
    audio_path: Path,
    pitch: str,
    sample_rate: int = 44100,
    *,
    drum_stem_path: Path | None = None,
) -> list[OnsetCandidate]:
    """ADTOF onsets for one stem, reading only the stem's class lane.

    `audio_path` is the isolated per-instrument stem (identity source).
    For the lanes in `_DRUM_STEM_INFERENCE_PITCHES` (today: hi-hat only)
    inference instead runs on `drum_stem_path` when supplied and present. ADTOF is in-distribution on a full drum mix, so the open-hat
    sizzle stops being read as a phantom hit-train. Kick / snare / toms
    / cymbals always use their own stem: cymbals specifically were
    pulled off the drum-mix path after the user-reported "phantom crash
    at bar 5" case where kick / snare bleed inside the full mix
    activated the cymbal lane at a moment the isolated cymbal stem was
    silent. When `drum_stem_path` is missing the drum-mix lanes fall
    back to the isolated stem.

    Returns `OnsetCandidate`s with the ADTOF sigmoid activation in
    [0, 1] as `strength` — a per-frame model-confidence proxy. It feeds
    the filter LLM's `(beat_in_bar, strength)` prompt block and the
    velocity-mapping percentile lookup in `onsets_to_midi_bytes`.

    Every stem is amplitude-normalized in memory (median-of-non-silent
    target) before inference; the noisy lanes additionally use an
    adaptive peak threshold computed from their activation distribution.
    """
    lane = _LANE_FOR_PITCH.get(pitch)
    if lane is None:
        # Unknown/custom stem pitch: nothing ADTOF can speak to.
        return []

    is_noisy = pitch in _NOISY_LANE_PITCHES
    use_drum_stem = pitch in _DRUM_STEM_INFERENCE_PITCHES
    # `_DRUM_STEM_INFERENCE_PITCHES` lanes prefer the in-distribution
    # full drum mix; everything else (and the fallback when no drum
    # stem is cached) uses the isolated stem. Cymbals were tried on the
    # drum-mix path and pulled off: kick/snare bleed inside the full mix
    # routinely activated the cymbal lane at frames the isolated cymbal
    # stem was silent. `source_path` is what ADTOF actually sees.
    source_path = audio_path
    used_drum_stem = False
    if (
        use_drum_stem
        and drum_stem_path is not None
        and drum_stem_path.exists()
    ):
        source_path = drum_stem_path
        used_drum_stem = True

    audio = _load_mono_audio(source_path)
    sr = _audio_processor().sample_rate
    # Keep a reference to the unscaled audio so per-onset amplitude
    # reflects the source stem's raw loudness, not the model-input
    # normalisation. The scaling factor below is uniform per stem so
    # relative ordering would be preserved either way, but raw values
    # are more meaningful in the per-note debug popup.
    audio_unscaled = audio
    scale = 1.0
    if settings.adtof_median_normalize and audio.size:
        scale = _median_scale_factor(audio)
        if scale != 1.0:
            audio = np.clip(audio * scale, -1.0, 1.0).astype(np.float32, copy=False)

    acts, fps = _adtof_activations(audio)
    activation = acts[:, lane]
    if activation.size == 0:
        return []

    threshold = _resolve_threshold(pitch, activation)
    if pitch == "h":
        # Hi-hat: looser min-distance than the cymbal-tuned noisy default
        # (fast hat patterns + band-limit weak activation; see config).
        min_dist_s = settings.adtof_hihat_peak_min_distance_s
    elif is_noisy:
        min_dist_s = settings.adtof_noisy_peak_min_distance_s
    else:
        min_dist_s = settings.adtof_peak_min_distance_s
    min_distance_frames = max(1, round(min_dist_s * fps))
    # Prominence gate, now applied to ALL lanes (was previously
    # noisy-only). Prominence rejects decay-tail wobbles that clear
    # `height` but don't rise above their local baseline; the
    # primary OOD failure mode on isolated kick/snare/tom stems
    # where the activation stays elevated for 100-300 ms after a
    # real hit. Cymbals keep the higher value tuned for plateau
    # ripples; hi-hat uses a lower value (band-limit weak activation);
    # the universal floor catches the snare/kick/tom case.
    # None = disabled (setting 0.0).
    if pitch == "h":
        prominence_val = settings.adtof_hihat_peak_prominence
    elif is_noisy:
        prominence_val = settings.adtof_noisy_peak_prominence
    else:
        prominence_val = settings.adtof_peak_prominence
    prominence = prominence_val if prominence_val > 0.0 else None
    peaks, _props = find_peaks(
        activation,
        height=threshold,
        distance=min_distance_frames,
        prominence=prominence,
    )

    reset_removed = 0
    reset_frac = settings.adtof_noisy_decay_reset_frac
    if is_noisy and reset_frac > 0.0:
        n_pre = peaks.size
        peaks = _decay_reset_filter(
            activation,
            peaks,
            reset_frac,
            settings.adtof_noisy_decay_reset_floor,
        )
        reset_removed = n_pre - int(peaks.size)

    # Audio-domain supplement (hi-hat only): union onsets detected straight
    # from the stem audio into the ADTOF peak set. The band-limit starves
    # ADTOF's HH activation, so it misses real hits no peak-pick gate can
    # recover; the clean isolated stem's transients fill the gap. High-recall
    # by design, the split + filter LLM prune.
    audio_added = 0
    if pitch == "h" and settings.adtof_hihat_audio_supplement:
        n_pre = int(peaks.size)
        peaks = _merge_audio_onsets(
            peaks,
            int(activation.size),
            _audio_onset_frames(
                source_path,
                fps,
                delta=settings.adtof_hihat_audio_supplement_delta,
                wait_s=settings.adtof_hihat_audio_supplement_wait_s,
                min_strength_mult=(
                    settings.adtof_hihat_audio_supplement_min_strength_mult
                ),
            ),
            dedup_frames=max(
                1,
                round(settings.adtof_hihat_audio_supplement_dedup_s * fps),
            ),
        )
        audio_added = int(peaks.size) - n_pre

    # Refine each peak's reported time against the AUDIO's onset
    # envelope inside a tight window. `strength` is still read at the
    # model's peak frame because that's where the network's
    # confidence lives, but `time` reflects where the actual audio
    # transient sits. Refine against the on-disk source (un-scaled);
    # constant-amplitude scaling doesn't shift the position of local
    # maxima in the onset-strength envelope, so the refinement result
    # is unchanged.
    peak_frames = [int(p) for p in peaks]
    peak_times_sec = [float(p) / fps for p in peak_frames]
    refined_times = _refine_peak_times_audio(
        source_path,
        peak_times_sec,
        window_sec=settings.adtof_audio_refine_window_s,
    )

    # Peak audio amplitude (|sample|) around each refined peak time,
    # sampled from the unscaled stem so the value reflects raw stem
    # loudness. Drives the per-pitch percentile-normalised velocity mapping
    # in `onsets_midi.py` and the energy floor below. Cymbal lanes use a
    # forward "bloom" window (crashes peak ~100ms after the strike); all
    # other lanes use the ±20ms attack window (they peak at the strike).
    if pitch in _BLOOM_LANE_PITCHES:
        amplitudes = [
            _bloom_amplitude(
                audio_unscaled,
                t,
                refined_times[i + 1] if i + 1 < len(refined_times)
                else t + _BLOOM_POST_S,
                sr,
            )
            for i, t in enumerate(refined_times)
        ]
    else:
        amplitudes = [
            _peak_amplitude(audio_unscaled, t, sr)
            for t in refined_times
        ]

    # `time` is the post-envelope-refine value (where the audio transient
    # actually sits); `raw_model_time` carries the pre-refine
    # `peak_frame / fps` so the per-note debug popup can surface the
    # envelope refinement as its own stage in the detected → final chain.
    candidates = [
        OnsetCandidate(
            time=refined_time,
            raw_model_time=raw_time,
            strength=float(activation[peak_idx]),
            amplitude=amplitude,
            bar=-1,
            beat_in_bar=-1.0,
        )
        for peak_idx, raw_time, refined_time, amplitude in zip(
            peak_frames, peak_times_sec, refined_times, amplitudes,
            strict=False,
        )
    ]

    # Energy floor: drop near-silent phantom onsets whose audio peak barely
    # rises above the noise floor. Catches phantoms at the source, before
    # they reach the split (where a near-zero peak makes the hat's `pre_rms`
    # explode, or a silent cymbal onset gets mislabelled ride/crash). The
    # hat floors on the ±20ms attack amplitude; the cymbal floors on the
    # bloom amplitude (set above) with its own, lower fraction.
    amp_dropped = 0
    floor_frac = 0.0
    if pitch == "h":
        floor_frac = settings.adtof_hihat_min_amplitude_frac
    elif pitch in _BLOOM_LANE_PITCHES:
        floor_frac = settings.adtof_cymbal_min_amplitude_frac
    if floor_frac > 0.0:
        candidates, amp_dropped = _apply_amplitude_floor(
            candidates, floor_frac, _MIN_ONSETS_FOR_AMPLITUDE_FLOOR,
        )

    # Crash-shadow filter (cymbal lanes): drop sustain re-triggers riding a
    # louder crash's decay. Real energy, so the amplitude floor misses them.
    shadow_dropped = 0
    shadow_mult = (
        settings.adtof_cymbal_shadow_louder_mult
        if pitch in _BLOOM_LANE_PITCHES
        else 0.0
    )
    if shadow_mult > 0.0:
        candidates, shadow_dropped = _crash_shadow_filter(
            candidates, audio_unscaled, sr,
            _SHADOW_WINDOW_S, shadow_mult, _SHADOW_INJECT_MAX,
        )

    log.info(
        "ADTOF: %d onsets in %s (lane=%d, src=%s, thr=%.3f%s, prom=%s, "
        "dist=%.0fms, reset=%s, audio_suppl=%s, amp_floor=%s, shadow=%s, "
        "med_norm=%s, refine=%.0fms, median strength=%.3f)",
        len(candidates),
        audio_path.name,
        lane,
        "drum_stem" if used_drum_stem else "stem",
        threshold,
        " adaptive"
        if is_noisy and settings.adtof_adaptive_threshold
        else " fixed",
        f"{prominence:.2f}" if prominence is not None else "off",
        min_dist_s * 1000.0,
        f"{reset_frac:.2f}(-{reset_removed})"
        if is_noisy and reset_frac > 0.0
        else "off",
        f"+{audio_added}"
        if pitch == "h" and settings.adtof_hihat_audio_supplement
        else "off",
        f"-{amp_dropped}" if floor_frac > 0.0 else "off",
        f"-{shadow_dropped}" if shadow_mult > 0.0 else "off",
        f"x{scale:.2f}" if scale != 1.0 else "off",
        settings.adtof_audio_refine_window_s * 1000.0,
        float(np.median([c.strength for c in candidates]))
        if candidates
        else 0.0,
    )
    return candidates


# ADTOF lane index -> (scoring lane name, pitch key for peak-pick params).
# The merged ride+crash lane (4) becomes `cy`; we use `c` for its params so
# it gets the noisy-lane (adaptive threshold + decay-reset) treatment.
_SCORING_LANES_FROM_ADTOF: tuple[tuple[str, int, str], ...] = (
    ("k", 0, "k"),
    ("s", 1, "s"),
    ("t", 2, "t"),
    ("h", 3, "h"),
    ("cy", 4, "c"),
)


def _peak_pick_lane(activation: np.ndarray, pitch: str, fps: float) -> np.ndarray:
    """Peak-pick one activation lane with `pitch`'s standard parameters
    (height/min-distance/prominence, plus the decay-reset filter for noisy
    lanes). Returns ascending peak frame indices. Shared by the all-lanes
    detector; the per-stem `detect_onsets_adtof` keeps its own copy because
    it also tracks the reset-removed count for logging."""
    is_noisy = pitch in _NOISY_LANE_PITCHES
    threshold = _resolve_threshold(pitch, activation)
    min_dist_s = (
        settings.adtof_noisy_peak_min_distance_s
        if is_noisy
        else settings.adtof_peak_min_distance_s
    )
    min_distance_frames = max(1, round(min_dist_s * fps))
    prominence_val = (
        settings.adtof_noisy_peak_prominence
        if is_noisy
        else settings.adtof_peak_prominence
    )
    prominence = prominence_val if prominence_val > 0.0 else None
    peaks, _props = find_peaks(
        activation, height=threshold, distance=min_distance_frames, prominence=prominence
    )
    reset_frac = settings.adtof_noisy_decay_reset_frac
    if is_noisy and reset_frac > 0.0:
        peaks = _decay_reset_filter(
            activation, peaks, reset_frac, settings.adtof_noisy_decay_reset_floor
        )
    return peaks


def detect_all_lanes_adtof(drum_stem_path: Path) -> dict[str, list[float]]:
    """Per-lane onset seconds for all five scoring lanes from ONE ADTOF
    inference on a drum stem.

    Unlike `detect_onsets_adtof` (one isolated stem -> one lane), this reads
    every lane off a single full-drum-mix inference, the in-distribution use
    ADTOF was trained for, so the alignment scorer needs no per-instrument
    separation. The merged ride+crash lane is returned as `cy`. Peak times
    are refined against the drum stem's onset envelope in one pass. Intended
    audio is a drum stem (a separated `stems_all` drum stem, or a ParaDB
    pack's drums-only track)."""
    audio = _load_mono_audio(drum_stem_path)
    if settings.adtof_median_normalize and audio.size:
        scale = _median_scale_factor(audio)
        if scale != 1.0:
            audio = np.clip(audio * scale, -1.0, 1.0).astype(np.float32, copy=False)

    acts, fps = _adtof_activations(audio)

    frames_by_lane: dict[str, list[int]] = {}
    for lane_name, idx, pitch in _SCORING_LANES_FROM_ADTOF:
        if idx >= acts.shape[1]:
            continue
        activation = acts[:, idx]
        if activation.size == 0:
            continue
        frames_by_lane[lane_name] = [int(p) for p in _peak_pick_lane(activation, pitch, fps)]

    # Refine every lane's times against the drum stem in a single
    # onset-envelope pass (pay the librosa cost once, not per lane).
    flat = [(lane_name, frame) for lane_name, frames in frames_by_lane.items() for frame in frames]
    refined = _refine_peak_times_audio(
        drum_stem_path,
        [frame / fps for _lane, frame in flat],
        window_sec=settings.adtof_audio_refine_window_s,
    )

    out: dict[str, list[float]] = {lane_name: [] for lane_name in frames_by_lane}
    for (lane_name, _frame), t in zip(flat, refined, strict=False):
        out[lane_name].append(t)
    for times in out.values():
        times.sort()
    log.info(
        "ADTOF all-lanes: %s onsets from %s",
        {ln: len(ts) for ln, ts in out.items()},
        drum_stem_path.name,
    )
    return out
