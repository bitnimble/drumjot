"""ADTOF onset backend — per-stem CRNN drum-onset detection.

The sole per-stem onset detector. (The legacy librosa spectral-flux
detector was removed in May 2026; see
`transcriber/docs/ai-midi-to-jot-notes.md` for the techniques captured
from the previous pathway.) We read only the stem's matching class
lane; per-stem identity still comes from MDX23C separation (we do
NOT use ADTOF to classify).

Input source per lane:

* kick / snare / toms read ADTOF off their own isolated stem.
* hihat + the merged cymbal lane (`_NOISY_LANE_PITCHES`) instead read
  ADTOF off the **full drum stem** when it's available. ADTOF was
  trained on full drum mixes; given an isolated hihat stem it reads the
  open-hat sizzle/rattle as a confident train of ~16th-rate phantom
  hits (an OOD failure no peak-pick parameter can undo, because the
  spurious events are strong, well-separated and prominent). Running it
  on the in-distribution drum stem and reading the HH/CY lane restores
  the behaviour it was built for. Identity is unaffected — it still
  comes from the separated stem; ADTOF only supplies the lane's timing.
  Falls back to the isolated stem when the drum stem isn't on disk
  (e.g. resume-from-onsets without it cached).

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
# ADTOF is in-distribution on a full kit mix and on the isolated hi-hat
# stem the open-hat sizzle and decay tail trigger phantom hit-trains; # routing hi-hat through the drum mix fixes that without losing real
# hits (snare bleed in the hat stem can't fool the network when it's
# scoring a full mix).
#
# Cymbals (`c`) are deliberately NOT on this list: the drum-mix path
# was tried for them too, and kick/snare bleed inside the drum mix
# routinely activated the cymbal lane at frames where the cymbal
# itself is silent (verified by checking `stem_c.mp3` plays nothing at
# the detected time). The isolated cymbal stem is the cleaner signal
# here even with the decay-tail noise; see the user-reported
# "phantom crash at bar 5" case that motivated this routing change.
_DRUM_STEM_INFERENCE_PITCHES: frozenset[str] = frozenset({"h"})


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


@lru_cache(maxsize=1)
def _load_model():
    """Load the ADTOF model once per process and cache (lazy singleton).

    Mirrors `beat_transformer._load_model`: multi-second load, so we pay
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
    """Move the cached ADTOF Frame_RNN model to CPU. No-op when the
    lru_cache hasn't been hit yet. Callers must hold the process-wide
    GPU lock; see `app.pipeline.gpu_park`."""
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


def _adtof_activations(model, device: str, audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ADTOF dense per-frame activations for an in-memory audio
    array (already mono, at the processor's sample rate).

    Bypasses `load_audio_for_model` so we can feed a median-scaled array
    without writing a temp wav. Returns `(acts, fps)` where `acts` is
    shape `(frames, 5)`.
    """
    import torch

    proc = _audio_processor()
    stft = proc.compute_stft(audio)
    filtered = proc.apply_filterbank(stft).T.astype(np.float32, copy=False)
    # AudioProcessor.process_audio adds a trailing channel dim of size 1
    # for the mono case; the model's conv layers expect [batch, time,
    # freq, channels]. Replicate that shape here.
    filtered = filtered[:, :, np.newaxis]
    x = torch.from_numpy(filtered).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()  # [1, frames, classes]
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
    to this stem's own activation distribution — `max(floor, k * pXX)` —
    so it self-calibrates to the lane's confidence range instead of
    assuming an absolute scale that doesn't hold OOD. All other lanes use
    the fixed global `adtof_peak_threshold`.
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
    return max(
        settings.adtof_adaptive_threshold_floor,
        settings.adtof_adaptive_threshold_k * pxx,
    )


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
    if peaks.size == 0:
        return peaks
    kept = [int(peaks[0])]
    for raw in peaks[1:]:
        cand = int(raw)
        prev = kept[-1]
        between = activation[prev + 1 : cand]
        if between.size == 0:
            # Adjacent (shouldn't occur post min-distance): same event.
            continue
        reset_level = max(reset_floor, reset_frac * float(activation[prev]))
        if float(between.min()) < reset_level:
            kept.append(cand)
        # else: ring never decayed -> same sustained event, drop.
    return np.asarray(kept, dtype=int)


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
    scale = 1.0
    if settings.adtof_median_normalize and audio.size:
        scale = _median_scale_factor(audio)
        if scale != 1.0:
            audio = np.clip(audio * scale, -1.0, 1.0).astype(np.float32, copy=False)

    model, device = _load_model()
    acts, fps = _adtof_activations(model, device, audio)
    activation = acts[:, lane]
    if activation.size == 0:
        return []

    threshold = _resolve_threshold(pitch, activation)
    min_dist_s = (
        settings.adtof_noisy_peak_min_distance_s
        if is_noisy
        else settings.adtof_peak_min_distance_s
    )
    min_distance_frames = max(1, round(min_dist_s * fps))
    # Prominence gate, now applied to ALL lanes (was previously
    # noisy-only). Prominence rejects decay-tail wobbles that clear
    # `height` but don't rise above their local baseline; the
    # primary OOD failure mode on isolated kick/snare/tom stems
    # where the activation stays elevated for 100-300 ms after a
    # real hit. Noisy lanes keep the higher value tuned for plateau
    # ripples; the universal floor catches the snare/kick/tom case.
    # None = disabled (setting 0.0).
    prominence_val = (
        settings.adtof_noisy_peak_prominence
        if is_noisy
        else settings.adtof_peak_prominence
    )
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

    # `time` is the post-envelope-refine value (where the audio transient
    # actually sits); `raw_model_time` carries the pre-refine
    # `peak_frame / fps` so the per-note debug popup can surface the
    # envelope refinement as its own stage in the detected → final chain.
    candidates = [
        OnsetCandidate(
            time=refined_time,
            raw_model_time=raw_time,
            strength=float(activation[peak_idx]),
            bar=-1,
            beat_in_bar=-1.0,
        )
        for peak_idx, raw_time, refined_time in zip(
            peak_frames, peak_times_sec, refined_times, strict=False
        )
    ]
    log.info(
        "ADTOF: %d onsets in %s (lane=%d, src=%s, thr=%.3f%s, prom=%s, "
        "dist=%.0fms, reset=%s, med_norm=%s, refine=%.0fms, "
        "median strength=%.3f)",
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
        f"x{scale:.2f}" if scale != 1.0 else "off",
        settings.adtof_audio_refine_window_s * 1000.0,
        float(np.median([c.strength for c in candidates]))
        if candidates
        else 0.0,
    )
    return candidates


# Lane index → pitch letter for the alignment pool. Includes one entry
# per ADTOF lane (the merged cymbal lane is keyed by `d` so it's picked
# up once; pulling both `d` and `c` would double-count the same peaks).
_ALIGNMENT_LANES: tuple[tuple[str, int], ...] = (
    ("k", 0),
    ("s", 1),
    ("t", 2),
    ("h", 3),
    ("d", 4),
)


def detect_drum_onsets_for_alignment(
    audio_path: Path,
) -> list[tuple[float, float]]:
    """Pool ADTOF onsets across all 5 drum lanes for beat-grid alignment.

    Runs a single ADTOF inference pass on `audio_path` (intended to be
    the full drum stem) and peak-picks each lane with its standard
    `detect_onsets_adtof` parameters. Returns a flat `(time, strength)`
    list sorted by time, suitable for `align_beats_to_onsets`.

    Pooling all five lanes — kick / snare / toms / hi-hat / cymbal —
    multiplies the alignment coverage on songs where the kick alone
    leaves too few beats with a nearby onset (the median-offset
    coverage gate rejects the offset otherwise). Quiet or off-the-beat
    hi-hat hits don't bias the result: `align_beats_to_onsets` picks
    the strongest onset within ±50 ms of each beat, so a louder kick
    or snare nearby still wins.

    No drum-stem-substitution for the noisy lanes (we only want the
    union of strong drum transients, not high-fidelity per-lane
    identity), but the same median-of-non-silent amplitude
    normalization as `detect_onsets_adtof` is applied so the activation
    distribution lines up between the two entry points. Inference
    failures are caught and surfaced as an empty list so the caller can
    degrade to "no alignment" cleanly.
    """
    try:
        audio = _load_mono_audio(audio_path)
        if settings.adtof_median_normalize and audio.size:
            scale = _median_scale_factor(audio)
            if scale != 1.0:
                audio = np.clip(audio * scale, -1.0, 1.0).astype(np.float32, copy=False)
        model, device = _load_model()
        acts, fps = _adtof_activations(model, device, audio)
    except Exception as exc:
        log.warning(
            "ADTOF alignment inference failed (%s); pool empty.", exc,
        )
        return []

    pool: list[tuple[float, float]] = []
    for pitch, lane in _ALIGNMENT_LANES:
        if lane >= acts.shape[1]:
            continue
        activation = acts[:, lane]
        if activation.size == 0:
            continue
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
            activation,
            height=threshold,
            distance=min_distance_frames,
            prominence=prominence,
        )
        reset_frac = settings.adtof_noisy_decay_reset_frac
        if is_noisy and reset_frac > 0.0:
            peaks = _decay_reset_filter(
                activation,
                peaks,
                reset_frac,
                settings.adtof_noisy_decay_reset_floor,
            )
        if peaks.size == 0:
            continue
        for peak_idx in peaks:
            pool.append((
                float(peak_idx) / fps,
                float(activation[int(peak_idx)]),
            ))

    # Refine every pool entry against the drum stem audio in one pass.
    # All lanes pooled here came from this same `audio_path`, so a single
    # onset_strength envelope covers them. Run AFTER pooling rather than
    # per-lane so we pay the librosa.load / onset_strength cost exactly
    # once instead of five times.
    if pool:
        raw_times = [t for t, _ in pool]
        refined = _refine_peak_times_audio(
            audio_path,
            raw_times,
            window_sec=settings.adtof_audio_refine_window_s,
        )
        pool = [
            (refined_t, strength)
            for refined_t, (_, strength) in zip(refined, pool, strict=False)
        ]

    pool.sort(key=lambda x: x[0])
    log.info(
        "ADTOF alignment pool: %d onsets across %d lanes (%s)",
        len(pool), len(_ALIGNMENT_LANES), audio_path.name,
    )
    return pool


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

    model, device = _load_model()
    acts, fps = _adtof_activations(model, device, audio)

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
