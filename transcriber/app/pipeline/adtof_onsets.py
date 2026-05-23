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
  deterministic peak-pick (threshold + min-distance) so the output is
  reproducible run-to-run and tuned for the "high-recall, the LLM
  prunes" contract. We deliberately do NOT use the package's own
  `PeakPicker` / `transcribe_to_midi` for this reason.
"""
from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from librosa.onset import onset_backtrack
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

# Lanes whose activation is OOD-compressed and bleed-contaminated on an
# isolated stem: hihat and the merged ride/crash cymbal lane. Only these
# get RMS input normalization + an adaptive per-stem threshold. Kick (k)
# / snare (s) / toms (t) lanes stay on the fixed high-recall threshold —
# their transients are well-defined even on isolated stems, so the
# "detect hot, LLM prunes" contract still holds there.
_NOISY_LANE_PITCHES: frozenset[str] = frozenset({"h", "d", "c"})


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


def _adtof_activations(model, device: str, audio_path: Path) -> tuple[np.ndarray, float]:
    """Return ADTOF dense per-frame activations for one audio file.

    Returns `(acts, fps)` where `acts` is shape `(frames, 5)`. This is
    the single seam coupled to adtof_pytorch's inference API; any
    mismatch raises and is handled by `detect_onsets_adtof_or_librosa`.
    """
    import torch
    from adtof_pytorch import load_audio_for_model

    x = load_audio_for_model(str(audio_path)).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()  # [1, frames, classes]
    acts = np.asarray(pred, dtype=np.float64)
    if acts.ndim == 3 and acts.shape[0] == 1:
        acts = acts[0]
    if acts.ndim != 2 or acts.shape[1] < 5:
        raise RuntimeError(
            f"Unexpected ADTOF activation shape {np.asarray(pred).shape} "
            f"for {audio_path.name} (expected (1, frames, >=5))."
        )
    return acts, _ADTOF_FPS


def _rms_normalized_tempfile(audio_path: Path) -> Path | None:
    """Write an RMS-normalized mono copy of `audio_path` to a temp wav.

    RMS (not peak) normalization on purpose: the separator's isolated
    hihat/cymbal stems carry snare/crash bleed spikes, so a peak-norm
    (what the ADTOF package does internally, if enabled) would key the
    whole stem off a transient that often isn't even the target
    instrument. Normalizing to a target RMS keeps the *bulk* instrument
    energy at a consistent level across tracks regardless of separator
    output gain, which is what makes a stable threshold meaningful.

    Returns the temp path, or `None` if the stem is effectively silent
    (nothing to normalize — caller should use the original file).
    """
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if y.size == 0:
        return None
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return None
    target = 10.0 ** (settings.adtof_rms_target_dbfs / 20.0)
    # Hard-clip post-gain: bleed spikes pushed past full-scale are
    # clipped rather than allowed to dominate (the package may still
    # peak-norm); the target instrument's level is what we're fixing.
    y = np.clip(y * (target / rms), -1.0, 1.0).astype(np.float32)
    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="adtof_rms_")
    os.close(fd)
    tmp = Path(tmp_name)
    sf.write(str(tmp), y, sr)
    return tmp


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


def detect_onsets_adtof(
    audio_path: Path,
    pitch: str,
    sample_rate: int = 44100,
    *,
    drum_stem_path: Path | None = None,
) -> list[OnsetCandidate]:
    """ADTOF onsets for one stem, reading only the stem's class lane.

    `audio_path` is the isolated per-instrument stem (identity source).
    For the noisy lanes (`_NOISY_LANE_PITCHES`) inference instead runs on
    `drum_stem_path` when supplied and present — ADTOF is in-distribution
    on a full drum mix, so the open-hat sizzle stops being read as a
    phantom hit-train; kick/snare/toms always use their own stem. When
    `drum_stem_path` is missing the noisy lanes fall back to the isolated
    stem.

    Returns `OnsetCandidate`s with the ADTOF sigmoid activation in
    [0, 1] as `strength` — a per-frame model-confidence proxy. It feeds
    the filter LLM's `(beat_in_bar, strength)` prompt block and the
    velocity-mapping percentile lookup in `onsets_to_midi_bytes`.

    For the noisy lanes the inference audio is RMS-normalized and the
    peak threshold is computed adaptively from that lane's activation;
    kick/snare/toms are unchanged.
    """
    lane = _LANE_FOR_PITCH.get(pitch)
    if lane is None:
        # Unknown/custom stem pitch: nothing ADTOF can speak to.
        return []

    is_noisy = pitch in _NOISY_LANE_PITCHES
    # Noisy lanes prefer the in-distribution drum stem; everything else
    # (and the fallback when no drum stem is cached) uses the isolated
    # stem. `source_path` is what ADTOF actually sees.
    source_path = audio_path
    used_drum_stem = False
    if (
        is_noisy
        and drum_stem_path is not None
        and drum_stem_path.exists()
    ):
        source_path = drum_stem_path
        used_drum_stem = True

    infer_path = source_path
    tmp_path: Path | None = None
    if settings.adtof_rms_normalize and is_noisy:
        tmp_path = _rms_normalized_tempfile(source_path)
        if tmp_path is not None:
            infer_path = tmp_path

    try:
        model, device = _load_model()
        acts, fps = _adtof_activations(model, device, infer_path)
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
        # Prominence only applies to the noisy lanes: it rejects plateau
        # ripples (open-hihat / cymbal sustain) that clear `height` but
        # don't rise above their local baseline. None = no prominence
        # gate (kick/snare/toms keep pure height+distance picking).
        prominence = (
            settings.adtof_noisy_peak_prominence
            if is_noisy and settings.adtof_noisy_peak_prominence > 0.0
            else None
        )
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

        # Backtrack each peak to the nearest preceding local minimum of
        # the activation, so the reported time sits at the transient's
        # leading edge instead of the model's peak-activation frame. The
        # librosa onset path applies the same correction (`backtrack=True`
        # in `librosa.onset.onset_detect`); without it ADTOF onsets lag
        # the true attack by tens of milliseconds, which is enough to
        # scatter ~50% of straight kicks across non-quarter 16th slots
        # after MIDI quantization. `strength` is still read at the peak
        # because that's where the model's confidence actually is.
        backtracked = (
            onset_backtrack(peaks, activation) if peaks.size else peaks
        )

        candidates = [
            OnsetCandidate(
                time=float(bt_idx) / fps,
                strength=float(activation[peak_idx]),
                bar=-1,
                beat_in_bar=-1.0,
            )
            for peak_idx, bt_idx in zip(peaks, backtracked, strict=False)
        ]
        log.info(
            "ADTOF: %d onsets in %s (lane=%d, src=%s, thr=%.3f%s, prom=%s, "
            "dist=%.0fms, reset=%s, rms_norm=%s, median strength=%.3f)",
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
            tmp_path is not None,
            float(np.median([c.strength for c in candidates]))
            if candidates
            else 0.0,
        )
        return candidates
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


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

    No RMS-normalize / drum-stem-substitution for the noisy lanes — we
    only want the union of strong drum transients, not high-fidelity
    per-lane identity. Inference failures are caught and surfaced as
    an empty list so the caller can degrade to "no alignment" cleanly.
    """
    try:
        model, device = _load_model()
        acts, fps = _adtof_activations(model, device, audio_path)
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
        prominence = (
            settings.adtof_noisy_peak_prominence
            if is_noisy and settings.adtof_noisy_peak_prominence > 0.0
            else None
        )
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
        backtracked = onset_backtrack(peaks, activation)
        for peak_idx, bt_idx in zip(peaks, backtracked, strict=False):
            pool.append((
                float(bt_idx) / fps,
                float(activation[peak_idx]),
            ))

    pool.sort(key=lambda x: x[0])
    log.info(
        "ADTOF alignment pool: %d onsets across %d lanes (%s)",
        len(pool), len(_ALIGNMENT_LANES), audio_path.name,
    )
    return pool
