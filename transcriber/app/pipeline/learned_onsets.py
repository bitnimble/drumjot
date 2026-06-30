"""Trained drum-onset model as a transcriber pipeline stage.

Loads a checkpoint produced by the `training/` package and emits
`OnsetCandidate`s per DSL pitch, mirroring `adtof_onsets.detect_onsets_adtof`
so it can slot into the pipeline. Runs the frozen MERT encoder + per-lane
heads and folds the 9 training lanes to the DSL pitch letters via
`inference.LANE_TO_PITCH`.

PER-STEM, matching the deployment architecture. The model is trained, tuned,
and evaluated PER STEM (`training/scripts/sota_eval.py::_predict_perstem` +
`enst.PERSTEM_TO_LANES`): run the model on each isolated per-instrument stem,
keep only that stem's owned lanes. The transcriber's `stems_per` stage already
produces exactly those k/s/h/c/t stems, so we run the model once per stem and
assemble. Running it once on the merged drum stem (the old spike shortcut)
would (a) not match the per-stem isolation the model was tuned for, leaking
cross-lane bleed the published numbers exclude, and (b) feed MERT one long
sequence (its attention is O(n^2)).

WINDOWED. We use `inference.stitched_probs` (overlapping ~30 s chunks, centre-
crop stitched) rather than a single full-song encode, bounding MERT's sequence
length / VRAM. This is the same path `inference.transcribe` (hence the eval
harness) uses, so the transcriber's onsets match the scored numbers.

Post-model peak-picking is the SHARED training-side picker
(`metrics.pick_onsets_lane`: per-lane min-distance + prominence + decay-reset
from `drumjot_dsp.peakpick`) at the checkpoint's TUNED per-lane thresholds.
We deliberately do NOT apply the ADTOF backend's OOD-correction machinery
(adaptive threshold, median-normalise, hi-hat audio supplement, amplitude
floor, crash-shadow): those compensate for ADTOF being out-of-distribution on
isolated stems, and the adaptive threshold in particular would override the
model's calibrated operating points. See `docs/learned-onsets-integration.md`
for the full picker comparison + rationale.

`amplitude` is left None (velocity falls back to `strength`): per-onset audio
amplitude would have to be read off the per-stem audio, a later refinement.
`refine_audio` (off by default) optionally snaps each peak time to the nearest
audio onset-strength maximum (the ADTOF backend's `_refine_peak_times_audio`),
upgrading the 75 fps / ~13 ms grid to a transient-honest time; keep it off
until validated F1-neutral against the eval harness.
"""
from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.models import OnsetCandidate


def _resolve_device() -> str:
    """CUDA when available (respecting `settings.device`), else CPU.

    Mirrors `adtof_onsets._resolve_device`: `cpu`/`mps` force CPU; anything
    else asks for CUDA but degrades to CPU when torch reports no GPU."""
    import torch

    pref = (settings.device or "auto").lower()
    if pref in ("cpu", "mps"):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _onset_onnx_enabled() -> bool:
    """Default ON: run MERT + heads through onnxruntime (cross-platform, torch-free
    inference). Opt out with DRUMJOT_ONSET_ONNX in {0,false,no,off,torch} for the
    torch path (mirrors DRUMJOT_SEP_ONNX for separation)."""
    import os

    return os.environ.get("DRUMJOT_ONSET_ONNX", "1").strip().lower() not in (
        "0", "false", "no", "off", "torch",
    )


def _onnx_providers(device: str) -> list[str] | None:
    """onnxruntime providers for `device`: CPU-pinned when CPU is forced, else
    `None` (np_onsets uses onnxruntime's available set -- CUDA/DirectML/CoreML
    then CPU -- with a CPU fallback) so a GPU EP is used when present."""
    return ["CPUExecutionProvider"] if device == "cpu" else None


def detect_all_pitches_learned(
    per_instrument_stems: dict[str, Path],
    checkpoint_dir: Path,
    encoder=None,
    *,
    device: str | None = None,
    refine_audio: bool = False,
) -> dict[str, list[OnsetCandidate]]:
    """Per-pitch `OnsetCandidate`s from a trained checkpoint, PER STEM.

    `per_instrument_stems` maps the transcriber's stem pitch letters
    (k/s/h/c/t, as produced by `stems_per`) to the isolated stem audio. For
    each stem we run the model and keep only the lanes that stem owns
    (`enst.PERSTEM_TO_LANES`), then fold to DSL pitches via
    `inference.LANE_TO_PITCH` (INJECTIVE: every trained class keeps a distinct
    pitch / GM note; any display folding happens later in MIDI->Jot).

    `strength` is the model's sigmoid activation at the peak frame (the same
    "is this a hit?" confidence `adtof_onsets` provides). One MERT encoder is
    built once and reused across stems.
    """
    # drumjot_training / drumjot_dsp (the training package) aren't installed in the
    # transcriber venv -- add the monorepo's training/ + dsp/ to sys.path so this stage
    # can import them. A path/editable dep in transcriber/pyproject.toml would be cleaner
    # but needs an install; a Docker image running this must include training/ + dsp/.
    import sys

    repo_root = Path(__file__).resolve().parents[3]  # transcriber/app/pipeline/ -> repo root
    for pkg_root in (repo_root / "training", repo_root / "dsp"):
        if pkg_root.is_dir() and str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))
    from drumjot_training import embeddings, enst, inference, metrics

    dev = device or _resolve_device()
    # Default: the torch-free ONNX path (MERT + heads on onnxruntime); opt out
    # with DRUMJOT_ONSET_ONNX=0 for the torch path. Both expose `stitched(audio)
    # -> (probs, fps)`; everything downstream (per-lane picking, tom split) is
    # shared. The ONNX path exports the two `.onnx` once (cached, torch needed
    # only then); the `encoder` arg applies to the torch path only.
    if _onset_onnx_enabled():
        from app.pipeline.onset_onnx.np_onsets import load_onnx_onset

        onnx_model, meta = load_onnx_onset(checkpoint_dir, providers=_onnx_providers(dev))
        stitched = onnx_model.stitched_probs
    else:
        model, meta = inference.load_model(checkpoint_dir, dev)
        # One MERT encoder, reused across stems (else each stem reloads the 330M
        # weights), co-located on `dev` so a forced settings.device=cpu is honoured.
        enc = encoder or embeddings.MertEncoder(
            name=meta["encoder"], layer=meta["encoder_layer"], device=dev
        )

        def stitched(audio_path):
            return inference.stitched_probs(audio_path, model, meta, encoder=enc)

    thresholds = meta["thresholds"]
    lane_index = {lane: i for i, lane in enumerate(meta["lanes"])}

    by_pitch: dict[str, list[OnsetCandidate]] = {}
    for stem_pitch, audio_path in per_instrument_stems.items():
        owned = enst.PERSTEM_TO_LANES.get(stem_pitch, ())
        if not owned:
            continue  # stem the model has no lane for (e.g. residual)
        probs, fps = stitched(audio_path)
        n_frames = probs.shape[1]
        for lane in owned:
            i = lane_index.get(lane)
            pitch = inference.LANE_TO_PITCH.get(lane)
            if i is None or pitch is None:
                continue
            thr = thresholds.get(lane, meta["peak_threshold"])
            times = [float(t) for t in metrics.pick_onsets_lane(probs[i], fps, lane, thr)]
            refined = (
                _refine_peak_times(audio_path, times)
                if refine_audio
                else times
            )
            for raw_t, t in zip(times, refined, strict=True):
                frame = min(int(round(raw_t * fps)), n_frames - 1)
                by_pitch.setdefault(pitch, []).append(
                    OnsetCandidate(
                        time=t,
                        strength=float(probs[i][frame]),
                        amplitude=None,
                        bar=-1,
                        beat_in_bar=-1.0,
                        raw_model_time=raw_t,
                    )
                )

    # Tom sub-classification (part of the model's post-processing, not a separate
    # pipeline stage): split the merged `t` lane into distinct toms
    # (floor/low/mid/high) by per-song pitch clustering, reusing the tom + kick
    # stems the model already consumed. Each onset is re-filed under its tom
    # pitch key (f/tl/tm/t, low->high). Degrades to the merged `t` lane on any
    # failure. See docs/tom-subclassification.md.
    toms = by_pitch.pop("t", None)
    if toms:
        from drumjot_training import tom_subclassify

        keys = tom_subclassify.subclassify(
            per_instrument_stems.get("t"),
            [c.time for c in toms],
            per_instrument_stems.get("k"),
            [c.time for c in by_pitch.get("k", ())],
        )
        for cand, key in zip(toms, keys, strict=True):
            by_pitch.setdefault(key, []).append(cand)

    for cands in by_pitch.values():
        cands.sort(key=lambda c: c.time)
    return by_pitch


def _refine_peak_times(audio_path: Path, times: list[float]) -> list[float]:
    """Snap each peak time to the nearest audio onset-strength maximum.

    Reuses the ADTOF backend's audio-domain refinement so the two onset paths
    share one implementation. Optional (off by default) for the learned model:
    its targets are onset-centred, but the 75 fps grid is coarse, so this can
    sharpen playback timing once validated F1-neutral."""
    from app.pipeline.adtof_onsets import _refine_peak_times_audio

    return _refine_peak_times_audio(
        audio_path, times, window_sec=settings.adtof_audio_refine_window_s
    )
