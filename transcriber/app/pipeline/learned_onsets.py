"""Trained drum-onset model as a transcriber pipeline stage (SPIKE).

Loads a checkpoint produced by the `training/` package and emits
`OnsetCandidate`s per DSL pitch, mirroring
`adtof_onsets.detect_all_lanes_adtof` so it can slot into the pipeline. Runs
the frozen MERT encoder + per-lane heads and folds the 11 training lanes to
the DSL pitch letters via `inference.LANE_TO_DSL`.

SPIKE STATUS, this validates the wiring (the transcriber can load the
trained model and produce pipeline-shaped onsets). Not yet called from
`runner.py`. Two follow-ups before production:
  - Packaging: the `drumjot_training` package must be importable here (add
    `training/` to PYTHONPATH or `uv pip install -e training`); torch +
    transformers are already transcriber deps, MERT weights download/cache
    on first use.
  - `amplitude` is left None (velocity falls back to `strength`); a later
    pass can read it off the stem like `adtof_onsets` does.
"""
from __future__ import annotations

from pathlib import Path

from app.models import OnsetCandidate


def detect_all_pitches_learned(
    audio_path: Path,
    checkpoint_dir: Path,
    encoder=None,
) -> dict[str, list[OnsetCandidate]]:
    """Per-pitch `OnsetCandidate`s from a trained checkpoint.

    `strength` is the model's sigmoid activation at the peak frame (the same
    "is this a hit?" confidence `adtof_onsets` provides). The lane->pitch map
    is INJECTIVE (`inference.LANE_TO_PITCH`): every trained class keeps a
    distinct pitch (kick/snare/side-stick/tom/closed-hat/pedal-hat/open-hat/
    ride/crash/misc-cymbal/misc-perc) and a distinct GM note downstream, so
    nothing is merged back down. Any display folding happens later in MIDI->Jot.
    """
    from drumjot_training import inference, metrics

    model, meta = inference.load_model(checkpoint_dir)
    probs, fps = inference.lane_probs(audio_path, model, meta, encoder=encoder)
    thresholds = meta["thresholds"]
    n_frames = probs.shape[1]

    by_pitch: dict[str, list[OnsetCandidate]] = {}
    for i, lane in enumerate(meta["lanes"]):
        pitch = inference.LANE_TO_PITCH.get(lane)
        if pitch is None:
            continue
        thr = thresholds.get(lane, meta["peak_threshold"])
        for t in metrics.pick_onsets_lane(probs[i], fps, lane, thr):
            frame = min(int(round(float(t) * fps)), n_frames - 1)
            by_pitch.setdefault(pitch, []).append(
                OnsetCandidate(
                    time=float(t),
                    strength=float(probs[i][frame]),
                    amplitude=None,
                    bar=-1,
                    beat_in_bar=-1.0,
                    raw_model_time=float(t),
                )
            )

    for cands in by_pitch.values():
        cands.sort(key=lambda c: c.time)
    return by_pitch
