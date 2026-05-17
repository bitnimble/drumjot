"""Fitness function for refinement: per-stem onset F1 of a Jot against the
ground-truth stem onsets detected by `pipeline/onsets.py`.

This is the exact same metric the rest of the ADT field uses (`mir_eval`
note-level F1 with a configurable onset tolerance), so progress on this
score translates directly to standard benchmarks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mir_eval
import numpy as np

from app.debug import current_debug_sink
from app.models import OnsetCandidate
from app.pipeline.beats import BeatStructure
from app.pipeline.jot_extract import ExtractedJot, PredictedOnset

log = logging.getLogger(__name__)

# Field-standard ADT onset tolerance. Most ADT papers (E-GMD, MDB Drums,
# IDMT-SMT-Drums) and the N2N benchmark we aim to compare against report
# F1 at this window. Callers that want stricter timing can override per-
# call.
DEFAULT_TOLERANCE_SECONDS = 0.05


@dataclass
class Score:
    onset_f1: float
    per_pitch_f1: dict[str, float] = field(default_factory=dict)

    @property
    def combined(self) -> float:
        return self.onset_f1

    def per_pitch_summary(self) -> str:
        """One-line breakdown for log messages: `k=0.85 s=0.40 h=0.00 ...`."""
        if not self.per_pitch_f1:
            return "(no pitches)"
        return " ".join(
            f"{p}={v:.2f}" for p, v in sorted(self.per_pitch_f1.items())
        )


def score_jot(
    extracted: ExtractedJot,
    stem_onsets: dict[str, list[OnsetCandidate]],
    tolerance: float = DEFAULT_TOLERANCE_SECONDS,
    time_offset: float = 0.0,
    structure: BeatStructure | None = None,
    debug_tag: str | None = None,
) -> Score:
    """Mean per-stem onset F1 with `mir_eval`-style matching.

    A pitch that exists in the predicted Jot but not in the source stems
    (or vice-versa) is scored as 0 for that pitch - this penalises both
    invented and dropped instruments.

    Predicted-time anchoring (in order of preference):
    1. If `structure` is provided AND a predicted onset carries a valid
       `bar` (>=0) and `beat_in_bar`, re-time the prediction using the
       AUDIO's bar boundaries:
           `audio_t = structure.bars[bar].start_time
                      + beat_in_bar * (60 / structure.bars[bar].tempo_bpm)`
       This bypasses the LLM's emitted BPM entirely and absorbs both the
       initial t=0 offset and any per-bar tempo drift in one step.
    2. Otherwise fall back to `predicted.time + time_offset`, which only
       handles the leading constant offset.

    `debug_tag` writes a per-pitch breakdown (predicted vs actual times,
    precision, recall, F1) to `<debug_dir>/score/<tag>.json` when a
    request-scoped `DebugSink` is active. No-op otherwise.
    """
    per_pitch_f1: dict[str, float] = {}
    per_pitch_details: dict[str, dict] = {}
    f1_values: list[float] = []

    all_pitches = set(extracted.onsets_by_pitch) | set(stem_onsets)
    for pitch in sorted(all_pitches):
        predicted = extracted.onsets_by_pitch.get(pitch, [])
        actual = stem_onsets.get(pitch, [])
        if not predicted and not actual:
            continue

        pred_times = np.array(sorted(
            _retime_predicted(p, structure, time_offset) for p in predicted
        ))
        actual_times = np.array(sorted(c.time for c in actual))
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        if len(pred_times) > 0 and len(actual_times) > 0:
            pred_intervals = np.column_stack([pred_times, pred_times + 0.1])
            actual_intervals = np.column_stack([actual_times, actual_times + 0.1])
            # mir_eval.transcription requires strictly positive pitch
            # values; for onset-only ADT scoring we pass a constant
            # placeholder so pitch_tolerance=0.0 trivially matches every
            # (ref, est) pair on pitch and the F1 reflects timing only.
            dummy_pitch = 60.0
            ref_pitches = np.full(len(actual_times), dummy_pitch)
            est_pitches = np.full(len(pred_times), dummy_pitch)
            try:
                precision, recall, f1, _ = (
                    mir_eval.transcription.precision_recall_f1_overlap(
                        actual_intervals,
                        ref_pitches,
                        pred_intervals,
                        est_pitches,
                        onset_tolerance=tolerance,
                        pitch_tolerance=0.0,
                    )
                )
                precision = float(precision)
                recall = float(recall)
                f1 = float(f1)
            except Exception as exc:  # pragma: no cover - mir_eval edge cases
                log.warning("mir_eval f1 failed for pitch=%s: %s", pitch, exc)
                precision = recall = f1 = 0.0
        per_pitch_f1[pitch] = f1
        f1_values.append(f1)
        per_pitch_details[pitch] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_predicted": int(len(pred_times)),
            "n_actual": int(len(actual_times)),
            "predicted_times": [round(float(t), 4) for t in pred_times.tolist()],
            "actual_times": [round(float(t), 4) for t in actual_times.tolist()],
        }

    score = Score(
        onset_f1=float(np.mean(f1_values)) if f1_values else 0.0,
        per_pitch_f1=per_pitch_f1,
    )

    if debug_tag:
        sink = current_debug_sink()
        if sink is not None:
            sink.write_json(
                f"score/{debug_tag}.json",
                {
                    "tag": debug_tag,
                    "tolerance_seconds": tolerance,
                    "time_offset_seconds": time_offset,
                    "structure_anchored": structure is not None,
                    "overall_f1": round(score.onset_f1, 4),
                    "per_pitch": per_pitch_details,
                },
            )

    return score


def _retime_predicted(
    onset: PredictedOnset,
    structure: BeatStructure | None,
    time_offset: float,
) -> float:
    """Return the comparison time for a predicted onset.

    When `structure` is provided AND the onset carries a valid bar
    index within that structure, the time is recomputed from the
    audio's bar boundary + per-bar tempo. Otherwise we fall back to the
    onset's bun-bridge-emitted DSL time shifted by `time_offset`.
    """
    if structure is not None and structure.bars and onset.bar >= 0:
        if onset.bar < len(structure.bars):
            bar = structure.bars[onset.bar]
            tempo = max(bar.tempo_bpm, 1.0)
            return float(bar.start_time + onset.beat_in_bar * (60.0 / tempo))
    return float(onset.time + time_offset)
