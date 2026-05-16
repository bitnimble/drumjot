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

from app.models import OnsetCandidate
from app.pipeline.jot_extract import ExtractedJot

log = logging.getLogger(__name__)

# Tighter than the ADT-paper default of 50ms: refinement should improve
# timing accuracy too, not just hit-vs-miss.
DEFAULT_TOLERANCE_SECONDS = 0.03


@dataclass
class Score:
    onset_f1: float
    per_pitch_f1: dict[str, float] = field(default_factory=dict)

    @property
    def combined(self) -> float:
        return self.onset_f1


def score_jot(
    extracted: ExtractedJot,
    stem_onsets: dict[str, list[OnsetCandidate]],
    tolerance: float = DEFAULT_TOLERANCE_SECONDS,
) -> Score:
    """Mean per-stem onset F1 with `mir_eval`-style matching.

    A pitch that exists in the predicted Jot but not in the source stems
    (or vice-versa) is scored as 0 for that pitch - this penalises both
    invented and dropped instruments.
    """
    per_pitch_f1: dict[str, float] = {}
    f1_values: list[float] = []

    all_pitches = set(extracted.onsets_by_pitch) | set(stem_onsets)
    for pitch in sorted(all_pitches):
        predicted = extracted.onsets_by_pitch.get(pitch, [])
        actual = stem_onsets.get(pitch, [])
        if not predicted and not actual:
            continue

        pred_times = np.array(sorted(p.time for p in predicted))
        actual_times = np.array(sorted(c.time for c in actual))
        if len(pred_times) == 0 or len(actual_times) == 0:
            f1 = 0.0
        else:
            pred_intervals = np.column_stack([pred_times, pred_times + 0.1])
            actual_intervals = np.column_stack([actual_times, actual_times + 0.1])
            try:
                _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    actual_intervals,
                    np.zeros(len(actual_times)),
                    pred_intervals,
                    np.zeros(len(pred_times)),
                    onset_tolerance=tolerance,
                    pitch_tolerance=0.0,
                )
            except Exception as exc:  # pragma: no cover - mir_eval edge cases
                log.warning("mir_eval f1 failed for pitch=%s: %s", pitch, exc)
                f1 = 0.0
        per_pitch_f1[pitch] = float(f1)
        f1_values.append(float(f1))

    return Score(
        onset_f1=float(np.mean(f1_values)) if f1_values else 0.0,
        per_pitch_f1=per_pitch_f1,
    )
