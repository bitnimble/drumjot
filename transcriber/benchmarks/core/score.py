"""mir_eval-based 3-class onset F1 scoring.

This is the metric N2N (and most ADT work) reports: per-class
`mir_eval.onset.f_measure` at a fixed match tolerance (default 50 ms),
then mean-averaged across classes for a per-track score, then
mean-averaged across tracks for a dataset score.

The benchmark deliberately keeps this module pure: no I/O, no logging,
no dataset-specific logic — just numbers in, numbers out. This makes
it pytest-friendly with synthetic event lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mir_eval
import numpy as np

from .classes import ALL_CLASSES, DrumClass
from .events import OnsetEvent, group_by_class

DEFAULT_TOLERANCE_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class ClassScore:
    """Precision/recall/F1 for a single drum class on a single track."""

    drum_class: DrumClass
    precision: float
    recall: float
    f1: float
    n_reference: int
    n_estimated: int


@dataclass(frozen=True, slots=True)
class TrackScore:
    track_id: str
    per_class: dict[DrumClass, ClassScore]
    f1_macro: float                  # mean of per-class F1
    f1_weighted: float               # F1 weighted by reference-count per class

    def as_jsonable(self) -> dict:
        return {
            "track_id": self.track_id,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "per_class": {
                cls.value: {
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                    "n_reference": s.n_reference,
                    "n_estimated": s.n_estimated,
                }
                for cls, s in self.per_class.items()
            },
        }


@dataclass
class DatasetSummary:
    dataset: str
    n_tracks: int = 0
    f1_macro_mean: float = 0.0       # primary headline number
    f1_weighted_mean: float = 0.0
    per_class_f1_mean: dict[DrumClass, float] = field(default_factory=dict)
    per_class_n_reference: dict[DrumClass, int] = field(default_factory=dict)
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS

    def as_jsonable(self) -> dict:
        return {
            "dataset": self.dataset,
            "n_tracks": self.n_tracks,
            "tolerance_seconds": self.tolerance_seconds,
            "f1_macro_mean": self.f1_macro_mean,
            "f1_weighted_mean": self.f1_weighted_mean,
            "per_class_f1_mean": {c.value: v for c, v in self.per_class_f1_mean.items()},
            "per_class_n_reference": {c.value: v for c, v in self.per_class_n_reference.items()},
        }


def score_track(
    track_id: str,
    reference: list[OnsetEvent],
    estimated: list[OnsetEvent],
    tolerance: float = DEFAULT_TOLERANCE_SECONDS,
) -> TrackScore:
    """3-class onset F1 against a single track.

    Empty classes (no reference *and* no estimate) are skipped. A class
    with reference onsets but no predictions scores 0; vice-versa.
    """
    ref_by_class = group_by_class(reference)
    est_by_class = group_by_class(estimated)

    per_class: dict[DrumClass, ClassScore] = {}
    f1_values: list[float] = []
    weighted_num = 0.0
    weighted_den = 0

    for cls in ALL_CLASSES:
        ref = ref_by_class.get(cls, [])
        est = est_by_class.get(cls, [])
        if not ref and not est:
            continue
        if not ref or not est:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            f1, precision, recall = mir_eval.onset.f_measure(
                np.asarray(ref, dtype=float),
                np.asarray(est, dtype=float),
                window=tolerance,
            )
            f1, precision, recall = float(f1), float(precision), float(recall)
        per_class[cls] = ClassScore(
            drum_class=cls,
            precision=precision,
            recall=recall,
            f1=f1,
            n_reference=len(ref),
            n_estimated=len(est),
        )
        f1_values.append(f1)
        weighted_num += f1 * len(ref)
        weighted_den += len(ref)

    f1_macro = float(np.mean(f1_values)) if f1_values else 0.0
    f1_weighted = weighted_num / weighted_den if weighted_den else 0.0
    return TrackScore(
        track_id=track_id,
        per_class=per_class,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
    )


def summarise(
    dataset: str,
    tracks: list[TrackScore],
    tolerance: float = DEFAULT_TOLERANCE_SECONDS,
) -> DatasetSummary:
    """Roll per-track scores up to a dataset-level summary."""
    if not tracks:
        return DatasetSummary(dataset=dataset, tolerance_seconds=tolerance)

    f1_macro_mean = float(np.mean([t.f1_macro for t in tracks]))
    f1_weighted_mean = float(np.mean([t.f1_weighted for t in tracks]))

    per_class_f1: dict[DrumClass, list[float]] = {}
    per_class_refs: dict[DrumClass, int] = {}
    for t in tracks:
        for cls, s in t.per_class.items():
            per_class_f1.setdefault(cls, []).append(s.f1)
            per_class_refs[cls] = per_class_refs.get(cls, 0) + s.n_reference

    per_class_mean = {
        cls: float(np.mean(values)) for cls, values in per_class_f1.items()
    }

    return DatasetSummary(
        dataset=dataset,
        n_tracks=len(tracks),
        f1_macro_mean=f1_macro_mean,
        f1_weighted_mean=f1_weighted_mean,
        per_class_f1_mean=per_class_mean,
        per_class_n_reference=per_class_refs,
        tolerance_seconds=tolerance,
    )
