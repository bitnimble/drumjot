"""Canonical onset-event type shared by loaders and the scorer.

A `OnsetEvent` is a (time-in-seconds, drum-class) pair. Both reference
(ground truth from the dataset) and estimate (Drumjot prediction) are
expressed as `list[OnsetEvent]` so the scorer has a uniform input.
"""
from __future__ import annotations

from dataclasses import dataclass

from .classes import DrumClass


@dataclass(frozen=True, slots=True)
class OnsetEvent:
    time: float
    drum_class: DrumClass


def group_by_class(events: list[OnsetEvent]) -> dict[DrumClass, list[float]]:
    """Bucket events by class, returning sorted onset-time lists.

    Empty buckets are omitted, so `dict[DrumClass, list[float]]` is the
    natural input shape for `mir_eval.onset.f_measure` per class.
    """
    grouped: dict[DrumClass, list[float]] = {}
    for ev in events:
        grouped.setdefault(ev.drum_class, []).append(ev.time)
    for times in grouped.values():
        times.sort()
    return grouped
