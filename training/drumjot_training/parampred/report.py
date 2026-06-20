"""Oracle-gap report: current vs predicted vs oracle, per lane.

Aggregates per-(song, lane) onset-F1 at three operating points into the headline
the whole effort is judged on (design spec §eval integration):

- **current** - today's single global-tuned param per lane,
- **predicted** - the param predictor's per-song params,
- **oracle** - the per-song best (the ceiling),

and the **fraction of the oracle gap captured** = (predicted - current) /
(oracle - current). A zero gap (oracle == current, nothing to win) counts as
fully captured rather than dividing by zero. Pure Python, no numpy.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True)
class GapRecord:
    """One song's onset-F1 for one lane at the three operating points."""

    lane: str
    current_f1: float
    predicted_f1: float
    oracle_f1: float


@dataclass(frozen=True)
class LaneGap:
    """Per-lane means across songs plus the derived gap metrics."""

    lane: str
    n_songs: int
    current_f1: float
    predicted_f1: float
    oracle_f1: float

    @property
    def gap(self) -> float:
        """The prize: how much per-song-oracle params beat today's global."""
        return self.oracle_f1 - self.current_f1

    @property
    def captured(self) -> float:
        """How much of it the predictor actually captured."""
        return self.predicted_f1 - self.current_f1

    @property
    def captured_frac(self) -> float:
        """`captured` / `gap`, with a zero gap treated as fully captured."""
        if self.gap <= _EPS:
            return 1.0
        return self.captured / self.gap


def aggregate(records: Sequence[GapRecord]) -> dict[str, LaneGap]:
    """Group per-(song, lane) records into per-lane means."""
    by_lane: dict[str, list[GapRecord]] = defaultdict(list)
    for r in records:
        by_lane[r.lane].append(r)
    out: dict[str, LaneGap] = {}
    for lane, rs in by_lane.items():
        n = len(rs)
        out[lane] = LaneGap(
            lane=lane,
            n_songs=n,
            current_f1=sum(r.current_f1 for r in rs) / n,
            predicted_f1=sum(r.predicted_f1 for r in rs) / n,
            oracle_f1=sum(r.oracle_f1 for r in rs) / n,
        )
    return out


def format_report(gaps: Mapping[str, LaneGap], lane_order: Sequence[str] | None = None) -> str:
    """Render the per-lane gap table as text (the eval-harness headline)."""
    order = [ln for ln in (lane_order or sorted(gaps)) if ln in gaps]
    lines = [
        "==== per-lane onset-F1: current (global) vs predicted (per-song) vs oracle ====",
        f"  {'lane':4s} {'current':>8s} {'predict':>8s} {'oracle':>8s} {'gap':>7s} {'captured':>9s} {'songs':>6s}",
    ]
    for ln in order:
        g = gaps[ln]
        lines.append(
            f"  {ln:4s} {g.current_f1:8.3f} {g.predicted_f1:8.3f} {g.oracle_f1:8.3f} "
            f"{g.gap:+7.3f} {g.captured_frac * 100:8.1f}% {g.n_songs:6d}"
        )
    return "\n".join(lines)
