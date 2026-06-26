"""Oracle-gap report: fixed-threshold vs predicted vs cheating, per lane.

Aggregates per-(song, lane) onset-F1 at three operating points into the headline
the whole effort is judged on (design spec §eval integration). Display labels in
parentheses (the internal field names keep the historical current/oracle terms):

- **fixed threshold** (current) - today's single global-tuned param per lane,
- **predicted** - the param predictor's per-song params,
- **cheating** (oracle) - the per-song best threshold, chosen against GT (the
  un-deployable ceiling),

and the **fraction of the cheating gap captured** = (predicted - fixed) /
(cheating - fixed). A zero gap (cheating == fixed, nothing to win) counts as
fully captured rather than dividing by zero. Pure Python, no numpy.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True)
class GapRecord:
    """One song's onset-F1 for one lane at the operating points. `deterministic_f1`
    (per-song self-calibrated params, no training) is optional."""

    lane: str
    current_f1: float
    predicted_f1: float
    oracle_f1: float
    deterministic_f1: float | None = None


def _captured_frac(captured: float, gap: float) -> float:
    """`captured / gap`, with a zero gap (nothing to win) treated as fully captured."""
    return 1.0 if gap <= _EPS else captured / gap


@dataclass(frozen=True)
class LaneGap:
    """Per-lane means across songs plus the derived gap metrics."""

    lane: str
    n_songs: int
    current_f1: float
    predicted_f1: float
    oracle_f1: float
    deterministic_f1: float | None = None

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
        return _captured_frac(self.captured, self.gap)

    @property
    def det_captured_frac(self) -> float | None:
        """Fraction of the gap the deterministic self-calibration captured."""
        if self.deterministic_f1 is None:
            return None
        return _captured_frac(self.deterministic_f1 - self.current_f1, self.gap)


def aggregate(records: Sequence[GapRecord]) -> dict[str, LaneGap]:
    """Group per-(song, lane) records into per-lane means."""
    by_lane: dict[str, list[GapRecord]] = defaultdict(list)
    for r in records:
        by_lane[r.lane].append(r)
    out: dict[str, LaneGap] = {}
    for lane, rs in by_lane.items():
        n = len(rs)
        dets = [r.deterministic_f1 for r in rs if r.deterministic_f1 is not None]
        out[lane] = LaneGap(
            lane=lane,
            n_songs=n,
            current_f1=sum(r.current_f1 for r in rs) / n,
            predicted_f1=sum(r.predicted_f1 for r in rs) / n,
            oracle_f1=sum(r.oracle_f1 for r in rs) / n,
            deterministic_f1=(sum(dets) / len(dets)) if dets else None,
        )
    return out


def format_report(gaps: Mapping[str, LaneGap], lane_order: Sequence[str] | None = None) -> str:
    """Render the per-lane gap table as text (the eval-harness headline). Adds a
    `determ` (self-calibrated) column when any lane carries it."""
    order = [ln for ln in (lane_order or sorted(gaps)) if ln in gaps]
    has_det = any(gaps[ln].deterministic_f1 is not None for ln in order)
    head = f"  {'lane':4s} {'fixed':>8s}"
    if has_det:
        head += f" {'determ':>8s}"
    head += f" {'predict':>8s} {'cheating':>8s} {'gap':>7s}"
    head += f" {'det%':>7s} {'pred%':>7s}" if has_det else f" {'captured':>9s}"
    head += f" {'songs':>6s}"
    title = (
        "==== per-lane onset-F1: fixed-threshold vs determ (self-cal) vs predict vs cheating (per-song best) ===="
        if has_det else
        "==== per-lane onset-F1: fixed-threshold (global) vs predicted (per-song) vs cheating (per-song best) ===="
    )
    lines = [title, head]
    for ln in order:
        g = gaps[ln]
        row = f"  {ln:4s} {g.current_f1:8.3f}"
        if has_det:
            row += f" {(g.deterministic_f1 if g.deterministic_f1 is not None else g.current_f1):8.3f}"
        row += f" {g.predicted_f1:8.3f} {g.oracle_f1:8.3f} {g.gap:+7.3f}"
        if has_det:
            dcf = g.det_captured_frac
            row += f" {(dcf * 100 if dcf is not None else 0.0):6.1f}% {g.captured_frac * 100:6.1f}%"
        else:
            row += f" {g.captured_frac * 100:8.1f}%"
        row += f" {g.n_songs:6d}"
        lines.append(row)
    return "\n".join(lines)
