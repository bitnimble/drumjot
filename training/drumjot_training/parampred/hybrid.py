"""Per-lane hybrid param policy: route each lane to its best-known param source.

ParaDB evidence (RESULTS.md, 2026-06-21) shows the deterministic self-calibration
and the learned real-domain (A2MD) predictor are COMPLEMENTARY, not competing:
determ wins the dense closed-hat (a clean peak-height histogram knee), the learned
predictor wins the sparse cymbals (per-song timbre is the diversity it needs). The
hybrid routes each lane to its winner and falls back to the global seed for any
lane without a known-good source (the conservative deploy rail), capturing more of
the oracle gap than either method alone.

`HybridParamPicker.params` is the deployable inference path (mirrors
eval_gap._predicted_f1 / _score_params); the report helpers score it offline from
already-computed `report.LaneGap`s, so no model re-run is needed to evaluate a
routing -- the hybrid F1 for a lane is just a selection among the current /
deterministic / predicted columns the gap report already holds.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from drumjot_training.parampred import baseline, features, report

DETERM, LEARNED, GLOBAL = "determ", "learned", "global"

# Source per lane; any lane absent here uses the global seed (the conservative
# rail). Chosen from the 6-song ParaDB gap table -- treat as a starting policy to
# re-validate on the larger dist0p20 A2MD corpus + MDB, not a tuned constant.
DEFAULT_ROUTING: dict[str, str] = {
    "hc": DETERM,   # dense closed-hat: self-cal knee wins (+59% of gap on ParaDB)
    "ho": LEARNED,  # open-hat: determ catastrophic (-169%), learned +24%
    "rd": LEARNED,  # ride: learned +89% (thin corpus -> noisy)
    "cr": LEARNED,  # crash: learned +63%, determ -61%
}


def _captured_frac(captured: float, gap: float) -> float:
    return 1.0 if gap <= 1e-9 else captured / gap


def lane_f1(g: report.LaneGap, source: str) -> float:
    """The routed source's mean F1 for one lane. Routing is per-lane (not per-song),
    so selecting on the aggregated `LaneGap` equals aggregating per-song selections."""
    if source == DETERM and g.deterministic_f1 is not None:
        return g.deterministic_f1
    if source == LEARNED:
        return g.predicted_f1
    return g.current_f1  # GLOBAL, or determ requested but unavailable


def hybrid_f1(g: report.LaneGap, routing: Mapping[str, str]) -> float:
    return lane_f1(g, routing.get(g.lane, GLOBAL))


def captured(gaps: Mapping[str, report.LaneGap], routing: Mapping[str, str]) -> float:
    """Mean over lanes of (hybrid_f1 - current_f1) -- comparable to report's
    'mean captured' line."""
    vals = [hybrid_f1(g, routing) - g.current_f1 for g in gaps.values()]
    return sum(vals) / len(vals) if vals else 0.0


def format_hybrid(
    gaps: Mapping[str, report.LaneGap], routing: Mapping[str, str],
    lane_order: Sequence[str] | None = None,
) -> str:
    """Per-lane hybrid table: the routed source, its F1, and the fraction of the
    oracle gap it captures."""
    order = [ln for ln in (lane_order or sorted(gaps)) if ln in gaps]
    lines = [
        "==== per-lane HYBRID picker (route each lane to its best source) ====",
        f"  {'lane':4s} {'source':>7s} {'current':>8s} {'hybrid':>8s} {'oracle':>8s} {'cap%':>7s}",
    ]
    for ln in order:
        g = gaps[ln]
        src = routing.get(ln, GLOBAL)
        hf = hybrid_f1(g, routing)
        cap = _captured_frac(hf - g.current_f1, g.gap)
        lines.append(
            f"  {ln:4s} {src:>7s} {g.current_f1:8.3f} {hf:8.3f} {g.oracle_f1:8.3f} {cap * 100:6.1f}%"
        )
    return "\n".join(lines)


class HybridParamPicker:
    """Deployable per-lane param source: routes a lane to {determ self-cal, learned
    predictor, global seed}. Used by the eval and (later) the transcriber. `params`
    returns a peak-pick param dict ready for `metrics.pick_onsets`."""

    def __init__(self, predictor=None, routing: Mapping[str, str] = DEFAULT_ROUTING):
        self.predictor = predictor
        self.routing = dict(routing)

    def params(
        self, activation, fps: float, lane: str, seed: Mapping[str, float], *,
        waveform=None, sr: int | None = None, beat_period_s: float | None = None,
    ) -> dict[str, float]:
        src = self.routing.get(lane, GLOBAL)
        if src == DETERM:
            return baseline.deterministic_params(activation, fps, dict(seed))
        if src == LEARNED and self.predictor is not None and waveform is not None and sr is not None:
            try:
                x = features.feature_vector(
                    activation, fps, seed["min_distance_s"], waveform, sr, beat_period_s=beat_period_s
                )
                return self.predictor.predict_row(lane, x)
            except KeyError:
                pass  # lane never fit -> global rail
        return dict(seed)
