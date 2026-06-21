import numpy as np

from drumjot_training.parampred import hybrid, report


def _gap(lane, current, determ, predicted, oracle):
    return report.LaneGap(lane=lane, n_songs=6, current_f1=current,
                          predicted_f1=predicted, oracle_f1=oracle, deterministic_f1=determ)


# per-lane gaps mirroring the ParaDB A2MD table (determ wins hc, learned wins cymbals)
_GAPS = {
    "hc": _gap("hc", 0.502, 0.526, 0.498, 0.542),
    "ho": _gap("ho", 0.630, 0.554, 0.640, 0.675),
    "rd": _gap("rd", 0.174, 0.176, 0.195, 0.198),
    "cr": _gap("cr", 0.373, 0.336, 0.411, 0.433),
}


def test_routing_selects_the_winning_source_per_lane():
    r = hybrid.DEFAULT_ROUTING
    assert hybrid.hybrid_f1(_GAPS["hc"], r) == 0.526   # determ
    assert hybrid.hybrid_f1(_GAPS["ho"], r) == 0.640   # learned
    assert hybrid.hybrid_f1(_GAPS["rd"], r) == 0.174   # global rail (current): ride doesn't generalize
    assert hybrid.hybrid_f1(_GAPS["cr"], r) == 0.411   # learned


def test_unrouted_lane_falls_back_to_global_current():
    g = _gap("k", current=0.8, determ=0.9, predicted=0.7, oracle=0.95)
    assert hybrid.hybrid_f1(g, hybrid.DEFAULT_ROUTING) == 0.8   # "k" not routed -> current


def test_determ_requested_but_missing_falls_back_to_current():
    g = report.LaneGap(lane="hc", n_songs=1, current_f1=0.5, predicted_f1=0.4,
                       oracle_f1=0.6, deterministic_f1=None)
    assert hybrid.hybrid_f1(g, {"hc": hybrid.DETERM}) == 0.5


def test_captured_beats_each_method_alone():
    hyb = hybrid.captured(_GAPS, hybrid.DEFAULT_ROUTING)
    learned = sum(g.predicted_f1 - g.current_f1 for g in _GAPS.values()) / len(_GAPS)
    determ = sum(g.deterministic_f1 - g.current_f1 for g in _GAPS.values()) / len(_GAPS)
    assert hyb > learned > 0       # hybrid > learned-only even with ride on the global rail
    assert hyb > determ            # and > determ-only (net-negative here)
    assert abs(hyb - 0.018) < 1e-3  # hc determ + ho/cr learned + rd global


def test_format_lists_each_lane_source():
    out = hybrid.format_hybrid(_GAPS, hybrid.DEFAULT_ROUTING, lane_order=["hc", "ho", "rd", "cr"])
    assert "determ" in out and "learned" in out and "HYBRID" in out


class _StubPredictor:
    def predict_row(self, lane, x):
        if lane == "ho":
            raise KeyError(lane)  # a learned-routed lane the predictor never fit
        return {"threshold": 0.42, "min_distance_s": 0.03, "prominence": 0.1,
                "decay_reset_frac": 0.5, "decay_reset_floor": 0.0}


def _activation(fps=100.0, n=400):
    a = np.full(n, 0.05)
    for f in (100, 200, 300):
        a[f] = 0.9
    return a


def test_picker_routes_determ_learned_and_global():
    seed = {"threshold": 0.5, "min_distance_s": 0.03, "prominence": 0.1,
            "decay_reset_frac": 0.5, "decay_reset_floor": 0.0}
    act, fps, sr = _activation(), 100.0, 44100
    wave = np.zeros(sr, dtype=np.float32)
    pick = hybrid.HybridParamPicker(predictor=_StubPredictor())

    # learned lane -> predictor's threshold
    p_cr = pick.params(act, fps, "cr", seed, waveform=wave, sr=sr)
    assert p_cr["threshold"] == 0.42
    # determ lane -> self-cal (threshold differs from seed, computed from the curve)
    p_hc = pick.params(act, fps, "hc", seed, waveform=wave, sr=sr)
    assert set(p_hc) == set(seed)
    # unrouted lane -> seed verbatim
    assert pick.params(act, fps, "k", seed, waveform=wave, sr=sr) == seed
    # ride is globally routed (doesn't generalize) -> seed verbatim
    assert pick.params(act, fps, "rd", seed, waveform=wave, sr=sr) == seed
    # learned lane the predictor never fit (KeyError) -> seed rail
    assert pick.params(act, fps, "ho", seed, waveform=wave, sr=sr) == seed
    # learned lane with no waveform -> seed rail
    assert pick.params(act, fps, "cr", seed) == seed
