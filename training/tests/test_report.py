from drumjot_training.parampred import report


def _rec(lane, cur, pred, orc):
    return report.GapRecord(lane=lane, current_f1=cur, predicted_f1=pred, oracle_f1=orc)


def test_aggregate_means_per_lane():
    recs = [_rec("k", 0.8, 0.85, 0.9), _rec("k", 0.6, 0.7, 0.8), _rec("s", 0.5, 0.5, 0.5)]
    gaps = report.aggregate(recs)
    assert gaps["k"].n_songs == 2
    assert abs(gaps["k"].current_f1 - 0.7) < 1e-9
    assert abs(gaps["k"].predicted_f1 - 0.775) < 1e-9
    assert abs(gaps["k"].oracle_f1 - 0.85) < 1e-9


def test_captured_fraction_endpoints():
    # predicted == oracle -> captured the whole gap
    full = report.aggregate([_rec("k", 0.6, 0.9, 0.9)])["k"]
    assert abs(full.captured_frac - 1.0) < 1e-9
    # predicted == current -> captured nothing
    none = report.aggregate([_rec("k", 0.6, 0.6, 0.9)])["k"]
    assert abs(none.captured_frac - 0.0) < 1e-9


def test_zero_gap_is_fully_captured():
    # oracle == current: no prize exists, treat as fully captured (not div-by-zero)
    g = report.aggregate([_rec("k", 0.8, 0.8, 0.8)])["k"]
    assert g.gap == 0.0
    assert g.captured_frac == 1.0


def test_format_report_mentions_lanes_and_columns():
    gaps = report.aggregate([_rec("k", 0.6, 0.7, 0.8), _rec("hc", 0.4, 0.55, 0.7)])
    text = report.format_report(gaps, lane_order=("k", "hc"))
    assert "k" in text and "hc" in text
    assert "oracle" in text.lower()
