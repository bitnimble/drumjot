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
    assert "cheating" in text.lower()  # display label for the per-song-best (oracle) column


def test_deterministic_column_aggregates_and_renders():
    recs = [
        report.GapRecord(lane="k", current_f1=0.6, predicted_f1=0.7, oracle_f1=0.8, deterministic_f1=0.75),
        report.GapRecord(lane="k", current_f1=0.8, predicted_f1=0.8, oracle_f1=0.9, deterministic_f1=0.85),
    ]
    g = report.aggregate(recs)["k"]
    assert abs(g.deterministic_f1 - 0.80) < 1e-9          # mean(0.75, 0.85)
    # det captured = (0.80 - 0.70) / (0.85 - 0.70) = 0.667
    assert abs(g.det_captured_frac - (0.10 / 0.15)) < 1e-9
    text = report.format_report(g and {"k": g})
    assert "determ" in text.lower()


def test_deterministic_absent_keeps_old_report():
    # records without deterministic_f1 -> no determ column, det metrics are None
    g = report.aggregate([_rec("k", 0.6, 0.7, 0.8)])["k"]
    assert g.deterministic_f1 is None
    assert "determ" not in report.format_report({"k": g}).lower()
