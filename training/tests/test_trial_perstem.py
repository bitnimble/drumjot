import json

from drumjot_training import trial_perstem
from drumjot_training.lanes import LANES


def _make_tree(root, cid, onsets, pitches):
    (root / "onsets").mkdir(parents=True, exist_ok=True)
    (root / "onsets" / f"{cid}.json").write_text(json.dumps(onsets))
    for p in pitches:
        d = root / "perstem" / p
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cid}.flac").write_bytes(b"")


def test_5way_layout_detected_and_cymbal_routed_to_both(tmp_path):
    onsets = {"k": [0.1], "s": [0.2], "ss": [0.25], "rd": [0.3], "cr": [0.4], "hc": [0.5]}
    _make_tree(tmp_path, "C1", onsets, pitches=("k", "s", "h", "c", "t"))
    assert trial_perstem.lane_map(tmp_path) is trial_perstem.LANES_5WAY
    clips = trial_perstem.perstem_index(tmp_path)
    assert {c.pitch for c in clips} == {"k", "s", "h", "c", "t"}
    c = next(x for x in clips if x.pitch == "c")  # merged cymbal stem -> rd + cr
    o = trial_perstem.restricted_onsets(c.onsets_path, c.lanes)
    assert o["rd"] == [0.3] and o["cr"] == [0.4] and o["k"] == []


def test_6way_layout_detected_and_splits_ride_crash(tmp_path):
    onsets = {"k": [0.1], "rd": [0.3], "cr": [0.4], "hc": [0.5]}
    _make_tree(tmp_path, "C2", onsets, pitches=("k", "s", "h", "rd", "cr", "t"))
    assert trial_perstem.lane_map(tmp_path) is trial_perstem.LANES_6WAY
    clips = trial_perstem.perstem_index(tmp_path)
    assert {"rd", "cr"} <= {c.pitch for c in clips} and "c" not in {c.pitch for c in clips}
    rd = next(x for x in clips if x.pitch == "rd")
    cr = next(x for x in clips if x.pitch == "cr")
    ord_ = trial_perstem.restricted_onsets(rd.onsets_path, rd.lanes)
    ocr = trial_perstem.restricted_onsets(cr.onsets_path, cr.lanes)
    assert ord_["rd"] == [0.3] and ord_["cr"] == []   # ride stem -> ride only
    assert ocr["cr"] == [0.4] and ocr["rd"] == []      # crash stem -> crash only


def test_each_layout_has_hard_routing_no_lane_double():
    for lm in (trial_perstem.LANES_5WAY, trial_perstem.LANES_6WAY):
        flat = [ln for lanes in lm.values() for ln in lanes]
        assert set(flat) == set(LANES)     # every lane covered
        assert len(flat) == len(set(flat))  # no lane in two stems within a layout


def test_split_deterministic_and_partitions(tmp_path):
    for i in range(40):
        _make_tree(tmp_path, f"M{i:03d}", {"k": [0.1]}, pitches=("k",))
    clips = trial_perstem.perstem_index(tmp_path)
    tr = trial_perstem.perstem_for_split(clips, "train")
    va = trial_perstem.perstem_for_split(clips, "validation")
    tr_ids, va_ids = {c.map_id for c in tr}, {c.map_id for c in va}
    assert tr_ids.isdisjoint(va_ids)
    assert tr_ids | va_ids == {c.map_id for c in clips}
    assert len(va_ids) > 0  # 40 clips at val_frac 0.1 -> some val
    # stable across calls
    assert {c.map_id for c in trial_perstem.perstem_for_split(clips, "validation")} == va_ids
