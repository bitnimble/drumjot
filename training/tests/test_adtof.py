"""ADTOF loader: reduced-pitch -> lane parsing, per-stem routing, split."""
from drumjot_training import adtof


def _write(p, lines):
    p.write_text("\n".join(lines) + "\n")


def test_onsets_by_lane_routes_reduced_pitches(tmp_path):
    ann = tmp_path / "x.txt"
    _write(ann, [
        "0.10\t35",      # BD -> k
        "0.20\t38",      # SD -> s
        "0.30\t47",      # TT -> t
        "0.40\t42",      # HH closed -> hc
        "0.50\t46",      # OH -> ho (task=7)
        "0.60\t51",      # RD -> rd (task=7)
        "0.70\t49",      # CY -> cr
        "0.80\t60",      # unknown reduced pitch -> dropped
        "garbage line",  # malformed (no tab) -> dropped
        "0.90\t38\t99",  # tolerated extra (velocity) column: time, pitch, ...
    ])
    o = adtof.onsets_by_lane(ann)
    assert o["k"] == [0.10]
    assert o["s"] == [0.20, 0.90]
    assert o["t"] == [0.30]
    assert o["hc"] == [0.40]
    assert o["ho"] == [0.50]
    assert o["rd"] == [0.60]
    assert o["cr"] == [0.70]
    assert "60" not in o  # only lane keys, never raw pitches


def test_restricted_onsets_keeps_only_the_stem_lanes(tmp_path):
    ann = tmp_path / "y.txt"
    _write(ann, ["0.1\t51", "0.2\t49", "0.3\t42", "0.4\t46", "0.5\t35"])
    c = adtof.restricted_onsets(ann, "c")  # cymbal stem -> rd/cr only
    assert c["rd"] == [0.1] and c["cr"] == [0.2]
    assert c["hc"] == [] and c["k"] == []
    h = adtof.restricted_onsets(ann, "h")  # hi-hat stem -> hc/ho only
    assert h["hc"] == [0.3] and h["ho"] == [0.4]
    assert h["rd"] == [] and h["cr"] == []


def test_for_split_is_deterministic_and_partitions(tmp_path):
    clips = [adtof.AdtofClip(tmp_path / f"trk{i}.ogg", tmp_path / f"trk{i}.txt", f"trk{i}")
             for i in range(40)]
    tr = adtof.for_split(clips, "train")
    va = adtof.for_split(clips, "validation")
    assert len(tr) + len(va) == len(clips)
    assert {c.track for c in tr}.isdisjoint({c.track for c in va})
    assert adtof.for_split(clips, "train") == tr  # stable across calls
