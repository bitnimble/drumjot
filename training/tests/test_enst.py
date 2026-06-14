from drumjot_training import enst
from drumjot_training.lanes import LANES


def test_label_normalization_and_mapping():
    f = enst.lane_for_enst_class
    assert f("bd") == "k"
    assert f("sd") == "s"
    assert f("rs") == "s"                              # rim shot -> snare
    assert f("cs") == "ss"                             # cross stick -> side stick
    assert f("chh") == "hc" and f("ohh") == "ho"
    assert f("rc") == "rd" and f("rc2") == "rd"        # ride + numbered variant
    assert f("cr") == "cr" and f("c") == "cr" and f("c3") == "cr"
    assert f("ch") == "mc" and f("spl") == "mc"        # china / splash -> misc cymbal
    assert f("cb") is None                             # cowbell dropped (mp removed)
    assert f("lt") == "t" and f("mt") == "t" and f("lft") == "t" and f("ltr") == "t"
    assert f("sd-") == "s"                             # rim/ghost punctuation stripped
    assert f("sticks") is None                         # count-in
    assert f("xyz") is None                            # out of kit


def test_onsets_by_lane_parses_drops_and_sorts(tmp_path):
    ann = tmp_path / "take.txt"
    ann.write_text(
        "0.75 sd\n"        # out of time order -> result must be sorted per lane
        "0.50 bd\n"
        "0.25 chh\n"
        "1.00 rc1\n"       # ride variant
        "1.10 c2\n"        # crash variant
        "1.20 cb\n"        # cowbell -> out-of-kit -> dropped
        "2.00 sticks\n"    # count-in -> dropped
        "2.10 xyz\n"       # out-of-kit -> dropped
        "garbage\n"        # malformed -> dropped
    )
    o = enst.onsets_by_lane(ann)
    assert set(o) == set(LANES)              # output lanes only
    assert "x" not in o
    assert o["k"] == [0.5]
    assert o["s"] == [0.75]
    assert o["hc"] == [0.25]
    assert o["rd"] == [1.0]
    assert o["cr"] == [1.1]
    assert sum(len(v) for v in o.values()) == 5   # 5 kit; cowbell/sticks/xyz/garbage dropped


def test_index_pairing_and_drummer_split(tmp_path):
    for d in ("drummer_1", "drummer_3"):
        (tmp_path / d / "annotation").mkdir(parents=True)
        (tmp_path / d / "audio" / "wet_mix").mkdir(parents=True)
        (tmp_path / d / "annotation" / "t1.txt").write_text("0.1 bd\n")
        (tmp_path / d / "audio" / "wet_mix" / "t1.wav").write_bytes(b"")  # placeholder
    # annotation with no matching audio -> skipped
    (tmp_path / "drummer_1" / "annotation" / "orphan.txt").write_text("0.1 sd\n")

    clips = enst.index(tmp_path)
    assert len(clips) == 2                                    # orphan skipped
    assert {c.drummer for c in clips} == {"drummer_1", "drummer_3"}

    tr = enst.for_split(clips, "train")                       # drummer_3 held out by default
    va = enst.for_split(clips, "validation")
    assert [c.drummer for c in tr] == ["drummer_1"]
    assert [c.drummer for c in va] == ["drummer_3"]

    assert enst.index(tmp_path, mix="dry_mix") == []          # no dry_mix audio present
