from drumjot_training import star
from drumjot_training.lanes import LANES


def test_perstem_to_lanes_covers_all_but_mp_no_overlap():
    flat = [ln for lanes in star.PERSTEM_TO_LANES.values() for ln in lanes]
    assert set(flat) == set(LANES)            # every lane has a stem (mp removed)
    assert len(flat) == len(set(flat))         # hard routing: no lane in two stems


def test_restricted_onsets_keeps_only_stem_lanes(tmp_path):
    ann = tmp_path / "t1.txt"
    ann.write_text(
        "0.10\tBD\t100\n0.20\tSD\t90\n0.30\tCRC\t80\n0.40\tRD\t80\n0.50\tCHC\t70\n0.60\tCB\t60\n"
    )
    # cymbals stem -> only ride/crash kept; china (CHC) dropped, everything else emptied
    o = star.restricted_onsets(ann, "c")
    assert set(o) == set(LANES)
    assert o["cr"] == [0.30] and o["rd"] == [0.40]
    assert "mc" not in o  # china dropped (mc lane removed)
    assert o["k"] == [] and o["s"] == []
    # kick stem -> only k
    ok = star.restricted_onsets(ann, "k")
    assert ok["k"] == [0.10] and sum(len(v) for v in ok.values()) == 1


def test_perstem_index_pairs_present_stems_and_infers_split(tmp_path):
    base = tmp_path / "data" / "training" / "song1"
    (base / "annotation").mkdir(parents=True)
    (base / "annotation" / "t1.txt").write_text("0.1\tBD\t100\n")
    for pitch in ("k", "c"):  # only 2 of the 5 stems produced
        d = base / "audio" / "perstem" / pitch
        d.mkdir(parents=True)
        (d / "t1.flac").write_bytes(b"")
    clips = star.perstem_index(tmp_path)
    assert len(clips) == 2
    assert {c.pitch for c in clips} == {"k", "c"}
    assert all(c.split == "training" for c in clips)
