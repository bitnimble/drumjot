import drumjot_training.lanes as lanes
import drumjot_training.star as star


def test_core_folding():
    assert star.lane_for_star_class("BD") == "k"
    assert star.lane_for_star_class("SD") == "s"
    assert star.lane_for_star_class("SS") == "ss"
    for c in ("HT", "MT", "LT"):
        assert star.lane_for_star_class(c) == "t"


def test_hat_subclasses():
    assert star.lane_for_star_class("CHH") == "hc"
    assert star.lane_for_star_class("PHH") == "hp"
    assert star.lane_for_star_class("OHH") == "ho"


def test_ride_and_crash_split():
    assert star.lane_for_star_class("RD") == "rd"
    assert star.lane_for_star_class("RC") == "rd"  # defensive alias for ride
    assert star.lane_for_star_class("CRC") == "cr"


def test_misc_cymbals():
    for c in ("SPC", "CHC", "RB"):  # splash, china, ride bell
        assert star.lane_for_star_class(c) == "mc"


def test_misc_percussion_dropped():
    # mp lane removed: cowbell / clap / tambourine are out-of-kit now
    for c in ("CB", "CL", "CLP", "TB"):
        assert star.lane_for_star_class(c) is None


def test_unknown_unmapped():
    assert star.lane_for_star_class("XYZ") is None


def test_onsets_by_lane_parses_annotation(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(
        "0.01\tBD\t120\n"
        "0.50\tSS\t90\n"
        "0.50\tCB\t80\n"
        "0.80\tOHH\t100\n"
    )
    out = star.onsets_by_lane(f)
    assert out["k"] == [0.01]
    assert out["ss"] == [0.50]
    assert "mp" not in out  # CB dropped (mp removed)
    assert out["ho"] == [0.80]


def test_all_lanes_present(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0.0\tBD\t100\n")
    assert set(star.onsets_by_lane(f)) == set(lanes.LANES)


def _make_song(root, split_dir, name):
    mixdir = root / "data" / split_dir / "audio" / "mix"
    anndir = root / "data" / split_dir / "annotation"
    mixdir.mkdir(parents=True, exist_ok=True)
    anndir.mkdir(parents=True, exist_ok=True)
    (mixdir / f"{name}.flac").write_bytes(b"flac")
    (anndir / f"{name}.txt").write_text("0.0\tBD\t100\n")


def test_index_pairs_mix_audio_with_annotation(tmp_path):
    _make_song(tmp_path, "test", "Song_mix_kit_full")
    clips = star.index(tmp_path)
    assert len(clips) == 1
    assert clips[0].audio_path.name == "Song_mix_kit_full.flac"
    assert clips[0].annotation_path.name == "Song_mix_kit_full.txt"
    assert clips[0].split == "test"


def test_index_infers_split_under_subcorpus(tmp_path):
    _make_song(tmp_path, "training/ismir04", "A_mix")
    clips = star.index(tmp_path)
    assert len(clips) == 1
    assert clips[0].split == "training"


def test_index_skips_annotation_without_matching_audio(tmp_path):
    anndir = tmp_path / "data" / "test" / "annotation"
    anndir.mkdir(parents=True)
    (anndir / "orphan.txt").write_text("0.0\tBD\t1\n")
    assert star.index(tmp_path) == []


def test_for_split_filters(tmp_path):
    _make_song(tmp_path, "training", "a_mix")
    _make_song(tmp_path, "test", "b_mix")
    clips = star.index(tmp_path)
    assert [c.split for c in star.for_split(clips, "test")] == ["test"]
