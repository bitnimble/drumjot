from drumjot_training import rlrr

_CHART = {
    "recordingMetadata": {"complexity": 3},
    "audioFileData": {"songTracks": ["song.ogg"], "drumTracks": ["drums.ogg", "song.ogg"]},
    "events": [
        {"name": "BP_Kick_C_1", "time": 0.5, "vel": 100},
        {"name": "BP_Snare_C_1", "time": "1.0000", "vel": 90},  # time as 4-dp string
        {"name": "BP_HiHat_C_1", "time": 0.25, "vel": 80},  # closed (no midi)
        {"name": "BP_HiHat_C_1", "time": 0.75, "vel": 80, "midi": 46},  # open
        {"name": "BP_HiHat_C_2", "time": 0.90, "vel": 80, "midi": 44},  # pedal
        {"name": "BP_Crash15_C_1", "time": 2.0},
        {"name": "BP_China15_C_1", "time": 2.5},  # -> dropped (mc lane removed)
        {"name": "BP_Ride17_C_1", "time": 1.5},
        {"name": "BP_FloorTom_C_1", "time": 1.1},  # -> toms
        {"name": "BP_Cowbell_C_1", "time": 3.0},  # -> misc perc
        {"name": "BP_Triangle_C_1", "time": 4.0},  # aux -> dropped
        {"name": "garbage", "time": 5.0},  # no class -> dropped
    ],
}


def test_load_handles_utf16_bom_chart(tmp_path):
    # ~10% of community .rlrr files are UTF-16 (Windows BOM); load reads bytes so
    # json.loads auto-detects the encoding instead of throwing UnicodeDecodeError.
    import json

    p = tmp_path / "utf16.rlrr"
    p.write_text(json.dumps(_CHART), encoding="utf-16")  # FF FE BOM
    assert p.read_bytes()[:2] == b"\xff\xfe"  # confirm it's really UTF-16-LE
    o = rlrr.onsets_by_lane(p)  # would raise UnicodeDecodeError before the fix
    assert o["k"] == [0.5] and o["cr"] == [2.0]
    assert rlrr.complexity(p) == 3


def test_onsets_by_lane_maps_and_refines_hihat():
    o = rlrr.onsets_by_lane(_CHART)
    assert o["k"] == [0.5]
    assert o["s"] == [1.0]  # string time parsed
    assert o["hc"] == [0.25]  # no midi -> closed
    assert o["ho"] == [0.75]  # midi 46 -> open
    assert o["hp"] == [0.90]  # midi 44 -> pedal
    assert o["cr"] == [2.0]
    assert "mc" not in o  # china dropped (mc lane removed)
    assert o["rd"] == [1.5]
    assert o["t"] == [1.1]  # floor tom
    assert "mp" not in o  # cowbell dropped (mp lane removed)
    assert o["ss"] == []  # no side-stick class in rlrr


def test_aux_and_unknown_events_dropped():
    o = rlrr.onsets_by_lane(_CHART)
    # triangle (aux) + garbage (no class) + china (mc removed) contribute to no lane
    assert sum(len(v) for v in o.values()) == 8


def test_audio_tracks_dedup_song_first():
    assert rlrr.audio_tracks(_CHART) == ["song.ogg", "drums.ogg"]


def test_complexity():
    assert rlrr.complexity(_CHART) == 3


def test_difficulty_rank_orders_known_names():
    assert rlrr.difficulty_rank("Song_Expert.rlrr") == 4
    assert rlrr.difficulty_rank("Song_Hard.rlrr") == 3
    assert rlrr.difficulty_rank("Song_Medium.rlrr") == 2
    assert rlrr.difficulty_rank("Song_Easy.rlrr") == 1
    assert rlrr.difficulty_rank("Song.rlrr") == 0  # unlabelled


def _write_chart(path, complexity, n_events):
    import json
    chart = {"recordingMetadata": {"complexity": complexity},
             "audioFileData": {"songTracks": [], "drumTracks": []},
             "events": [{"name": "BP_Kick_C_1", "time": float(i)} for i in range(n_events)]}
    path.write_text(json.dumps(chart))


def test_pick_hardest_breaks_complexity_tie_deterministically(tmp_path):
    # Expert + Hard at the SAME complexity but different onset counts (the real
    # Kaikai_Kitan case): the tie must resolve to Expert regardless of input order.
    expert = tmp_path / "Song_Expert.rlrr"
    hard = tmp_path / "Song_Hard.rlrr"
    _write_chart(expert, complexity=4, n_events=1640)
    _write_chart(hard, complexity=4, n_events=1516)
    assert rlrr.pick_hardest([expert, hard]) == expert
    assert rlrr.pick_hardest([hard, expert]) == expert  # order-independent
    assert rlrr.pick_hardest([]) is None


def test_pick_hardest_prefers_higher_complexity_over_difficulty_name(tmp_path):
    easy_but_complex = tmp_path / "Song_Easy.rlrr"
    expert_simple = tmp_path / "Song_Expert.rlrr"
    _write_chart(easy_but_complex, complexity=4, n_events=10)
    _write_chart(expert_simple, complexity=2, n_events=10)
    assert rlrr.pick_hardest([easy_but_complex, expert_simple]) == easy_but_complex


def test_song_and_drum_tracks_split():
    # song.ogg appears in both arrays -> counts as a song track only
    assert rlrr.song_tracks(_CHART) == ["song.ogg"]
    assert rlrr.drum_tracks(_CHART) == ["drums.ogg"]


def _pairs_dict(gt, est):
    return {label: (ref, es) for label, ref, es in rlrr.comparison_pairs(gt, est)}


def test_comparison_folds_group_when_chart_has_one_subclass():
    # chart has only closed hats + only crash -> fold both groups
    gt = {"hc": [0.5], "cr": [1.0], "k": [0.0]}
    est = {"hc": [0.5], "ho": [0.6], "rd": [1.0], "cr": [1.1], "k": [0.0]}
    p = _pairs_dict({**{ln: [] for ln in rlrr.LANES}, **gt},
                    {**{ln: [] for ln in rlrr.LANES}, **est})
    # hats folded to "h": model's hc+ho both count toward the single chart hat
    assert "hc" not in p and "ho" not in p
    assert p["h"] == ([0.5], [0.5, 0.6])
    # cymbals folded to "cym": model rd+cr vs the single chart cymbal
    assert "rd" not in p and "cr" not in p
    assert p["cym"] == ([1.0], [1.0, 1.1])
    assert p["k"] == ([0.0], [0.0])  # ungrouped passthrough


def test_comparison_splits_group_when_chart_distinguishes():
    # chart has BOTH open and closed hats, BOTH ride and crash -> score per subclass
    gt = {"hc": [0.5], "ho": [0.7], "rd": [1.0], "cr": [2.0]}
    est = {"hc": [0.5], "ho": [0.7], "rd": [1.0], "cr": [2.0]}
    p = _pairs_dict({**{ln: [] for ln in rlrr.LANES}, **gt},
                    {**{ln: [] for ln in rlrr.LANES}, **est})
    assert "h" not in p and "cym" not in p
    assert p["hc"] == ([0.5], [0.5])
    assert p["ho"] == ([0.7], [0.7])
    assert p["rd"] == ([1.0], [1.0])
    assert p["cr"] == ([2.0], [2.0])


def _hat_chart(events):
    return {"recordingMetadata": {"complexity": 1}, "audioFileData": {}, "events": events}


def test_bimodal_hihat_velocity_splits_open_closed():
    ev = [
        {"name": "BP_HiHat_C_1", "time": 0.1, "vel": 20},
        {"name": "BP_HiHat_C_1", "time": 0.2, "vel": 100},
        {"name": "BP_HiHat_C_1", "time": 0.3, "vel": 20},
        {"name": "BP_HiHat_C_1", "time": 0.4, "vel": 100},
    ]
    o = rlrr.onsets_by_lane(_hat_chart(ev))
    assert o["ho"] == [0.1, 0.3]  # quieter -> open
    assert o["hc"] == [0.2, 0.4]  # louder -> closed


def test_single_velocity_hihat_stays_closed():
    ev = [{"name": "BP_HiHat_C_1", "time": t, "vel": 100} for t in (0.1, 0.2, 0.3)]
    o = rlrr.onsets_by_lane(_hat_chart(ev))
    assert o["hc"] == [0.1, 0.2, 0.3]
    assert o["ho"] == []


def test_three_velocities_not_treated_as_open_closed():
    ev = [
        {"name": "BP_HiHat_C_1", "time": 0.1, "vel": 20},
        {"name": "BP_HiHat_C_1", "time": 0.2, "vel": 60},
        {"name": "BP_HiHat_C_1", "time": 0.3, "vel": 100},
    ]
    o = rlrr.onsets_by_lane(_hat_chart(ev))
    assert o["hc"] == [0.1, 0.2, 0.3]  # real dynamics, not an open/closed code
    assert o["ho"] == []


def test_midi_extension_overrides_velocity_bimodal():
    ev = [
        {"name": "BP_HiHat_C_1", "time": 0.1, "vel": 20, "midi": 46},  # open by midi
        {"name": "BP_HiHat_C_1", "time": 0.2, "vel": 100, "midi": 42},  # closed by midi
    ]
    o = rlrr.onsets_by_lane(_hat_chart(ev))
    assert o["ho"] == [0.1]  # midi wins despite low vel
    assert o["hc"] == [0.2]  # midi wins despite high vel


def test_has_lane_track():
    chart = {"instruments": [{"class": "BP_Kick_C"}, {"class": "BP_Cowbell_C"}], "events": []}
    assert rlrr.has_lane_track(chart, "mp") is False  # mp removed: cowbell unmapped
    assert rlrr.has_lane_track(chart, "mc") is False  # no china/splash/ride-bell
    assert rlrr.has_lane_track(chart, "k") is True
    assert rlrr.has_lane_track({"instruments": []}, "mp") is False


def test_instance_name_to_class():
    assert rlrr.instance_name_to_class("BP_Snare_C_1") == "BP_Snare_C"
    assert rlrr.instance_name_to_class("BP_Crash17_C_12") == "BP_Crash17_C"
    assert rlrr.instance_name_to_class("nope") is None
