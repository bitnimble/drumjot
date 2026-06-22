import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from drumjot_training import clean, paradb, rlrr
from drumjot_training.lanes import LANES


def _load_gate_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_paradb_manifest.py"
    spec = importlib.util.spec_from_file_location("build_paradb_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_perstem_to_lanes_covers_all_lanes_no_overlap():
    flat = [ln for lanes in paradb.PERSTEM_TO_LANES.values() for ln in lanes]
    assert set(flat) == set(LANES)        # every lane has a stem
    assert len(flat) == len(set(flat))    # hard routing: no lane in two stems


def test_map_id_of_zip():
    assert paradb.map_id_of_zip("/x/maps__M000A2B.zip") == "M000A2B"
    assert paradb.map_id_of_zip("/x/weird_name.zip") == "weird_name"  # non-convention fallback


def _write_chart(path, complexity, events=None):
    path.write_text(json.dumps({
        "recordingMetadata": {"complexity": complexity},
        "events": events or [],
    }))


def test_pick_chart_prefers_hardest_deterministically(tmp_path):
    d = tmp_path / "Song"
    d.mkdir()
    _write_chart(d / "Song_Easy.rlrr", 1)
    _write_chart(d / "Song_Expert.rlrr", 1)   # same complexity -> filename breaks the tie
    _write_chart(d / "Song_Hard.rlrr", 1)
    chart = paradb.pick_chart(tmp_path)
    assert chart is not None and "Expert" in chart.name
    # no charts -> None
    assert paradb.pick_chart(tmp_path / "nope") is None


def test_pick_chart_ignores_macos_appledouble_junk(tmp_path):
    # macOS-zipped packs carry a binary __MACOSX/._<name>.rlrr resource fork next
    # to every real chart; it's not JSON and must not be selected/parsed.
    d = tmp_path / "Song"
    d.mkdir()
    _write_chart(d / "Song_Expert.rlrr", 3)
    junk = tmp_path / "__MACOSX" / "Song"
    junk.mkdir(parents=True)
    (junk / "._Song_Expert.rlrr").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00")  # AppleDouble magic
    (d / "._Song_Expert.rlrr").write_bytes(b"\x00\x05\x16\x07")  # sibling fork in the same dir
    chart = paradb.pick_chart(tmp_path)
    assert chart is not None
    assert chart.name == "Song_Expert.rlrr" and "__MACOSX" not in chart.parts
    assert rlrr.complexity(chart) == 3  # the selected one parses cleanly


def test_containment_high_when_drums_in_song_low_when_backing():
    rng = np.random.RandomState(0)
    drums = rng.randn(44100).astype(np.float32)
    backing = rng.randn(44100).astype(np.float32)
    full_mix = backing + drums                  # song already contains the drums
    assert paradb.containment(full_mix, drums, 44100, None) > 0.4
    assert paradb.containment(backing, drums, 44100, None) < 0.2


def test_build_mix_decides_song_only_vs_backing_plus_drums(tmp_path):
    sf = pytest.importorskip("soundfile")
    rng = np.random.RandomState(1)
    sr, n = 44100, 44100
    drums = rng.randn(n).astype(np.float32)
    backing = rng.randn(n).astype(np.float32)

    # full-mix map: the song track already contains the drums -> song-only
    fm = tmp_path / "fm"; fm.mkdir()
    sf.write(str(fm / "song.wav"), backing + drums, sr)
    sf.write(str(fm / "drums.wav"), drums, sr)
    ok, case = paradb.build_mix(fm, ["song.wav"], ["drums.wav"], sr,
                                fm / "_mix.wav", None, 0.5)
    assert ok and "song-only" in case

    # stems map: drumless backing -> backing+drums
    st = tmp_path / "st"; st.mkdir()
    sf.write(str(st / "song.wav"), backing, sr)
    sf.write(str(st / "drums.wav"), drums, sr)
    ok, case = paradb.build_mix(st, ["song.wav"], ["drums.wav"], sr,
                               st / "_mix.wav", None, 0.5)
    assert ok and "backing+drums" in case

    # no audio at all
    ok, case = paradb.build_mix(tmp_path / "empty", [], [], sr,
                               tmp_path / "_x.wav", None, 0.5)
    assert not ok and case == "no audio"


def test_global_offset_recovers_known_shift():
    fps = 100.0
    env = np.zeros(100, dtype=np.float64)
    env[[10, 20, 30]] = 1.0                       # transients at 0.1/0.2/0.3 s
    gt = {"k": [0.08, 0.18, 0.28]}                # charted 20 ms early (clearly nearest one peak)
    off, s0 = paradb.global_offset(gt, env, fps, floor=0.5, window_s=0.02, search_s=0.05)
    assert abs(off - 0.02) < 0.011                # ~+20 ms to align chart to audio
    assert 0.0 <= s0 <= 1.0


def test_shift_onsets():
    out = paradb.shift_onsets({"k": [1.0, 2.0], "s": []}, 0.5)
    assert out == {"k": [1.5, 2.5], "s": []}
    same = paradb.shift_onsets({"k": [1.0]}, 0.0)  # no-op, fresh lists
    assert same == {"k": [1.0]}


def _make_sep_tree(root, map_id, onsets, pitches):
    (root / "onsets").mkdir(parents=True, exist_ok=True)
    (root / "onsets" / f"{map_id}.json").write_text(json.dumps(onsets))
    for p in pitches:
        d = root / "perstem" / p
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{map_id}.flac").write_bytes(b"")


def test_perstem_index_and_onsets_readers(tmp_path):
    onsets = {"k": [0.1], "s": [0.2], "ss": [0.25], "rd": [0.3], "cr": [0.4], "hc": [0.5]}
    _make_sep_tree(tmp_path, "M1", onsets, pitches=("k", "c"))  # only 2 of 5 stems produced
    clips = paradb.perstem_index(tmp_path)
    assert len(clips) == 2
    assert {c.pitch for c in clips} == {"k", "c"}
    assert all(c.map_id == "M1" for c in clips)

    full = paradb.onsets_by_lane(tmp_path / "onsets" / "M1.json")
    assert set(full) == set(LANES)
    assert full["k"] == [0.1] and full["rd"] == [0.3] and full["t"] == []

    # cymbals stem -> only ride/crash ride along; everything else emptied
    o = paradb.restricted_onsets(tmp_path / "onsets" / "M1.json", "c")
    assert o["rd"] == [0.3] and o["cr"] == [0.4]
    assert o["k"] == [] and o["s"] == [] and o["hc"] == []
    # snare stem -> snare + side stick
    osn = paradb.restricted_onsets(tmp_path / "onsets" / "M1.json", "s")
    assert osn["s"] == [0.2] and osn["ss"] == [0.25]
    assert sum(len(v) for v in osn.values()) == 2


# --- distributed per-map claiming (build_paradb_manifest.py) ----------------


def _gate_dirs(tmp_path):
    wd = tmp_path / "_gate"
    (wd / "claims").mkdir(parents=True)
    (wd / "results").mkdir(parents=True)
    return wd


def test_claim_is_exclusive_and_done_is_skipped(tmp_path):
    g = _load_gate_module()
    wd = _gate_dirs(tmp_path)
    assert g._try_claim(wd, "M1", stale_s=3600) is True     # first runner wins
    assert g._try_claim(wd, "M1", stale_s=3600) is False    # second sees a live claim
    g._release(wd, "M1")                                     # crash-free finish releases
    g._write_result(wd, "M1", {"map_id": "M1", "status": "ok", "support_corr": 0.9})
    assert g._try_claim(wd, "M1", stale_s=3600) is False     # result exists -> done, never reclaimed


def test_stale_lock_is_reclaimed_but_fresh_is_not(tmp_path):
    g = _load_gate_module()
    wd = _gate_dirs(tmp_path)
    assert g._try_claim(wd, "M2", stale_s=3600) is True
    # a fresh lock held by "someone else" is respected
    assert g._try_claim(wd, "M2", stale_s=3600) is False
    # age the lock past the staleness window (crashed runner, no result) -> reclaimable
    old = time.time() - 7200
    os.utime(g._claim_path(wd, "M2"), (old, old))
    assert g._try_claim(wd, "M2", stale_s=3600) is True


def test_recall_score_flags_missing_obvious_hits():
    # drum-stem envelope with 4 clear transients at 0.1/0.2/0.3/0.4 s
    fps = 100.0
    env = np.zeros(100, dtype=np.float64)
    env[[10, 20, 30, 40]] = 1.0
    floor = 0.5  # confident-onset floor below the spikes, above the zeros
    # complete chart: a note near every audio onset -> recall 1.0
    full = {"k": [0.1, 0.2, 0.3, 0.4]}
    r = clean.recall_score(full, env, fps, confident_floor=floor, window_s=0.03, min_distance_s=0.05)
    assert r["n_confident"] == 4 and r["n_covered"] == 4 and r["fraction"] == 1.0
    # simpler chart: omits two real hits (100% precision, low recall)
    simple = {"k": [0.1, 0.3]}
    r2 = clean.recall_score(simple, env, fps, confident_floor=floor, window_s=0.03, min_distance_s=0.05)
    assert r2["n_confident"] == 4 and r2["n_covered"] == 2 and r2["fraction"] == 0.5
    # silent envelope -> no confident onsets -> nothing can be missing
    r3 = clean.recall_score(simple, np.zeros(100), fps, confident_floor=floor, window_s=0.03)
    assert r3["n_confident"] == 0 and r3["fraction"] == 1.0


def test_kept_map_ids_applies_cull():
    manifest = {
        "A": {"map_id": "A", "status": "ok", "support_corr": 0.99, "recall": 0.97, "n_onsets": 500},
        "B": {"map_id": "B", "status": "ok", "support_corr": 0.99, "recall": 0.80, "n_onsets": 500},  # low recall
        "C": {"map_id": "C", "status": "ok", "support_corr": 0.90, "recall": 0.99, "n_onsets": 500},  # low support
        "D": {"map_id": "D", "status": "no_chart"},                                                   # not ok
        "E": {"map_id": "E", "status": "ok", "support_corr": 1.0, "recall": 1.0, "n_onsets": 10},      # too few
    }
    keep = paradb.kept_map_ids(manifest, min_support=0.95, min_recall=0.90, min_onsets=50)
    assert keep == ["A"]
    # looser cut keeps B + C too (sorted)
    assert paradb.kept_map_ids(manifest, min_support=0.85, min_recall=0.75, min_onsets=50) == ["A", "B", "C"]


def test_holdout_split_is_deterministic_and_disjoint():
    ids = [f"M{i:05d}" for i in range(2000)]
    tr1, ev1 = paradb.holdout_split(ids, 0.1)
    tr2, ev2 = paradb.holdout_split(ids, 0.1)
    assert (tr1, ev1) == (tr2, ev2)            # stable across calls
    assert set(tr1).isdisjoint(ev1)            # disjoint
    assert sorted(tr1 + ev1) == sorted(ids)    # partition (no loss)
    assert 0.06 < len(ev1) / len(ids) < 0.14   # ~10% held out
    # a smaller holdout is a subset of the larger one's eval set (monotone by hash)
    _, ev_small = paradb.holdout_split(ids, 0.05)
    assert set(ev_small).issubset(set(ev1))


def test_merge_results_collects_entries(tmp_path):
    g = _load_gate_module()
    wd = _gate_dirs(tmp_path)
    g._write_result(wd, "A", {"map_id": "A", "status": "ok", "support_corr": 0.95})
    g._write_result(wd, "B", {"map_id": "B", "status": "no_chart"})
    manifest = g._merge_results(wd)
    assert set(manifest) == {"A", "B"}
    assert manifest["A"]["support_corr"] == 0.95 and manifest["B"]["status"] == "no_chart"
