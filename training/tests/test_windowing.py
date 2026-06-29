"""Multi-window clip segmentation (train._window_specs / plan_windows) + the
cache-key compatibility that keeps pre-windowing caches valid."""
import numpy as np
import pytest

from drumjot_training import embeddings, train


def test_cache_key_start_appended_only_when_nonzero(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    # start=0 must be byte-identical to the legacy key (so existing caches are reused)
    assert embeddings.cache_key(f, "enc", 10, 30.0) == embeddings.cache_key(f, "enc", 10, 30.0, start=0.0)
    # a non-zero window offset gives a distinct key
    assert embeddings.cache_key(f, "enc", 10, 30.0, start=30.0) != embeddings.cache_key(f, "enc", 10, 30.0)


def test_window_specs_legacy_single_window_is_just_a_cap():
    # max_windows=1 -> one window at start 0, onsets capped at `window`, NOT shifted,
    # and NO audio read (path never opened).
    specs = [("/x/a.flac", {"k": [0.1, 35.0], "s": [5.0]})]
    out = train._window_specs(specs, window=30.0, search=3.0, max_windows=1)
    assert len(out) == 1
    audio, onsets, weight, start, length = out[0]
    assert (start, length, weight) == (0.0, 30.0, None)
    assert list(onsets["k"]) == [0.1] and list(onsets["s"]) == [5.0]  # array.array; 35.0s dropped


def test_window_specs_carries_weight_onsets():
    specs = [("/x/a.flac", {"k": [1.0]}, {"k": [1.0], "s": [2.0]})]  # 3-tuple = per-stem weight
    out = train._window_specs(specs, window=30.0, search=3.0, max_windows=1)
    assert {ln: list(v) for ln, v in out[0][2].items()} == {"k": [1.0], "s": [2.0]}  # weight (array.array)


def _write(tmp_path, name, secs, sr=24000, gaps=()):
    import soundfile as sf

    y = (np.random.RandomState(0).randn(int(secs * sr)) * 0.3).astype("float32")
    for c in gaps:  # carve ~1s near-silence centered at c
        y[int((c - 0.5) * sr): int((c + 0.5) * sr)] = 0.0
    p = tmp_path / name
    sf.write(str(p), y, sr)
    return str(p)


def test_plan_windows_short_clip_no_read(tmp_path):
    pytest.importorskip("soundfile")
    p = _write(tmp_path, "short.flac", 10.0)
    assert train.plan_windows(p, 30.0, 3.0, 0) == [(0.0, 30.0)]


def test_plan_windows_cut_snaps_to_low_energy_gap(tmp_path):
    pytest.importorskip("soundfile")
    # gaps at 28s and 60s: first cut snaps to 28 (within 30+/-3); second to 60, so the
    # tail is ~5s and stays its own window (not merged), keeping the count at 3.
    p = _write(tmp_path, "long.flac", 65.0, gaps=(28.0, 60.0))
    wins = train.plan_windows(p, 30.0, 3.0, max_windows=0)
    assert len(wins) == 3  # ceil(65/30)
    # first window length = first cut -> snaps into the 28s gap, not 30
    assert abs(wins[0][1] - 28.0) < 1.0


def test_window_specs_partitions_onsets_without_loss(tmp_path):
    pytest.importorskip("soundfile")
    p = _write(tmp_path, "l2.flac", 65.0, gaps=(30.0, 60.0))
    onsets = {"k": [1.0, 31.0, 61.0], "s": [29.0]}
    out = train._window_specs([(p, onsets)], window=30.0, search=3.0, max_windows=0)
    assert len(out) == 3
    # every onset lands in exactly one window; absolute times reconstruct exactly
    recon = sorted(t + start for _a, o, _w, start, _l in out for t in o["k"])
    assert recon == [1.0, 31.0, 61.0]
    # each window's onsets are window-relative (>= 0, < its length)
    for _a, o, _w, _start, length in out:
        for ts in o.values():
            assert all(0.0 <= t < length for t in ts)


def test_window_specs_max_windows_caps_count(tmp_path):
    pytest.importorskip("soundfile")
    p = _write(tmp_path, "l3.flac", 100.0)  # ceil(100/30)=4; tail ~7-13s, never merged
    full = train._window_specs([(p, {"k": [1.0]})], 30.0, 3.0, max_windows=0)
    capped = train._window_specs([(p, {"k": [1.0]})], 30.0, 3.0, max_windows=2)
    assert len(full) == 4 and len(capped) == 2  # cap drops the tail windows


def test_plan_windows_merges_short_tail(tmp_path):
    pytest.importorskip("soundfile")
    # cuts snap to the 30s/60s gaps -> a ~3s tail [60, 63), too short for MERT
    p = _write(tmp_path, "tail.flac", 63.0, gaps=(30.0, 60.0))
    wins = train.plan_windows(p, 30.0, 3.0, max_windows=0)
    assert len(wins) == 2  # 3s sliver folded into the previous window
    assert all(length >= train.MIN_WINDOW for _s, length in wins)  # no sub-floor sliver
    assert abs((wins[-1][0] + wins[-1][1]) - 63.0) < 1e-6  # whole clip still covered


def test_window_specs_short_tail_loses_no_onsets(tmp_path):
    pytest.importorskip("soundfile")
    p = _write(tmp_path, "tail2.flac", 63.0, gaps=(30.0, 60.0))
    onsets = {"k": [1.0, 31.0, 61.5]}  # 61.5 is in the merged tail region
    out = train._window_specs([(p, onsets)], window=30.0, search=3.0, max_windows=0)
    assert len(out) == 2
    recon = sorted(t + start for _a, o, _w, start, _l in out for t in o["k"])
    assert recon == [1.0, 31.0, 61.5]  # merged tail keeps its onset, reconstructs exactly


def test_plan_cache_skips_audio_reread(tmp_path):
    pytest.importorskip("soundfile")
    p = _write(tmp_path, "pc.flac", 65.0, gaps=(30.0, 60.0))
    cache = tmp_path / "cache"
    cache.mkdir()
    out1 = train._window_specs([(p, {"k": [1.0]})], 30.0, 3.0, 0, plan_cache_dir=str(cache))
    assert (cache / "_window_plan.json").exists()
    # delete the audio: a re-plan would fail, so a matching result proves the cache hit
    import os
    os.remove(p)
    out2 = train._window_specs([(p, {"k": [1.0]})], 30.0, 3.0, 0, plan_cache_dir=str(cache))
    w1 = [(s, length) for _a, _o, _w, s, length in out1]
    w2 = [(s, length) for _a, _o, _w, s, length in out2]
    assert w1 == w2 and len(w1) >= 2


def test_plan_cache_shared_across_max_windows(tmp_path):
    pytest.importorskip("soundfile")
    p = _write(tmp_path, "pc2.flac", 95.0)  # ceil(95/30)=4 windows
    cache = tmp_path / "c2"
    cache.mkdir()
    full = train._window_specs([(p, {"k": [1.0]})], 30.0, 3.0, 0, plan_cache_dir=str(cache))  # caches full
    import os
    os.remove(p)  # now only the cache can serve
    capped = train._window_specs([(p, {"k": [1.0]})], 30.0, 3.0, 2, plan_cache_dir=str(cache))
    assert len(full) == 4 and len(capped) == 2
    wf = [(s, length) for _a, _o, _w, s, length in full]
    wc = [(s, length) for _a, _o, _w, s, length in capped]
    assert wf[:2] == wc  # capped run = first 2 of the cached full plan (no re-read)
