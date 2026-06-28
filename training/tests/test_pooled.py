import argparse

from drumjot_training.lanes import LANES


def _enst_sep_tree(tmp_path):
    """Minimal ENST sep tree: drummer_1 (train) + drummer_3 (val), 5 perstem each."""
    for d in ("drummer_1", "drummer_3"):
        base = tmp_path / d
        (base / "annotation").mkdir(parents=True)
        (base / "annotation" / "t.txt").write_text("0.10 bd\n0.20 sd\n0.30 cr\n0.40 rc\n")
        for p in ("k", "s", "h", "c", "t"):
            pd = base / "audio" / "perstem" / p
            pd.mkdir(parents=True)
            (pd / "t.flac").write_bytes(b"")
    return tmp_path


def test_cap_by_windows(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import pytest

    pytest.importorskip("soundfile")
    import soundfile as sf

    from drumjot_training import train

    def clip(name, secs):
        p = tmp_path / name
        sf.write(str(p), np.zeros(int(secs * 24000), dtype="float32"), 24000)
        return SimpleNamespace(audio_path=str(p))

    c90 = [clip(f"{i}.flac", 90.0) for i in range(3)]  # 90/30 = 3 windows each (no tail merge)
    assert train._cap_by_windows(c90, 0) == c90        # 0 = keep all
    assert train._cap_by_windows(c90, 5) == c90[:2]    # 3+3 >= 5 -> first 2 clips
    assert train._cap_by_windows(c90, 3) == c90[:1]    # 3 >= 3 -> first 1
    # short tail merges: a 63s clip is 2 windows (not 3), so cap-3 needs a 2nd clip
    short = [clip("a63.flac", 63.0), clip("b90.flac", 90.0)]
    assert train._cap_by_windows(short, 3) == short    # 2 (merged) < 3 -> add the 90s clip


def test_pooled_specs_single_source(tmp_path, monkeypatch):
    from drumjot_training import train

    root = _enst_sep_tree(tmp_path)
    monkeypatch.setenv("DRUMJOT_ENST", str(root))
    # --pool-cache is honored: the returned cache dir is the one we pass in.
    pool_cache = tmp_path / "mert_cache"
    args = argparse.Namespace(
        pool_sources="enst", pool_cap=0, pool_balance=False, pool_cache=str(pool_cache)
    )
    tr, va, cache = train._pooled_specs(args)
    assert len(tr) == 5 and len(va) == 5            # drummer_1 train, drummer_3 val; 5 stems each
    assert len(tr[0]) == 3                          # (audio, restricted_onsets, full_onsets)
    _audio, restr, full = tr[0]
    # restricted = this stem's lanes; full = all output lanes (for sibling weighting)
    assert set(restr) == set(LANES) and set(full) == set(LANES)
    assert cache.name == "mert_cache"


def test_pooled_balance_oversamples_small_source(tmp_path, monkeypatch):
    from drumjot_training import train

    root = _enst_sep_tree(tmp_path)
    monkeypatch.setenv("DRUMJOT_ENST", str(root))
    # single source can't be drowned, but exercise the cap + balance code paths.
    # cap is in WINDOWS now: the empty-flac fixtures estimate 1 window each, so
    # cap=5 windows -> drummer_1's 5 perstem clips (balance no-op alone).
    args = argparse.Namespace(pool_sources="enst", pool_cap=5, pool_balance=True)
    tr, _va, _cache = train._pooled_specs(args)
    assert len(tr) == 5


def test_parse_pool_caps():
    from drumjot_training import train

    # per-source spec
    assert train._parse_pool_caps("paradb:9000,egmd:2500,star:2000") == {
        "paradb": 9000, "egmd": 2500, "star": 2000,
    }
    # bare int -> uniform "*" cap (string and actual int both work)
    assert train._parse_pool_caps("3000") == {"*": 3000}
    assert train._parse_pool_caps(3000) == {"*": 3000}
    # empty / None -> no caps
    assert train._parse_pool_caps("") == {}
    assert train._parse_pool_caps(None) == {}

    # _cap_for: explicit hit, then "*" fallback, then 0 (no cap)
    caps = {"paradb": 9000, "*": 1000}
    assert train._cap_for(caps, "paradb") == 9000   # explicit
    assert train._cap_for(caps, "egmd") == 1000     # "*" fallback
    assert train._cap_for({"paradb": 9000}, "egmd") == 0  # no "*", no entry -> 0


def test_per_source_resampler_caps_and_resamples():
    from drumjot_training import train

    source_indices = {
        "paradb": list(range(0, 100)),
        "egmd": list(range(100, 140)),
        "enst": list(range(140, 150)),
    }
    caps = {"paradb": 20, "egmd": 10}  # enst uncapped (no entry, no "*")
    sampler = train.PerSourceResampler(source_indices, caps, seed=7)

    assert len(sampler) == 40  # 20 + 10 + 10

    e0 = list(iter(sampler))
    assert len(e0) == len(sampler) == 40
    pdb0 = [i for i in e0 if 0 <= i < 100]
    egmd0 = [i for i in e0 if 100 <= i < 140]
    enst0 = [i for i in e0 if 140 <= i < 150]
    assert len(pdb0) == 20 and len(set(pdb0)) == 20      # exactly 20 distinct paradb
    assert len(egmd0) == 10 and len(set(egmd0)) == 10    # exactly 10 distinct egmd
    assert sorted(enst0) == list(range(140, 150))        # all 10 enst (uncapped)
    assert set(pdb0) <= set(range(100))                  # valid paradb subset

    # next epoch resamples: a DIFFERENT valid 20-subset of paradb
    e1 = list(iter(sampler))
    pdb1 = [i for i in e1 if 0 <= i < 100]
    assert len(pdb1) == 20 and len(set(pdb1)) == 20
    assert set(pdb1) <= set(range(100))
    assert set(pdb0) != set(pdb1)                        # resampled, not the same slice

    # full coverage over enough epochs: union of paradb selections == all 100.
    # A 20-of-100 sample needs ~20-35 epochs to hit every index (each epoch misses
    # ~80% of any given index); 40 (fixed seed) is a comfortable, deterministic
    # margin -- the point is that EVERY window is eventually trained on.
    fresh = train.PerSourceResampler(source_indices, caps, seed=7)
    seen: set[int] = set()
    for _ in range(40):
        seen |= {i for i in iter(fresh) if 0 <= i < 100}
    assert seen == set(range(100))


def test_pooled_specs_resample_path_uncapped_train(tmp_path, monkeypatch):
    """--pool-resample materializes train UNCAPPED via _per_source_specs(cap_train=None);
    the per-source dicts preserve --pool-sources order."""
    from drumjot_training import train

    root = _enst_sep_tree(tmp_path)
    monkeypatch.setenv("DRUMJOT_ENST", str(root))
    args = argparse.Namespace(
        pool_sources="enst", pool_cap="2", pool_balance=False,
        pool_cache=str(tmp_path / "mert_cache"), pool_val_cap=0,
    )
    # cap_train=None -> no cap applied regardless of pool_cap; all 5 train stems kept.
    per_train, per_val, _cache = train._per_source_specs(args, cap_train=None, cap_val=None)
    assert list(per_train) == ["enst"]
    assert len(per_train["enst"]) == 5 and len(per_val["enst"]) == 5


def test_leakage_from_probs_counts_wrong_lane_firing():
    """Cross-instrument leak: a fire in a lane the stem doesn't own (plays in the
    song via weight_targets, absent in onsets_by_lane) counts as leaked; a fire in
    an owned lane counts as matched. Full-mix clips (weight_targets=None) -> (0,0)."""
    import numpy as np

    from drumjot_training import train
    from drumjot_training.config import Config

    cfg = Config()
    n_lanes, T = len(cfg.lanes), 100
    probs = np.zeros((n_lanes, T), dtype=np.float32)

    def _bump(row, center):  # a clean onset the picker will pick at thr 0.5
        for off, v in ((-2, 0.3), (-1, 0.7), (0, 1.0), (1, 0.7), (2, 0.3)):
            probs[row, center + off] = v

    s_i, cr_i = cfg.lanes.index("s"), cfg.lanes.index("cr")
    _bump(s_i, 50)   # model fires snare ...
    _bump(cr_i, 30)  # ... and crash
    wt = np.zeros((n_lanes, T), dtype=np.float32)
    wt[s_i, 50] = 1.0   # snare DOES play in the song (full kit)
    wt[cr_i, 30] = 1.0  # crash too
    # a cymbal stem: owns crash (has a cr onset), does NOT own snare
    clip = train.Clip(features=np.zeros((T, 1), dtype=np.float32),
                      targets=np.zeros((n_lanes, T), dtype=np.float32),
                      onsets_by_lane={"cr": [30 / cfg.encoder_fps]}, weight_targets=wt)
    matched, leaked = train._leakage_from_probs(probs, clip, cfg)
    assert matched == 1   # crash (owned) fired once
    assert leaked == 1    # snare (plays in song, not on this stem) -> leaked

    full_mix = train.Clip(features=clip.features, targets=clip.targets,
                          onsets_by_lane={"cr": [30 / cfg.encoder_fps]}, weight_targets=None)
    assert train._leakage_from_probs(probs, full_mix, cfg) == (0, 0)
