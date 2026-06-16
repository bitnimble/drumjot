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
    args = argparse.Namespace(pool_sources="enst", pool_cap=0, pool_balance=False)
    tr, va, cache = train._pooled_specs(args)
    assert len(tr) == 5 and len(va) == 5            # drummer_1 train, drummer_3 val; 5 stems each
    assert len(tr[0]) == 3                          # (audio, restricted_onsets, full_onsets)
    _audio, restr, full = tr[0]
    # restricted = this stem's lanes; full = all output lanes (for sibling weighting)
    assert set(restr) == set(LANES) and set(full) == set(LANES)
    assert cache.name == "_cache_mert_pooled"


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
