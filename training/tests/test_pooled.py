import argparse

from drumjot_training.lanes import LANES, WEIGHT_LANES


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


def test_cap_by_clip():
    from drumjot_training import train

    items = [("a", 1), ("a", 2), ("b", 1), ("c", 1), ("c", 2)]
    key = lambda x: x[0]  # noqa: E731
    assert train._cap_by_clip(items, key, 0) == items                 # 0 = keep all
    assert train._cap_by_clip(items, key, 2) == [("a", 1), ("a", 2), ("b", 1)]  # first 2 keys


def test_pooled_specs_single_source(tmp_path, monkeypatch):
    from drumjot_training import train

    root = _enst_sep_tree(tmp_path)
    monkeypatch.setenv("DRUMJOT_ENST", str(root))
    args = argparse.Namespace(pool_sources="enst", pool_cap=0, pool_balance=False)
    tr, va, cache = train._pooled_specs(args)
    assert len(tr) == 5 and len(va) == 5            # drummer_1 train, drummer_3 val; 5 stems each
    assert len(tr[0]) == 3                          # (audio, restricted_onsets, full_onsets)
    _audio, restr, full = tr[0]
    # restricted = output lanes only; full carries the `x` negative lane too
    assert set(restr) == set(LANES) and set(full) == set(WEIGHT_LANES)
    assert cache.name == "_cache_mert_pooled"


def test_pooled_balance_oversamples_small_source(tmp_path, monkeypatch):
    from drumjot_training import train

    root = _enst_sep_tree(tmp_path)
    monkeypatch.setenv("DRUMJOT_ENST", str(root))
    # single source can't be drowned, but exercise the cap + balance code paths
    args = argparse.Namespace(pool_sources="enst", pool_cap=1, pool_balance=True)
    tr, _va, _cache = train._pooled_specs(args)
    assert len(tr) == 5                              # cap=1 take -> its 5 stems (balance no-op alone)
