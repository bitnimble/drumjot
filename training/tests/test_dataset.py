import numpy as np

import drumjot_training.targets as targets
from drumjot_training.parampred import dataset, features, oracle


def _two_lane_probs(fps, n):
    k = targets.onsets_to_target([0.5, 1.5, 2.5], n_frames=n, fps=fps, sigma_frames=1.0)
    s = np.maximum(
        targets.onsets_to_target([0.6, 1.6, 2.6], n_frames=n, fps=fps, sigma_frames=1.0),
        targets.onsets_to_target([1.1, 2.1, 3.1], n_frames=n, fps=fps, sigma_frames=1.0) * 0.35,
    )
    return np.stack([k, s])


def _wave():
    sr = 44100
    return np.sin(2 * np.pi * 1000 * np.arange(sr) / sr).astype(np.float32), sr


def test_build_rows_for_song_yields_one_row_per_lane_with_gt():
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    wave, sr = _wave()
    rows = dataset.build_rows_for_song(
        probs, fps, ["k", "s"], {"k": 0.3, "s": 0.1}, {"k": [0.5, 1.5, 2.5], "s": [0.6, 1.6, 2.6]},
        wave, sr, song_id="song1", aug="identity",
    )
    assert {r.lane for r in rows} == {"k", "s"}
    r = next(r for r in rows if r.lane == "s")
    assert r.features.shape == (len(features.FEATURE_NAMES),)
    assert r.params.shape == (len(oracle.PARAM_NAMES),)
    assert r.oracle_f1 >= r.baseline_f1
    assert r.song == "song1" and r.aug == "identity"


def test_clean_lane_does_not_mark_decay_params_swept():
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    wave, sr = _wave()
    rows = dataset.build_rows_for_song(
        probs, fps, ["k"], {"k": 0.3}, {"k": [0.5, 1.5, 2.5]}, wave, sr, song_id="s", aug="a",
    )
    swept = dict(zip(oracle.PARAM_NAMES, rows[0].swept, strict=True))
    assert swept["threshold"] and swept["prominence"]
    assert not swept["decay_reset_frac"]   # clean lane: decay held at seed, not a target


def test_table_save_load_round_trip(tmp_path):
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    wave, sr = _wave()
    rows = dataset.build_rows_for_song(
        probs, fps, ["k", "s"], {"k": 0.3, "s": 0.1}, {"k": [0.5, 1.5, 2.5], "s": [0.6, 1.6, 2.6]},
        wave, sr, song_id="s", aug="a",
    )
    table = dataset.Table.from_rows(rows)
    path = tmp_path / "t.npz"
    table.save(path)
    loaded = dataset.Table.load(path)
    assert loaded.feature_names == features.FEATURE_NAMES
    np.testing.assert_allclose(loaded.X, table.X)
    np.testing.assert_array_equal(loaded.lane, table.lane)


def _rows(song_id):
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    wave, sr = _wave()
    return dataset.build_rows_for_song(
        probs, fps, ["k", "s"], {"k": 0.3, "s": 0.1},
        {"k": [0.5, 1.5, 2.5], "s": [0.6, 1.6, 2.6]}, wave, sr, song_id=song_id, aug="a")


def test_concat_stacks_tables_and_preserves_songs():
    a = dataset.Table.from_rows(_rows("synthA") + _rows("synthB"))
    b = dataset.Table.from_rows(_rows("a2md_1"))
    cat = dataset.Table.concat([a, b])
    assert len(cat) == len(a) + len(b)
    assert set(cat.song.tolist()) == {"synthA", "synthB", "a2md_1"}
    assert cat.feature_names == a.feature_names and cat.param_names == a.param_names
    np.testing.assert_array_equal(cat.X[: len(a)], a.X)


def test_concat_rejects_mismatched_feature_names():
    a = dataset.Table.from_rows(_rows("s"))
    b = dataset.Table.from_rows(_rows("t"))
    b.feature_names = a.feature_names[:-1] + ("bogus",)  # plain class; mutate directly
    try:
        dataset.Table.concat([a, b])
        raise AssertionError("expected ValueError on mismatched feature names")
    except ValueError:
        pass


def test_training_matrices_select_lane_and_swept_params():
    fps = 100.0
    probs = _two_lane_probs(fps, 400)
    wave, sr = _wave()
    rows = []
    for i in range(4):  # a few "songs" so a lane has multiple rows
        rows += dataset.build_rows_for_song(
            probs, fps, ["k", "s"], {"k": 0.3, "s": 0.1},
            {"k": [0.5, 1.5, 2.5], "s": [0.6, 1.6, 2.6]}, wave, sr, song_id=f"s{i}", aug="a",
        )
    table = dataset.Table.from_rows(rows)
    X, tgt = table.training_matrices("k")
    assert X.shape == (4, len(features.FEATURE_NAMES))
    assert "threshold" in tgt and "decay_reset_frac" not in tgt   # only swept params
    assert tgt["threshold"].shape == (4,)
