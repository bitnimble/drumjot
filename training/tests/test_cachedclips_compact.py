"""CachedClips compacts per-window onset/weight float lists to array.array('d') to cut
resident RAM at the ~170k-window scale -- bit-exact (float64) so targets/values are
unchanged, and without breaking the list-like consumers (iterate / len / truthiness)."""
import array

import numpy as np

from drumjot_training.config import Config
from drumjot_training.train import CachedClips, build_targets


def test_onsets_compacted_to_array_in_place(tmp_path):
    cfg = Config(lanes=("k", "s"), encoder_fps=75.0)
    onsets = {"k": [0.1, 0.5, 1.0], "s": [0.5]}
    weight = {"k": [0.1, 0.5, 1.0], "s": [0.5], "t": [0.2]}
    spec = ("/x/a.flac", onsets, weight, {}, 150, 0.0, 2.0)
    ref = build_targets({"k": [0.1, 0.5, 1.0], "s": [0.5]}, 150, cfg)  # from plain lists

    cc = CachedClips([spec], cfg, tmp_path, 2.0)

    # converted in place to array.array (both onsets and weight_onsets, every lane)
    assert isinstance(cc._specs[0][1]["k"], array.array)
    assert isinstance(cc._specs[0][2]["t"], array.array)
    # float64 ('d') keeps values bit-exact -- no precision drift
    assert list(cc._specs[0][1]["k"]) == [0.1, 0.5, 1.0]
    # targets identical (float32 onset times land on the same frames)
    got = build_targets(cc._specs[0][1], 150, cfg)
    assert np.array_equal(got, ref)
    # list-like truthiness preserved (the `if onsets.get(lane):` idiom consumers rely on)
    assert bool(cc._specs[0][1]["k"]) is True
    assert bool(array.array("f", [])) is False


def test_weight_none_untouched(tmp_path):
    cfg = Config(lanes=("k",), encoder_fps=75.0)
    spec = ("/x/a.flac", {"k": [0.2]}, None, {}, 100, 0.0, 2.0)  # full-mix: weight is None
    cc = CachedClips([spec], cfg, tmp_path, 2.0)
    assert isinstance(cc._specs[0][1]["k"], array.array)
    assert cc._specs[0][2] is None
