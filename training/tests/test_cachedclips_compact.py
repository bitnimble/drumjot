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


def test_window_specs_slices_to_array_at_source(tmp_path):
    """_window_specs builds window onsets straight as array.array (no big-list peak the
    allocator would retain), bit-exact."""
    from drumjot_training.train import _window_specs
    # t's only onset (40.0s) is past the 30s window -> the whole lane should be DROPPED
    specs = [("/x/a.flac", {"k": [0.1, 0.5, 35.0], "s": [0.2], "t": [40.0]})]
    out = _window_specs(specs, 30.0, 3.0, 1)  # legacy single window, no audio read
    assert len(out) == 1
    onsets = out[0][1]
    assert isinstance(onsets["k"], array.array)
    assert list(onsets["k"]) == [0.1, 0.5]  # window-relative, 35.0 sliced out, values exact
    assert "t" not in onsets and "s" in onsets  # empty-window lane dropped (consumers .get(.,[]))


def test_activate_onsets_cymbal_softmax_no_clone():
    """activate_onsets writes the joint ride/crash softmax straight into the sigmoid output
    (no .clone()): k stays sigmoid, rd/cr become the {none=0,ride,crash} posteriors, and the
    input logits are not mutated."""
    import torch

    from drumjot_training.model import activate_onsets
    torch.manual_seed(0)
    lanes = ("k", "rd", "cr")
    logits = torch.randn(2, 3, 5)
    logits_ref = logits.clone()
    out = activate_onsets(logits, lanes, cymbal_softmax=True)
    z = torch.zeros_like(logits[:, 1])
    sm = torch.softmax(torch.stack([z, logits[:, 1], logits[:, 2]], dim=-2), dim=-2)
    assert torch.allclose(out[:, 0], torch.sigmoid(logits[:, 0]))  # k unchanged (sigmoid)
    assert torch.allclose(out[:, 1], sm[:, 1])  # rd = ride posterior
    assert torch.allclose(out[:, 2], sm[:, 2])  # cr = crash posterior
    assert torch.equal(logits, logits_ref)  # logits not mutated (we wrote into sigmoid output)


def test_weight_none_untouched(tmp_path):
    cfg = Config(lanes=("k",), encoder_fps=75.0)
    spec = ("/x/a.flac", {"k": [0.2]}, None, {}, 100, 0.0, 2.0)  # full-mix: weight is None
    cc = CachedClips([spec], cfg, tmp_path, 2.0)
    assert isinstance(cc._specs[0][1]["k"], array.array)
    assert cc._specs[0][2] is None
