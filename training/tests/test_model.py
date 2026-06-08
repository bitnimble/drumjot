import pytest

# torch is the CUDA build in the transcriber venv; it can't load on the
# host (no CUDA libs). Skip on the host; this runs where torch imports
# (the sandbox / a CUDA box). A standalone sandbox check covers it too.
pytest.importorskip("torch", exc_type=ImportError)

import drumjot_training.lanes as lanes  # noqa: E402
import drumjot_training.model as model  # noqa: E402


def test_onset_head_outputs_one_logit_per_frame():
    import torch

    head = model.OnsetHead(in_dim=16, hidden=8, num_layers=1)
    x = torch.zeros(2, 30, 16)  # (batch, time, feat)
    y = head(x)
    assert y.shape == (2, 30)


def test_multilane_heads_output_per_lane_per_frame():
    import torch

    m = model.MultiLaneHeads(in_dim=16, hidden=8, num_layers=1)
    x = torch.zeros(2, 30, 16)
    y = m(x)
    assert y.shape == (2, len(lanes.LANES), 30)


def test_multilane_heads_are_separate_modules_per_lane():
    # Decision (spec §4): separate per-lane heads, not one shared head.
    m = model.MultiLaneHeads(in_dim=16, hidden=8, num_layers=1)
    assert set(m.heads) == set(lanes.LANES)
