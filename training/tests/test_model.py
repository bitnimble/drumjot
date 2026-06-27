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


def test_activate_onsets_sigmoid_by_default():
    import torch

    logits = torch.randn(2, len(lanes.LANES), 12)
    p = model.activate_onsets(logits, lanes.LANES, cymbal_softmax=False)
    assert torch.allclose(p, torch.sigmoid(logits))


def test_activate_onsets_cymbal_softmax_couples_ride_crash():
    import torch

    ln = lanes.LANES
    logits = torch.randn(2, len(ln), 12)
    p = model.activate_onsets(logits, ln, cymbal_softmax=True)
    rd, cr = ln.index("rd"), ln.index("cr")
    other = next(i for i, x in enumerate(ln) if x not in ("rd", "cr"))
    # non-cymbal lanes stay independent sigmoids
    assert torch.allclose(p[:, other], torch.sigmoid(logits[:, other]))
    # rd/cr are a joint softmax over {none=0, ride, crash}: P(ride)+P(crash) <= 1
    assert (p[:, rd] + p[:, cr] <= 1.0 + 1e-5).all()
    sm = torch.softmax(
        torch.stack([torch.zeros_like(logits[:, rd]), logits[:, rd], logits[:, cr]], dim=1), dim=1)
    assert torch.allclose(p[:, rd], sm[:, 1], atol=1e-5)
    assert torch.allclose(p[:, cr], sm[:, 2], atol=1e-5)


def test_masked_cymbal_ce_lower_when_logits_match_target():
    import torch

    from drumjot_training.train import masked_cymbal_ce

    b, t = 2, 60
    tgt = torch.zeros(b, 2, t)  # (ride, crash)
    tgt[:, 0, 12] = 1.0  # a ride onset
    tgt[:, 1, 40] = 1.0  # a crash onset
    mask = torch.ones(b, t)
    good = torch.full((b, 2, t), -5.0)  # confident: ride@12, crash@40, none elsewhere
    good[:, 0, 12] = 5.0
    good[:, 1, 40] = 5.0
    bad = torch.zeros(b, 2, t)  # uniform over {none, ride, crash}
    lg = masked_cymbal_ce(good, tgt, mask, pos_weight=10.0)
    lb = masked_cymbal_ce(bad, tgt, mask, pos_weight=10.0)
    assert torch.isfinite(lg) and torch.isfinite(lb)
    assert lg < lb
