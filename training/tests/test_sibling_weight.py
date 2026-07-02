import numpy as np

from drumjot_training import lanes


def test_confusable_lanes_are_valid_and_not_self():
    for ln, sibs in lanes.CONFUSABLE.items():
        assert ln in lanes.LANES
        for s in sibs:
            assert s in lanes.LANES
            assert s != ln


def test_sibling_matrix_shape_and_entries():
    S = np.asarray(lanes.sibling_matrix())
    n = len(lanes.LANES)
    assert S.shape == (n, n)
    assert not S.diagonal().any()  # never self-sibling
    idx = {ln: i for i, ln in enumerate(lanes.LANES)}
    # the #1 measured confusion: hats are siblings of ride
    assert S[idx["rd"], idx["hc"]] and S[idx["rd"], idx["ho"]]
    assert S[idx["k"], idx["t"]] and not S[idx["k"], idx["cr"]]


def test_sibling_weight_torch_behaviour():
    import torch

    from drumjot_training.losses import masked_bce, sibling_weight

    # 2 lanes, 4 frames; lane0 has a positive at frame1; lane1 silent throughout
    Y = torch.tensor([[[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]).transpose(0, 1)  # (1,2,4)
    sib = torch.tensor([[[0.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]).transpose(0, 1)
    W = sibling_weight(Y, sib, pos_w=3.0, neg_w=8.0)
    assert W[0, 0, 1] == 1.0  # lane0 positive, no sibling active -> unweighted
    assert W[0, 1, 1] == 8.0  # lane1 silent while its sibling (lane0) fires -> hard negative
    assert W[0, 1, 0] == 1.0  # nothing active -> unweighted
    # co-occurring positive: sibling active AND this lane positive -> pos_w
    Wco = sibling_weight(torch.ones(1, 1, 1), torch.ones(1, 1, 1), pos_w=3.0, neg_w=8.0)
    assert float(Wco) == 3.0

    # weighted loss really is larger when the model wrongly fires on the hard negative
    logits = torch.full((1, 2, 4), -4.0)
    logits[0, 1, 1] = 4.0  # confident false positive on the hard-negative frame
    mask = torch.ones(1, 4)
    pw = torch.ones(2, 1)
    base = masked_bce(logits, Y, mask, pw)
    weighted = masked_bce(logits, Y, mask, pw, frame_weight=W)
    assert float(weighted) > float(base) * 3  # the 8x frame dominates


def test_perstem_spec_carries_full_onsets_for_weighting(tmp_path):
    from drumjot_training import star

    ann = tmp_path / "t1.txt"
    ann.write_text("0.10\tBD\t100\n0.30\tCRC\t80\n0.50\tCHH\t90\n")
    restricted = star.restricted_onsets(ann, "c")
    full = star.onsets_by_lane(ann)
    assert restricted["hc"] == [] and restricted["cr"] == [0.30]  # targets: stem lanes only
    assert full["hc"] == [0.50]  # weighting source still sees the hat bleed
