"""Dropped-percussion hard negatives: onsets that map to no output lane (the
removed `mp` + non-kit aux perc) are carried as the catch-all `x` lane and fed to
the loss as hard negatives for every output lane, without ever being predicted."""
import numpy as np
import torch

from drumjot_training import lanes
from drumjot_training.config import Config
from drumjot_training.model import MultiLaneHeads
from drumjot_training.train import (
    Clip,
    build_negative_targets,
    collate_clips,
    train_loop,
)


def test_build_negative_targets_renders_x_lane():
    cfg = Config(encoder_fps=100.0)
    onsets = {"k": [0.0], "x": [0.5]}  # x = dropped percussion
    neg = build_negative_targets(onsets, n_frames=100, cfg=cfg)
    assert neg.shape == (len(lanes.NEGATIVE_LANES), 100)
    assert int(neg[0].argmax()) == 50  # Gaussian bump centered on 0.5 s @ 100 fps
    # no `x` key -> all-zero negative target (graceful: no head, just no signal)
    assert float(build_negative_targets({"k": [0.0]}, 100, cfg).sum()) == 0.0


def test_collate_carries_negative_targets():
    nl = len(Config().lanes)
    nneg = len(lanes.NEGATIVE_LANES)
    neg = np.zeros((nneg, 6), dtype=np.float32)
    neg[0, 2] = 1.0
    c = Clip(
        features=np.zeros((6, 4), dtype=np.float32),
        targets=np.zeros((nl, 6), dtype=np.float32),
        onsets_by_lane={},
        negative_targets=neg,
    )
    *_, Aneg, _ = collate_clips([c])  # (X, Y, Yw, A, Aneg, mask)
    assert Aneg.shape == (1, nneg, 6)
    assert float(Aneg[0, 0, 2]) == 1.0


def test_dropped_negatives_change_the_weighted_loss():
    # A dropped-percussion hit on otherwise-silent output lanes is a HARD NEGATIVE:
    # with use_dropped_neg on, those frames' loss is up-weighted, so the epoch loss
    # differs from the same run with the feature off (identical model/data/shuffle).
    nl = len(Config().lanes)
    nneg = len(lanes.NEGATIVE_LANES)

    def _clips():
        out = []
        for _ in range(2):
            neg = np.zeros((nneg, 20), dtype=np.float32)
            neg[0, 5:15] = 1.0  # sustained dropped-perc activity
            out.append(Clip(
                features=np.ones((20, 8), dtype=np.float32),
                targets=np.zeros((nl, 20), dtype=np.float32),  # all output lanes silent
                onsets_by_lane={},
                negative_targets=neg,
            ))
        return out

    def _loss0(use_neg: bool) -> float:
        cfg = Config(encoder_fps=100.0, use_dropped_neg=use_neg)
        torch.manual_seed(0)
        model = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1, lane_names=cfg.lanes)
        hist = train_loop(model, _clips(), cfg, epochs=1, batch_size=2, log=lambda s: None)
        return hist["train_loss"][0]

    assert _loss0(True) != _loss0(False)
