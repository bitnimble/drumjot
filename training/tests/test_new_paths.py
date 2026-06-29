"""Tests for the 2026-06 batch: high-band block, ring-activity targets, and
the rare-lane threshold floor."""
import numpy as np

from drumjot_training import embeddings, targets


def test_highband_from_wave_shape_and_range():
    sr = embeddings.HB_SR
    t = np.arange(sr) / sr
    # 12 kHz tone: inside the 6-20 kHz band -> some band must light up
    y = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    n_frames = 75
    hb = embeddings.highband_from_wave(y, n_frames)
    assert hb.shape == (n_frames, embeddings.HB_BANDS)
    assert hb.dtype == np.float32
    assert float(hb.min()) >= 0.0 and float(hb.max()) <= 1.0
    assert float(hb.max()) > 0.3  # the tone is visible
    # silence -> near-zero everywhere
    hb0 = embeddings.highband_from_wave(np.zeros(sr, dtype=np.float32), n_frames)
    assert float(hb0.max()) < 0.05


def test_highband_pads_short_audio():
    hb = embeddings.highband_from_wave(np.zeros(100, dtype=np.float32), 10)
    assert hb.shape == (10, embeddings.HB_BANDS)


def test_ring_spans_follow_decay_and_next_onset():
    sr, fps = 24000, 75.0
    n = sr * 2  # 2s
    y = np.zeros(n, dtype=np.float32)
    # exponentially decaying noise burst at t=0.2 with ~0.3s decay
    t0 = int(0.2 * sr)
    dur = int(0.6 * sr)
    env = np.exp(-np.arange(dur) / (0.1 * sr))
    rng = np.random.default_rng(0)
    y[t0 : t0 + dur] = (env * rng.standard_normal(dur) * 0.5).astype(np.float32)
    spans = targets.ring_spans(y, sr, [0.2], fps)
    assert len(spans) == 1
    t, d = spans[0]
    assert t == 0.2
    assert 0.1 < d < 1.0  # ring ends when the envelope decays, well before 2s

    # a second onset truncates the first ring
    spans2 = targets.ring_spans(y, sr, [0.2, 0.3], fps)
    assert spans2[0][1] <= 0.11  # capped at the next onset (0.1s later)


def test_spans_to_activity_renders_rects():
    act = targets.spans_to_activity([(0.1, 0.2)], n_frames=100, fps=100.0)
    assert act[5] == 0.0 and act[15] == 1.0 and act[29] == 1.0 and act[40] == 0.0


def test_sustained_lanes_subset_of_vocab():
    from drumjot_training.lanes import LANES

    assert set(targets.SUSTAINED_LANES) <= set(LANES)


def test_tune_thresholds_floors_rare_lanes():
    import torch

    from drumjot_training.config import Config
    from drumjot_training.train import Clip, tune_thresholds

    cfg = Config(encoder_fps=100.0)
    nl = len(cfg.lanes)

    class _Probs(torch.nn.Module):
        # a "model" whose sigmoid output is ~1 at a few frames for every lane
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))  # _clip_probs reads device
            self.lane_names = cfg.lanes  # activate_onsets (via _clip_probs_batched) needs it

        def forward(self, x):
            out = torch.full((x.shape[0], nl, x.shape[1]), -8.0)
            out[:, :, 10] = 8.0
            return out

    feat = np.zeros((50, 4), dtype=np.float32)
    # rd: 1 onset (rare -> floored grid); k: same onset but pretend many via grid
    onsets = {"rd": [0.10], "k": [0.10]}
    clip = Clip(
        features=feat,
        targets=np.zeros((nl, 50), dtype=np.float32),
        onsets_by_lane=onsets,
    )
    thr = tune_thresholds(_Probs(), [clip], cfg)
    # both lanes are "rare" here (1 onset < 50): no tuned threshold below floor
    assert thr["rd"] >= cfg.rare_thr_floor
    assert thr["k"] >= cfg.rare_thr_floor
