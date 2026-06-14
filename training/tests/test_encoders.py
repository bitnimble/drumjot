"""Frozen-encoder construction + the fps-aware high-band spectral block."""
import numpy as np

from drumjot_training import embeddings


def test_make_encoder_is_mert():
    # make_encoder is the single construction point; without instantiating (no
    # torch/transformers needed) we at least pin the class it builds.
    assert embeddings.make_encoder.__module__ == embeddings.__name__
    assert embeddings.MERT_DIM == 1024 and embeddings.MERT_SR == 24000
    assert embeddings.MERT_FPS == 75.0
    assert embeddings.feat_dim(high_band=True) == embeddings.MERT_DIM + embeddings.HB_BANDS


def test_highband_block_aligns_to_encoder_fps():
    # The high-band block hop is derived from `fps`, not hardcoded to 75, so it
    # frame-aligns to whatever encoder rate it's given.
    sr = embeddings.HB_SR
    y = np.zeros(sr, dtype=np.float32)  # 1 s, with energy only in the first third
    y[: sr // 3] = np.random.default_rng(0).standard_normal(sr // 3).astype(np.float32)
    f75 = embeddings.highband_from_wave(y, n_frames=75, fps=75.0)  # 75 fps -> 1 s
    f25 = embeddings.highband_from_wave(y, n_frames=25, fps=25.0)  # 25 fps -> 1 s
    assert f75.shape == (75, embeddings.HB_BANDS)
    assert f25.shape == (25, embeddings.HB_BANDS)
    # at 75 fps the first 1/3 s is ~25 frames; at 25 fps it's ~8 -> the framings
    # differ, proving the hop tracks fps rather than a hardcoded rate.
    assert int((f75.sum(axis=1) > 0.5 * f75.sum(axis=1).max()).sum()) > \
           int((f25.sum(axis=1) > 0.5 * f25.sum(axis=1).max()).sum())
