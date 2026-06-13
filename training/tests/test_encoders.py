"""Encoder selection + fps-aware spectral blocks (MERT vs MuQ pathway)."""
import numpy as np

from drumjot_training import embeddings


def test_encoder_class_dispatch():
    # name -> implementation, WITHOUT instantiating (so it's importable/testable
    # without torch/transformers/muq present).
    assert embeddings._encoder_class(embeddings.MUQ_NAME) is embeddings.MuQEncoder
    assert embeddings._encoder_class("OpenMuQ/MuQ-large-msd-iter") is embeddings.MuQEncoder
    assert embeddings._encoder_class(embeddings.MERT_NAME) is embeddings.MertEncoder
    assert embeddings._encoder_class("m-a-p/MERT-v1-95M") is embeddings.MertEncoder


def test_muq_shares_mert_dim_and_sr():
    # MuQ is 1024-dim @ 24 kHz like MERT, so the model input width is identical
    # (feat_dim unchanged) and the only knock-on is the frame rate.
    assert embeddings.MUQ_DIM == embeddings.MERT_DIM == 1024
    assert embeddings.MUQ_SR == embeddings.MERT_SR == 24000
    assert embeddings.MUQ_FPS == 25.0 and embeddings.MERT_FPS == 75.0
    assert embeddings.feat_dim(high_band=True, cym=False) == embeddings.MERT_DIM + embeddings.HB_BANDS


def test_highband_block_aligns_to_encoder_fps():
    sr = embeddings.HB_SR
    y = np.zeros(sr, dtype=np.float32)  # 1 s, with energy only in the first third
    y[: sr // 3] = np.random.default_rng(0).standard_normal(sr // 3).astype(np.float32)
    f75 = embeddings.highband_from_wave(y, n_frames=75, fps=75.0)  # 75 fps -> 1 s
    f25 = embeddings.highband_from_wave(y, n_frames=25, fps=25.0)  # 25 fps -> 1 s
    assert f75.shape == (75, embeddings.HB_BANDS)
    assert f25.shape == (25, embeddings.HB_BANDS)
    # at 75 fps the first 1/3 s is ~25 frames; at 25 fps it's ~8 -> the framings
    # differ, proving the hop tracks fps rather than a hardcoded 75.
    assert int((f75.sum(axis=1) > 0.5 * f75.sum(axis=1).max()).sum()) > \
           int((f25.sum(axis=1) > 0.5 * f25.sum(axis=1).max()).sum())
