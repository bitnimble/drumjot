"""Unit tests for the torch-free ONNX onset glue (no models, fast).

The heavy MERT/heads parity is validated out-of-band against torch (CPU); these
cover the pure-numpy activation port and the single-layer guard.
"""

import json

import numpy as np
import pytest

from app.pipeline.onset_onnx.np_onsets import activate, load_onnx_onset


def test_activate_sigmoid_matches_reference():
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((4, 20)) * 3).astype(np.float32)
    got = activate(logits, ("k", "s", "hc", "ho"), cymbal_softmax=False)
    ref = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(got, ref, atol=1e-6)


def test_activate_cymbal_softmax_rd_cr_rows():
    lanes = ("k", "rd", "cr")
    rng = np.random.default_rng(1)
    logits = (rng.standard_normal((3, 16)) * 2).astype(np.float32)
    got = activate(logits, lanes, cymbal_softmax=True)
    # k row stays a plain sigmoid; rd/cr become the joint 3-way softmax {0, rd, cr}.
    assert np.allclose(got[0], 1.0 / (1.0 + np.exp(-logits[0])), atol=1e-6)
    denom = 1.0 + np.exp(logits[1]) + np.exp(logits[2])
    assert np.allclose(got[1], np.exp(logits[1]) / denom, atol=1e-5)
    assert np.allclose(got[2], np.exp(logits[2]) / denom, atol=1e-5)


def test_load_onnx_onset_rejects_per_lane_layer(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps({"encoder_layer": 10, "lane_layers": {"k": 8, "s": 10}})
    )
    with pytest.raises(NotImplementedError):
        load_onnx_onset(tmp_path)
