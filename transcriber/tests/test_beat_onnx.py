"""Unit tests for the torch-free Beat This! ONNX glue (no model, fast).

The heavy transformer parity is validated out-of-band against File2Beats (CPU);
these cover the numpy postprocessing + frontend shape.
"""

import numpy as np

from app.pipeline.beat_onnx import _deduplicate, _logmel, _mel_fb, _peaks, _postp


def test_deduplicate_merges_adjacent_by_running_mean():
    assert np.allclose(_deduplicate(np.array([10, 11, 20]), width=1), [10.5, 20.0])


def test_deduplicate_empty():
    assert _deduplicate(np.array([], dtype=int)).size == 0


def test_peaks_picks_local_maxima_over_zero_logit():
    logits = np.full(100, -1.0)
    logits[50] = 3.0
    logits[80] = 2.0
    assert np.allclose(sorted(_peaks(logits)), [50 / 50, 80 / 50])


def test_postp_snaps_downbeats_to_nearest_beat():
    beat = np.full(100, -1.0)
    beat[10] = beat[30] = beat[50] = 2.0
    downbeat = np.full(100, -1.0)
    downbeat[31] = 2.0  # one frame past the beat at 30
    bt, dt = _postp(beat, downbeat)
    assert np.allclose(bt, [10 / 50, 30 / 50, 50 / 50])
    assert np.allclose(dt, [30 / 50])  # snapped 31 -> 30, deduped


def test_logmel_shape():
    mel = _logmel(np.zeros(22050, dtype=np.float32), _mel_fb())
    assert mel.ndim == 2 and mel.shape[1] == 128 and mel.shape[0] > 0
