"""Per-clip robust peak-normalisation (the `input_norm` opt-in) + its cache-key token."""
import numpy as np

from drumjot_training import embeddings


def test_robust_peak_normalize_scales_to_unit_percentile():
    rng = np.random.default_rng(0)
    y = (0.02 * rng.standard_normal(48000)).astype(np.float32)  # quiet clip
    out = embeddings.robust_peak_normalize(y)
    assert np.isclose(np.percentile(np.abs(out), 99.5), 1.0, atol=1e-3)


def test_robust_peak_normalize_ignores_single_spike():
    rng = np.random.default_rng(1)
    y = (0.3 * rng.standard_normal(48000)).astype(np.float32)
    clean = embeddings.robust_peak_normalize(y.copy())
    y[1234] = 500.0  # one separation-artifact spike, far above the 99.5th pct
    spiked = embeddings.robust_peak_normalize(y)
    # the spike sits above the 99.5th percentile, so it barely moves the scale --
    # the bulk still lands at ~unit level (a naive max-normalize would divide the
    # whole clip by 500 and crush it to ~0).
    assert np.allclose(clean[:1234], spiked[:1234], rtol=1e-3)
    assert np.isclose(np.percentile(np.abs(spiked), 99.5), 1.0, atol=1e-3)
    assert float(np.percentile(np.abs(spiked), 50)) > 0.1  # not crushed by the spike


def test_robust_peak_normalize_empty_and_silent():
    assert embeddings.robust_peak_normalize(np.zeros(0, dtype=np.float32)).size == 0
    silent = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(embeddings.robust_peak_normalize(silent), silent)  # scale ~0 -> unchanged


def test_feat_variant_input_norm_token():
    # OFF is byte-identical to the pre-norm default (existing caches/keys unchanged)
    assert embeddings.feat_variant(True, False) == "hb16" == embeddings.FEAT_VARIANT
    assert embeddings.feat_variant(True) == "hb16"
    assert embeddings.feat_variant(False, False) == ""
    # ON appends "_pn" so normalised and raw-level caches never collide
    assert embeddings.feat_variant(True, input_norm=True) == "hb16_pn"
    assert embeddings.feat_variant(True, True) != embeddings.feat_variant(True, False)
    assert embeddings.feat_variant(False, True) == "_pn"
