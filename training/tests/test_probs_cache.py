import numpy as np

from drumjot_training.parampred import probs_cache


def test_probs_key_is_stable_and_recipe_sensitive():
    a = probs_cache.probs_key("/x/y.flac", "identity", encoder="mert", layer=10, in_dim=1040,
                              max_seconds=45.0, window_seconds=30.0)
    b = probs_cache.probs_key("/x/y.flac", "identity", encoder="mert", layer=10, in_dim=1040,
                              max_seconds=45.0, window_seconds=30.0)
    c = probs_cache.probs_key("/x/y.flac", "v1:gain+3dB", encoder="mert", layer=10, in_dim=1040,
                              max_seconds=45.0, window_seconds=30.0)
    assert a == b           # deterministic
    assert a != c           # recipe changes the key
    assert len(a) == 40     # sha1 hex


def test_save_load_round_trips_probs(tmp_path):
    probs = np.random.default_rng(0).random((5, 200)).astype(np.float32)
    key = "deadbeef"
    probs_cache.save_probs(tmp_path, key, probs, fps=75.0)
    assert probs_cache.load_probs(tmp_path, key) is not None
    got, fps = probs_cache.load_probs(tmp_path, key)
    assert fps == 75.0
    assert got.shape == probs.shape
    np.testing.assert_allclose(got, probs, atol=1e-3)   # fp16 storage tolerance


def test_load_miss_returns_none(tmp_path):
    assert probs_cache.load_probs(tmp_path, "nope") is None


def test_window_plan_key_matches_train_format():
    assert probs_cache.window_plan_key("/a/b.flac", 30.0, 3.0) == "/a/b.flac|30.0|3.0"


def test_window_onsets_clips_and_shifts():
    gt = {"hc": [0.5, 31.0, 35.0], "rd": [10.0]}
    w = probs_cache.window_onsets(gt, start=30.0, length=10.0)  # [30, 40)
    assert w["hc"] == [1.0, 5.0]   # 31->1, 35->5; 0.5 dropped (before window)
    assert w["rd"] == []           # 10.0 outside the window


def test_variant_audio_is_deterministic_and_identity_at_zero():
    sr = 44100
    wave = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    w0, r0 = probs_cache.variant_audio("/x/stem.flac", 0, wave, sr, codec=False)
    assert r0 == "identity"
    np.testing.assert_array_equal(w0, wave)
    # same (stem, variant) -> identical audio + recipe across calls (stable cache key)
    a, ra = probs_cache.variant_audio("/x/stem.flac", 3, wave, sr, codec=False)
    b, rb = probs_cache.variant_audio("/x/stem.flac", 3, wave, sr, codec=False)
    assert ra == rb and ra.startswith("v3:")
    np.testing.assert_array_equal(a, b)
    # a different variant index gives a different recipe
    _, rc = probs_cache.variant_audio("/x/stem.flac", 4, wave, sr, codec=False)
    assert rc != ra
