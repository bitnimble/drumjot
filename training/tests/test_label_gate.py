import numpy as np
import soundfile as sf

from drumjot_training import train
from drumjot_training.config import Config


def _stem_with_clicks(path, sr=44100, dur=2.0, clicks=(0.5, 1.0, 1.5)):
    rng = np.random.default_rng(0)
    y = (0.02 * rng.standard_normal(int(sr * dur))).astype(np.float32)  # low noise floor
    for t in clicks:
        i = int(t * sr)
        y[i:i + 64] += np.hanning(64).astype(np.float32)               # a sharp transient
    sf.write(str(path), y, sr)


def test_clean_window_keeps_aligned_labels_and_caches(tmp_path):
    stem = tmp_path / "h.flac"
    _stem_with_clicks(stem)
    cache = tmp_path / "cache"
    cache.mkdir()
    cfg = Config()  # label_min_support 0.95, window 0.04 by default
    onsets = {"hc": [0.5, 1.0, 1.5]}                                   # on the transients
    out, keep = train._clean_window_labels(stem, onsets, cfg, cache, length=2.0, start=0.0)
    assert keep                                                       # well-aligned -> kept
    assert len(out["hc"]) == 3                                        # onsets retained (snapped)
    assert all(abs(t - s) < 0.04 for t, s in zip(sorted(out["hc"]), [0.5, 1.0, 1.5], strict=True))
    assert list(cache.glob("*.support.*.json"))                      # side-cache written
    # second call resolves from the cache (no audio) and returns the same thing
    out2, keep2 = train._clean_window_labels(stem, onsets, cfg, cache, length=2.0, start=0.0)
    assert keep2 and out2["hc"] == out["hc"]


def test_changing_gate_params_invalidates_the_cache(tmp_path):
    stem = tmp_path / "h.flac"
    _stem_with_clicks(stem)
    cache = tmp_path / "cache"
    cache.mkdir()
    onsets = {"hc": [0.5, 1.0, 1.5]}

    base = Config()
    train._clean_window_labels(stem, onsets, base, cache, length=2.0, start=0.0)
    # a different gate param must NOT reuse the prior verdict -> a distinct cache file
    tighter = Config(label_support_window_s=0.02)
    train._clean_window_labels(stem, onsets, tighter, cache, length=2.0, start=0.0)
    files = {p.name for p in cache.glob("*.support.*.json")}
    assert len(files) == 2                                            # one per param set
    # and the algorithm version is baked into the filename
    assert all(f".support.v{train._SUPPORT_CACHE_VERSION}-" in f for f in files)
