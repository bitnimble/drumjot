import os

import numpy as np
import pytest

import drumjot_training.embeddings as embeddings


def test_package_forces_hf_offline():
    # importing drumjot_training must pin HF offline so runs never hit the network
    import drumjot_training  # noqa: F401

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_missing_model_offline_raises_clear_fetch_error():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    # offline + not cached -> a clear "run fetch_models" error, not a raw HF OSError
    with pytest.raises(RuntimeError, match="fetch_models"):
        embeddings.MertEncoder(name="drumjot-nonexistent/not-a-real-model")


class _StubEncoder:
    name, layer, sr = "enc", 10, 24000

    def encode(self, waveform, sr):
        return np.ones((5, 4), dtype=np.float32)


def _stub_hb(monkeypatch):
    # the high-band block loads real 44.1k audio; stub it for fake-path tests
    monkeypatch.setattr(
        embeddings, "highband_features",
        lambda p, n, max_seconds=None, start_seconds=0.0, fps=embeddings.MERT_FPS:
            np.zeros((n, embeddings.HB_BANDS), dtype=np.float32),
    )


def test_embed_clip_caches_fp16_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "load_audio", lambda p, sr=None: np.zeros(100, dtype=np.float32))
    _stub_hb(monkeypatch)
    feat = embeddings.embed_clip("/x/a.flac", _StubEncoder(), cache_dir=tmp_path)
    assert feat.dtype == np.float16  # returned features are the cached precision
    assert feat.shape[1] == 4 + embeddings.HB_BANDS  # encoder dims + high-band block
    saved = np.load(next(tmp_path.glob("*.npy")))
    assert saved.dtype == np.float16  # and the on-disk cache is fp16


def test_embed_clip_cache_dtype_override_to_fp32(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "load_audio", lambda p, sr=None: np.zeros(100, dtype=np.float32))
    _stub_hb(monkeypatch)
    feat = embeddings.embed_clip("/x/b.flac", _StubEncoder(), cache_dir=tmp_path, cache_dtype="float32")
    assert feat.dtype == np.float32
    assert np.load(next(tmp_path.glob("*.npy"))).dtype == np.float32


def test_embed_clip_no_high_band_is_raw_mert(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "load_audio", lambda p, sr=None: np.zeros(100, dtype=np.float32))
    _stub_hb(monkeypatch)  # would add HB_BANDS cols if (wrongly) called
    feat = embeddings.embed_clip("/x/c.flac", _StubEncoder(), cache_dir=tmp_path, high_band=False)
    assert feat.shape[1] == 4  # encoder dims only, no high-band block appended


def test_embed_clip_high_band_on_off_caches_dont_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "load_audio", lambda p, sr=None: np.zeros(100, dtype=np.float32))
    _stub_hb(monkeypatch)
    on = embeddings.embed_clip("/x/d.flac", _StubEncoder(), cache_dir=tmp_path, high_band=True)
    off = embeddings.embed_clip("/x/d.flac", _StubEncoder(), cache_dir=tmp_path, high_band=False)
    assert on.shape[1] == 4 + embeddings.HB_BANDS and off.shape[1] == 4  # distinct widths
    assert len(list(tmp_path.glob("*.npy"))) == 2  # separate cache files (variant in key)


def test_cache_key_depends_on_variant(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert embeddings.cache_key(f, "enc", 10, variant="") != embeddings.cache_key(f, "enc", 10)


def test_feat_variant_and_dim_compose():
    assert embeddings.feat_variant(True, False) == "hb16" == embeddings.FEAT_VARIANT
    assert embeddings.feat_variant(False, False) == ""
    assert embeddings.feat_variant(False, True) == "cym5"
    assert embeddings.feat_variant(True, True) == "hb16+cym5"
    md = embeddings.MERT_DIM
    assert embeddings.feat_dim(True, False) == md + embeddings.HB_BANDS == embeddings.FEAT_DIM
    assert embeddings.feat_dim(False, False) == md
    assert embeddings.feat_dim(False, True) == md + embeddings.CYM_BANDS
    assert embeddings.feat_dim(True, True) == md + embeddings.HB_BANDS + embeddings.CYM_BANDS


def test_embed_clip_cym_widths_and_distinct_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "load_audio", lambda p, sr=None: np.zeros(100, dtype=np.float32))
    _stub_hb(monkeypatch)
    monkeypatch.setattr(
        embeddings, "cym_features",
        lambda p, n, max_seconds=None, start_seconds=0.0, fps=embeddings.MERT_FPS:
            np.zeros((n, embeddings.CYM_BANDS), dtype=np.float32),
    )
    widths = {}
    for hb, cy in [(True, False), (False, False), (False, True), (True, True)]:
        feat = embeddings.embed_clip("/x/z.flac", _StubEncoder(), cache_dir=tmp_path,
                                     high_band=hb, cym=cy)
        # _StubEncoder emits 4 enc dims (not MERT_DIM); width = enc + appended blocks
        expect = 4 + (embeddings.HB_BANDS if hb else 0) + (embeddings.CYM_BANDS if cy else 0)
        widths[(hb, cy)] = feat.shape[1]
        assert feat.shape[1] == expect, (hb, cy, feat.shape[1], expect)
    assert len(set(widths.values())) == 4  # all four recipes distinct widths
    assert len(list(tmp_path.glob("*.npy"))) == 4  # ...and four distinct cache files


def test_cache_key_is_stable(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert embeddings.cache_key(f, "enc", 10) == embeddings.cache_key(f, "enc", 10)


def test_cache_key_depends_on_layer_and_encoder(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    base = embeddings.cache_key(f, "enc", 10)
    assert embeddings.cache_key(f, "enc", 11) != base
    assert embeddings.cache_key(f, "other", 10) != base


def test_cache_key_depends_on_path(tmp_path):
    a = tmp_path / "a.wav"
    a.write_bytes(b"x")
    b = tmp_path / "b.wav"
    b.write_bytes(b"x")
    assert embeddings.cache_key(a, "enc", 10) != embeddings.cache_key(b, "enc", 10)


def test_cache_key_depends_on_window(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert embeddings.cache_key(f, "enc", 10, 30.0) != embeddings.cache_key(f, "enc", 10, None)
