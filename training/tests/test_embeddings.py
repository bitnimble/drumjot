import numpy as np

import drumjot_training.embeddings as embeddings


class _StubEncoder:
    name, layer, sr = "enc", 10, 24000

    def encode(self, waveform, sr):
        return np.ones((5, 4), dtype=np.float32)


def test_embed_clip_caches_fp16_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "load_audio", lambda p, sr=None: np.zeros(100, dtype=np.float32))
    feat = embeddings.embed_clip("/x/a.flac", _StubEncoder(), cache_dir=tmp_path)
    assert feat.dtype == np.float16  # returned features are the cached precision
    saved = np.load(next(tmp_path.glob("*.npy")))
    assert saved.dtype == np.float16  # and the on-disk cache is fp16


def test_embed_clip_cache_dtype_override_to_fp32(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "load_audio", lambda p, sr=None: np.zeros(100, dtype=np.float32))
    feat = embeddings.embed_clip("/x/b.flac", _StubEncoder(), cache_dir=tmp_path, cache_dtype="float32")
    assert feat.dtype == np.float32
    assert np.load(next(tmp_path.glob("*.npy"))).dtype == np.float32


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
