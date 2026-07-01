"""Guard the capability-scoped provisioning: each capability downloads only its
own assets (never everything), and every name a loader resolves via `shipped_onnx`
is provisioned by some capability -- so the upload / download / lookup filenames
can't drift, and the dependency-group split isn't silently defeated.
"""

from app.config import settings
from app.pipeline.provision import _capability_assets, provisioned_file, shipped_onnx


def _names(capability):
    return {a.filename for a in _capability_assets(capability)}


def test_separation_is_scoped_to_separation():
    names = _names("separation")
    assert {"model_bs_roformer_sw.fp16.onnx", "drumsep_5stems_mdx23c_jarredou.fp16.onnx",
            "config_bs_roformer_sw.yaml"} <= names
    # never pulls onset / beat / adtof / lyrics weights
    assert not any(k in f for f in names for k in ("mert", "onset", "beat", "adtof", "ctc_align"))


def test_lyrics_composes_separation_but_not_transcription():
    names = _names("lyrics")
    assert "model_bs_roformer_sw.fp16.onnx" in names  # /lyrics needs the vocals stem
    assert any(f.startswith("ctc_align__") for f in names)
    # the big onset/beat models are NOT pulled for a lyrics-only install
    assert not any(k in f for f in names for k in ("mert", "beat_this", "adtof"))


def test_transcription_composes_separation_onsets_beats_not_lyrics():
    names = _names("transcription")
    assert {"mert_L10.fp16.onnx", "onset_heads.fp16.onnx", "onset_meta.json",
            "beat_this.fp16.onnx", "adtof_frame_rnn.fp16.onnx",
            "model_bs_roformer_sw.fp16.onnx"} <= names
    assert not any(f.startswith("ctc_align__") for f in names)


def test_every_loader_lookup_is_provisioned_by_some_capability():
    provisioned = set()
    for cap in ("separation", "transcription", "lyrics"):
        provisioned |= _names(cap)
    loader_names = {
        "model_bs_roformer_sw", "drumsep_5stems_mdx23c_jarredou",  # separation
        "mert_L10", "onset_heads",  # learned onsets
        "adtof_frame_rnn", "beat_this",  # adtof, beats
        f"ctc_align__{settings.lyrics_align_model_english.replace('/', '__')}",
        f"ctc_align__{settings.lyrics_align_model_default.replace('/', '__')}",  # lyrics
    }
    for name in loader_names:
        assert f"{name}.fp16.onnx" in provisioned, f"{name}.fp16.onnx not provisioned"


def test_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    assert shipped_onnx("beat_this") is None
    (tmp_path / "beat_this.fp16.onnx").write_bytes(b"onnx")
    assert shipped_onnx("beat_this") == tmp_path / "beat_this.fp16.onnx"
    (tmp_path / "empty.fp16.onnx").write_bytes(b"")  # interrupted download != present
    assert shipped_onnx("empty") is None
    assert provisioned_file("onset_meta.json") is None
    (tmp_path / "onset_meta.json").write_text("{}")
    assert provisioned_file("onset_meta.json") == tmp_path / "onset_meta.json"
