"""Guard the ONNX provisioning wiring: every name a loader resolves via
`shipped_onnx` must be in the provisioned set (`_ONNX_FILES`), so the uploaded
files, the download list, and the loader lookups can't drift apart.
"""

from app.config import settings
from app.pipeline.lyrics_onnx import _sanitize
from app.pipeline.provision import _ONNX_FILES, provisioned_file, shipped_onnx

# The `shipped_onnx(name)` keys each loader looks up (name -> {name}.fp16.onnx).
_LOADER_NAMES = {
    "model_bs_roformer_sw",  # separation (stage 1)
    "drumsep_5stems_mdx23c_jarredou",  # separation (stage 2)
    "mert_L10",  # learned-onset encoder (ab3_prev is layer 10)
    "onset_heads",  # learned-onset heads
    "adtof_frame_rnn",  # ADTOF onsets
    "beat_this",  # beats
    f"ctc_align__{_sanitize('facebook/wav2vec2-large-robust-ft-libri-960h')}",  # lyrics EN
    f"ctc_align__{_sanitize('MahmoudAshraf/mms-300m-1130-forced-aligner')}",  # lyrics MMS
}


def test_every_loader_name_is_provisioned():
    for name in _LOADER_NAMES:
        assert f"{name}.fp16.onnx" in _ONNX_FILES, f"{name}.fp16.onnx missing from _ONNX_FILES"


def test_onset_meta_sidecar_is_provisioned():
    assert "onset_meta.json" in _ONNX_FILES


def test_no_unexpected_provisioned_onnx():
    # every provisioned .onnx must correspond to a loader lookup (no orphans)
    provisioned = {f[: -len(".fp16.onnx")] for f in _ONNX_FILES if f.endswith(".fp16.onnx")}
    assert provisioned == _LOADER_NAMES


def test_shipped_onnx_and_provisioned_file_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    assert shipped_onnx("beat_this") is None
    (tmp_path / "beat_this.fp16.onnx").write_bytes(b"onnx")
    assert shipped_onnx("beat_this") == tmp_path / "beat_this.fp16.onnx"
    assert provisioned_file("onset_meta.json") is None
    (tmp_path / "onset_meta.json").write_text("{}")
    assert provisioned_file("onset_meta.json") == tmp_path / "onset_meta.json"
    # empty file is treated as absent (an interrupted download never counts)
    (tmp_path / "empty.fp16.onnx").write_bytes(b"")
    assert shipped_onnx("empty") is None
