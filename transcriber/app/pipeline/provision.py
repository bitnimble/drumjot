"""On-startup provisioning of the separation model weights.

The vendored separation wrapper (`pipeline/separation/`) loads two models by
local filename from `settings.models_dir`:

  - Stage 1 (`stems_all`): BS-Roformer SW (`model_bs_roformer_sw.ckpt`), a
    6-stem (vocals / drums / bass / guitar / piano / other) Band-Split RoPE
    Transformer. We keep its `drums` stem for Stage 2 and its `vocals` stem
    for /lyrics alignment.
  - Stage 2 (`stems_per`): jarredou 5-stem MDX23C DrumSep
    (`drumsep_5stems_mdx23c_jarredou.ckpt`): kick / snare / toms / hh /
    cymbals (ride + crash merged into `cymbals`; see
    `separate.STEM_NAME_TO_PITCH`).

Each model is a ckpt plus a paired architecture yaml, downloaded from our own
HF mirror (`bitnimble/stem_separation`; jarredou's upstream was deleted). The
alignment wav2vec2 weights load lazily via HuggingFace on first /lyrics call,
so they are not staged here. Idempotent and safe to call on every startup; an
interrupted download is never mistaken for a completed one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import settings

log = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(30.0, read=None)  # read=None: large weights


@dataclass(frozen=True)
class _CustomModel:
    """A separation model we fetch into models_dir (a ckpt + its paired yaml)."""

    ckpt_local: str  # filename under models_dir == settings.*_model value
    yaml_local: str  # filename under models_dir
    ckpt_url: str
    yaml_url: str


# The two filenames here are the source of truth for
# `settings.demucs_model` / `settings.drum_pieces_model` — keep them in
# sync if either is overridden via env.
_MODELS: list[_CustomModel] = [
    _CustomModel(
        ckpt_local="model_bs_roformer_sw.ckpt",
        yaml_local="config_bs_roformer_sw.yaml",
        # Vendored to our own HF account: jarredou's GitHub was deleted and
        # the upstream HF repo can't be relied on. These are byte copies of
        # the original ckpt/yaml (renamed to our local filenames).
        ckpt_url=(
            "https://huggingface.co/bitnimble/stem_separation/"
            "resolve/main/model_bs_roformer_sw.ckpt"
        ),
        yaml_url=(
            "https://huggingface.co/bitnimble/stem_separation/"
            "resolve/main/config_bs_roformer_sw.yaml"
        ),
    ),
    _CustomModel(
        ckpt_local="drumsep_5stems_mdx23c_jarredou.ckpt",
        yaml_local="config_drumsep_5stems_mdx23c.yaml",
        ckpt_url=(
            "https://huggingface.co/bitnimble/stem_separation/"
            "resolve/main/drumsep_5stems_mdx23c_jarredou.ckpt"
        ),
        yaml_url=(
            "https://huggingface.co/bitnimble/stem_separation/"
            "resolve/main/config_drumsep_5stems_mdx23c.yaml"
        ),
    ),
]


def yaml_for_ckpt(ckpt_filename: str) -> str:
    """Local yaml filename paired with a custom model's ckpt filename.

    The two filenames are independent (a bare state_dict can't be loaded
    without its architecture yaml), so the pairing lives here, next to the
    download definitions. Used by `separate.py` to build the (ckpt, yaml)
    pair the vendored loader needs."""
    for m in _MODELS:
        if m.ckpt_local == ckpt_filename:
            return m.yaml_local
    raise KeyError(f"no custom-model yaml registered for ckpt {ckpt_filename!r}")


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, atomically. No-op if `dest` already exists.

    GitHub release assets and Hugging Face `resolve/` URLs both 302 to a
    CDN, so redirects are followed. Writes to a `.part` sidecar and
    renames on success so an interrupted download is never mistaken for a
    completed one on the next startup (which would skip the re-fetch).
    """
    if dest.exists() and dest.stat().st_size > 0:
        log.info("provision: %s already present, skipping download", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("provision: downloading %s -> %s", url, dest.name)
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=_HTTP_TIMEOUT
        ) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log.info("provision: fetched %s (%d bytes)", dest.name, dest.stat().st_size)


def provision_custom_models() -> None:
    """Download the BS-Roformer SW + jarredou DrumSep weights into
    `settings.models_dir` so the vendored separation wrapper can load them.

    Call once at startup. The /lyrics vocals stem now comes from the same
    BS-Roformer SW model (no separate vocals model to fetch); the alignment
    wav2vec2 weights load lazily via HuggingFace on first /lyrics/align.
    """
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for m in _MODELS:
        _download(m.ckpt_url, models_dir / m.ckpt_local)
        _download(m.yaml_url, models_dir / m.yaml_local)

    log.info("provision: separation models ready in %s", models_dir)


# ---- shipped ONNX models (fp16) ------------------------------------------
#
# Every ML model the transcriber runs on onnxruntime is pre-exported to fp16 and
# hosted at bitnimble/drumjot-onnx, so the packaged (torch-free) app downloads
# ready `.onnx` instead of exporting at first run. Each loader prefers the
# shipped file (`shipped_onnx`) and only falls back to a local torch export in a
# dev checkout where the file isn't present. fp16 is GPU-only (validated on CUDA
# at corr >= 0.99998 vs fp32); a CPU-EP deployment would need the fp32 set.
_ONNX_BASE = "https://huggingface.co/bitnimble/drumjot-onnx/resolve/main"
_ONNX_FILES: tuple[str, ...] = (
    "model_bs_roformer_sw.fp16.onnx",
    "drumsep_5stems_mdx23c_jarredou.fp16.onnx",
    "mert_L10.fp16.onnx",
    "onset_heads.fp16.onnx",
    # The learned-onset heads are checkpoint-specific; meta.json carries their
    # lane vocab, encoder/layer, fps, and tuned per-lane thresholds. Ships with
    # the heads so the onset path needs no local training checkpoint.
    "onset_meta.json",
    "adtof_frame_rnn.fp16.onnx",
    "beat_this.fp16.onnx",
    "ctc_align__facebook__wav2vec2-large-robust-ft-libri-960h.fp16.onnx",
    "ctc_align__MahmoudAshraf__mms-300m-1130-forced-aligner.fp16.onnx",
)


def provision_onnx_models() -> None:
    """Download the shipped fp16 `.onnx` set into `settings.models_dir`.
    Idempotent (skips present files); call at install time."""
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for fname in _ONNX_FILES:
        _download(f"{_ONNX_BASE}/{fname}", models_dir / fname)
    log.info("provision: onnx models ready in %s", models_dir)


def provisioned_file(filename: str) -> Path | None:
    """Path to a provisioned asset `filename` under `settings.models_dir` if
    present + non-empty, else None."""
    path = Path(settings.models_dir) / filename
    return path if path.exists() and path.stat().st_size > 0 else None


def shipped_onnx(name: str) -> Path | None:
    """Path to the shipped fp16 onnx `{name}.fp16.onnx` if present, else None.
    Loaders use this to prefer the downloaded weights and skip the local
    (torch-dependent) export."""
    return provisioned_file(f"{name}.fp16.onnx")


def main(argv: list[str]) -> int:
    """`python -m app.pipeline.provision <uv-group>...` - pre-fetch the heavy model
    assets a freshly-installed capability needs (separation models + vocals, and
    the Beat This! weights for transcribe), so they download at install time
    rather than on first use. Best-effort; the lazy fallbacks still cover a
    failure here. The desktop installer runs this after `uv sync`."""
    logging.basicConfig(level=logging.INFO)
    groups = set(argv)
    if groups & {"separation", "transcription", "lyrics"}:
        # Separation ckpt+yaml (the yaml carries the STFT params the numpy path
        # reads; the ckpt is only needed for a dev/torch local export), plus the
        # shipped fp16 .onnx set every model runs on.
        provision_custom_models()
        provision_onnx_models()
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
