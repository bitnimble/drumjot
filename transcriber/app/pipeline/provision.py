"""Capability-scoped provisioning of model assets.

Every model the packaged app runs on is downloaded into `settings.models_dir`
here, and downloads are **capability-scoped**: `provision("separation")` fetches
only the separation assets, `provision("lyrics")` only what /lyrics needs, and so
on. This is the whole point of the dependency-group split -- a user who installs
one capability must never pull another capability's weights (the lyrics models
alone are >1 GB). The capability -> asset map mirrors the pyproject
dependency-groups, where `transcription` and `lyrics` both compose `separation`.

Shipped runtime assets = the fp16 `.onnx` bodies plus the small sidecars they
need: the separation architecture yamls (STFT params the numpy path reads) and
the onset `meta.json` (lane vocab / thresholds / fps). All come from the one
`settings.onnx_repo`. The heavy torch `.ckpt`s are NOT fetched -- the
shipped runtime is torch-free and loads the onnx; a dev checkout exports locally
from ckpts already in its `models_dir`.

Every URL / HF id is a `settings.*` field (see config.py "Model asset sources"),
so a build can repoint them without code changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import settings

log = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(30.0, read=None)  # read=None: large weights

# ckpt -> paired architecture yaml (a bare state_dict can't load without it).
_CKPT_YAML: dict[str, str] = {
    "model_bs_roformer_sw.ckpt": "config_bs_roformer_sw.yaml",
    "drumsep_5stems_mdx23c_jarredou.ckpt": "config_drumsep_5stems_mdx23c.yaml",
}


def yaml_for_ckpt(ckpt_filename: str) -> str:
    """Local yaml filename paired with a separation ckpt filename."""
    try:
        return _CKPT_YAML[ckpt_filename]
    except KeyError:
        raise KeyError(f"no yaml registered for ckpt {ckpt_filename!r}") from None


@dataclass(frozen=True)
class _Asset:
    """One downloadable file: local name under models_dir + its source URL."""

    filename: str
    url: str


def _onnx(name: str) -> _Asset:
    return _Asset(name, f"{settings.onnx_repo}/{name}")


def _separation_assets() -> list[_Asset]:
    """Both separation model bodies (fp16 onnx) + their yamls. No ckpts (the
    shipped runtime uses the onnx). Names derive from `settings.*_model`."""
    out: list[_Asset] = []
    for ckpt in (settings.demucs_model, settings.drum_pieces_model):
        out.append(_onnx(yaml_for_ckpt(ckpt)))
        out.append(_onnx(f"{Path(ckpt).stem}.fp16.onnx"))
    return out


def _onset_assets() -> list[_Asset]:
    # MERT encoder (layer 10 = the shipped ab3_prev checkpoint) + per-lane heads
    # + the heads' meta.json sidecar (lanes / thresholds / fps).
    return [_onnx("mert_L10.fp16.onnx"), _onnx("onset_heads.fp16.onnx"), _onnx("onset_meta.json")]


def _lyrics_assets() -> list[_Asset]:
    return [
        _onnx(f"ctc_align__{m.replace('/', '__')}.fp16.onnx")
        for m in (settings.lyrics_align_model_english, settings.lyrics_align_model_default)
    ]


def _capability_assets(capability: str) -> list[_Asset]:
    """Assets one capability needs. `transcription` and `lyrics` both compose
    `separation`, mirroring the pyproject dependency-groups."""
    if capability == "separation":
        return _separation_assets()
    if capability == "transcription":
        return (
            _separation_assets()
            + _onset_assets()
            + [_onnx("beat_this.fp16.onnx"), _onnx("adtof_frame_rnn.fp16.onnx")]
        )
    if capability == "lyrics":
        return _separation_assets() + _lyrics_assets()
    return []


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, atomically. No-op if `dest` already exists.

    HF `resolve/` URLs 302 to a CDN, so redirects are followed. Writes to a
    `.part` sidecar and renames on success so an interrupted download is never
    mistaken for a completed one on the next startup."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("provision: %s already present, skipping download", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("provision: downloading %s -> %s", url, dest.name)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=_HTTP_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log.info("provision: fetched %s (%d bytes)", dest.name, dest.stat().st_size)


def provision(*capabilities: str) -> None:
    """Download only the assets the given capabilities need (deduped by filename)
    into `settings.models_dir`. Idempotent.

    This is the capability-scoped entry point: NEVER fetch every model regardless
    of capability -- that defeats the dependency-group split (a separation-only
    install would pull the >1 GB lyrics models). Add a model to `_capability_assets`
    under the one capability that uses it."""
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, _Asset] = {}
    for capability in capabilities:
        for asset in _capability_assets(capability):
            assets[asset.filename] = asset
    for asset in assets.values():
        _download(asset.url, models_dir / asset.filename)
    log.info("provision: %d assets ready in %s for %s", len(assets), models_dir, list(capabilities))


def provision_custom_models() -> None:
    """Provision the separation capability's assets (yaml + fp16 onnx). Called
    eagerly by `separate.py` so the separation stage's model is present."""
    provision("separation")


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
    """`python -m app.pipeline.provision <capability>...` -- pre-fetch the assets a
    freshly-installed capability needs, so they download at install time rather
    than on first use. The desktop installer runs this after `uv sync` with the
    capabilities it installed. Best-effort; lazy fallbacks still cover a failure."""
    logging.basicConfig(level=logging.INFO)
    provision(*[g for g in argv if g in ("separation", "transcription", "lyrics")])
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
