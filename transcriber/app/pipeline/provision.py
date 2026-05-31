"""On-startup provisioning of custom separation models for audio-separator.

audio-separator resolves a `model_filename` against a registry it builds
by merging `<model_file_dir>/download_checks.json` (fetched from TRvlvr's
GitHub) with its bundled, in-package `models.json`. Neither lists the two
models this project now uses:

  - Stage 1 (`stems_all`): **BS-Roformer SW** (`jarredou/BS-ROFO-SW-Fixed`)
    — a 6-stem (vocals / drums / bass / guitar / piano / other) Band-Split
    RoPE Transformer. Drums SDR ~14 vs htdemucs_ft's ~10, with markedly
    better high-frequency (cymbal / hi-hat) preservation in the drum stem,
    which is what Stage 2 then has to split.
  - Stage 2 (`stems_per`): **jarredou 5-stem MDX23C DrumSep**
    (kick / snare / toms / hh / cymbals). Note ride + crash are merged
    into one `cymbals` stem — see `separate.STEM_NAME_TO_PITCH`.

`audio-separator`'s only supported model sources are its package registry
and a narrow set of upstream repos (its yaml fallback knows just two), so
neither model is loadable out of the box and there is no public API to
register a local checkpoint. We make them loadable WITHOUT patching the
installed package by exploiting two facts about audio-separator's loader:

  1. It builds its registry from `<models_dir>/download_checks.json`, a
     plain file on the (persistent) models volume — writable by us.
  2. `download_file_if_not_exists()` is a no-op when the target file is
     already present in `models_dir`.

So this module, on every startup and before `Separator.load()` builds the
audio-separator models:

  1. Ensures `<models_dir>/download_checks.json` exists (downloading the
     real TRvlvr list if absent so the rest of the registry is preserved;
     falling back to an empty registry only if that download fails).
  2. Idempotently injects one entry per model into the relevant
     `*_download_list`, mapping a **local yaml filename -> local ckpt
     filename** (plain names, never URLs).
  3. Downloads each model's ckpt + yaml from its upstream release into
     `models_dir` under exactly those local filenames.

Because the files are already on disk under the injected names,
audio-separator's own download step is skipped and it loads straight from
local — sidestepping its narrow yaml-URL fallback entirely.

Idempotent and safe to call on every startup. Fails loud: a model we no
longer ship a fallback for must surface as a startup error, not silently
degrade transcription quality.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import settings

log = logging.getLogger(__name__)

# The exact list audio-separator itself fetches to build its registry.
# We fetch it to the same path so audio-separator's own
# download-if-missing is a no-op and the full upstream registry is
# preserved alongside our injected entries.
_DOWNLOAD_CHECKS_URL = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/"
    "main/filelists/download_checks.json"
)

_HTTP_TIMEOUT = httpx.Timeout(30.0, read=None)  # read=None: large weights


@dataclass(frozen=True)
class _CustomModel:
    """A model audio-separator doesn't know about that we inject + fetch.

    `list_key` is the `download_checks.json` sub-list the entry joins;
    audio-separator groups entries by which list they came from and
    dispatches the architecture from that grouping, so this MUST match
    the model's true architecture (`mdx23c_download_list` -> MDXC,
    `roformer_download_list` -> Roformer).

    `yaml_local` deliberately contains the substring "roformer" for the
    Roformer model: audio-separator sets its `is_roformer` flag from
    `"roformer" in <yaml path>.lower()`, and the upstream filename
    (`BS-Rofo-SW-Fixed.yaml`) does NOT contain it.
    """

    list_key: str
    friendly_name: str
    ckpt_local: str  # filename under models_dir == settings.*_model value
    yaml_local: str  # filename under models_dir
    ckpt_url: str
    yaml_url: str


# The two filenames here are the source of truth for
# `settings.demucs_model` / `settings.drum_pieces_model` — keep them in
# sync if either is overridden via env.
_MODELS: list[_CustomModel] = [
    _CustomModel(
        list_key="roformer_download_list",
        friendly_name="Roformer Model: BS-Roformer SW (jarredou BS-ROFO-SW-Fixed)",
        ckpt_local="model_bs_roformer_sw.ckpt",
        yaml_local="config_bs_roformer_sw.yaml",
        ckpt_url=(
            "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/"
            "resolve/main/BS-Rofo-SW-Fixed.ckpt"
        ),
        yaml_url=(
            "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/"
            "resolve/main/BS-Rofo-SW-Fixed.yaml"
        ),
    ),
    _CustomModel(
        list_key="mdx23c_download_list",
        friendly_name="MDX23C Model: DrumSep 5-stem (jarredou)",
        ckpt_local="drumsep_5stems_mdx23c_jarredou.ckpt",
        yaml_local="config_drumsep_5stems_mdx23c.yaml",
        ckpt_url=(
            "https://github.com/jarredou/models/releases/download/"
            "DrumSep/drumsep_5stems_mdx23c_jarredou.ckpt"
        ),
        yaml_url=(
            "https://github.com/jarredou/models/releases/download/"
            "DrumSep/config_mdx23c.yaml"
        ),
    ),
]


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


def _ensure_registry(models_dir: Path) -> None:
    """Inject our entries into `<models_dir>/download_checks.json`.

    Preserves the full upstream registry when present (fetches it if the
    file is absent); falls back to an empty registry only if that fetch
    fails, since our two injected entries are all the pipeline actually
    needs. Idempotent: re-writes only when an entry is missing or stale.
    """
    checks_path = models_dir / "download_checks.json"
    if checks_path.exists():
        try:
            data = json.loads(checks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{checks_path} is present but not valid JSON ({exc}); "
                "refusing to clobber a possibly-real registry. Delete it "
                "to force a clean re-fetch."
            ) from exc
    else:
        try:
            _download(_DOWNLOAD_CHECKS_URL, checks_path)
            data = json.loads(checks_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning(
                "provision: could not fetch upstream download_checks.json "
                "(%s); starting from an empty registry with only the "
                "injected models.",
                exc,
            )
            data = {}

    changed = False
    for m in _MODELS:
        bucket = data.setdefault(m.list_key, {})
        desired = {m.yaml_local: m.ckpt_local}
        if bucket.get(m.friendly_name) != desired:
            bucket[m.friendly_name] = desired
            changed = True
            log.info(
                "provision: injected %r into %s", m.friendly_name, m.list_key
            )

    if changed or not checks_path.exists():
        tmp = checks_path.with_name(checks_path.name + ".part")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(checks_path)


# Default vocals separator URL: audio-separator can fetch it on demand
# from its bundled registry, but we pre-stage the file so first
# /lyrics/align "mix" call doesn't pay the download. Only used when
# `settings.vocals_model` matches the default filename; if an operator
# overrides it via env, the lazy audio-separator path handles the new
# choice. Stable TRvlvr release URL.
_VOCALS_FP32_FILENAME = "UVR-MDX-NET-Voc_FT.onnx"
_VOCALS_DEFAULT_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
)


def _provision_lyrics_assets(models_dir: Path) -> None:
    """Pre-fetch the vocals separator the /lyrics/align endpoint pulls
    lazily.

    The download lands in the bind-mounted `/models` volume, so a fresh
    deployment pays the cost once at container start (matching the
    drum-pipeline weights above) rather than on the first user-facing
    /lyrics/align call. Fails soft: a network blip or upstream rename
    shouldn't break the drum pipeline, since audio-separator will retry
    the lazy download itself.

    Only the default vocals filename is pre-fetched; arbitrary overrides
    defer to audio-separator's own download path on first /lyrics/align.

    Alignment model weights (English wav2vec2-large-robust + multilingual
    MMS-300m) are loaded through HuggingFace transformers on first
    /lyrics/align call; they're not pre-staged here because HF's local
    cache is in a different directory and the lazy load is the only
    path through `ctc-forced-aligner.load_alignment_model`.
    """
    configured = settings.vocals_model
    if configured == _VOCALS_FP32_FILENAME:
        try:
            _download(_VOCALS_DEFAULT_URL, models_dir / _VOCALS_FP32_FILENAME)
        except Exception as exc:
            log.warning(
                "provision: vocals model pre-fetch failed (%s); "
                "audio-separator will retry on first /lyrics/align call.",
                exc,
            )
    else:
        log.info(
            "provision: vocals_model=%s differs from default; skipping "
            "pre-fetch (audio-separator will resolve on first use).",
            configured,
        )


def provision_custom_models() -> None:
    """Make the BS-Roformer SW + jarredou DrumSep models loadable, and
    pre-stage the /lyrics/align assets so the first lyrics request
    doesn't pay multi-GB download latency.

    Call once at startup, BEFORE constructing audio-separator's
    `Separator` / calling `load_model`, so the registry and local files
    are in place by the time audio-separator reads them.
    """
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    _ensure_registry(models_dir)

    for m in _MODELS:
        _download(m.ckpt_url, models_dir / m.ckpt_local)
        _download(m.yaml_url, models_dir / m.yaml_local)

    _provision_lyrics_assets(models_dir)

    log.info("provision: custom separation models ready in %s", models_dir)
