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

import hashlib
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
# Derived fp16 sibling produced at container startup by graph-level
# conversion of the fp32 source above. The runtime default
# (`settings.vocals_model`) points here so onnxruntime loads the
# already-converted model directly; we keep the fp32 file around
# alongside as a fallback / A-B reference, not as a runtime path. See
# `_ensure_vocals_fp16` for the conversion details.
_VOCALS_FP16_FILENAME = "UVR-MDX-NET-Voc_FT_fp16.onnx"

# audio-separator looks up MDX-Net architecture parameters by the MD5 of
# the model file's trailing ~10MB (see `Separator.get_model_hash`) in
# `<models_dir>/mdx_model_data.json`. Our fp16 derivative has a different
# MD5 than the fp32 source, so even after we register it in
# `download_checks.json` it still needs its own entry here. We fetch
# upstream's copy ourselves so the file is present before
# `_register_vocals_fp16` reads it, mirroring how `_ensure_registry`
# eagerly stages `download_checks.json`.
_MDX_MODEL_DATA_URL = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/"
    "main/mdx_model_data/model_data_new.json"
)

# whisperx's English aligner is torchaudio's WAV2VEC2_ASR_BASE_960H
# bundle, which torch.hub fetches into `$TORCH_HOME/hub/checkpoints/`.
# Pre-stage it at the exact filename torch.hub computes from the URL so
# the runtime `load_state_dict_from_url` is a cache hit. Other-language
# aligners are still resolved lazily through HuggingFace.
_WAV2VEC2_EN_URL = (
    "https://download.pytorch.org/torchaudio/models/"
    "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
)
_WAV2VEC2_EN_FILENAME = "wav2vec2_fairseq_base_ls960_asr_ls960.pth"


def _ensure_vocals_fp16(fp32_path: Path, fp16_path: Path) -> None:
    """Derive an fp16-internal ONNX sibling of the fp32 vocals model.

    Idempotent: skips when `fp16_path` already exists at non-zero size,
    so re-running provision on warm containers is free. Fails-soft: on
    any conversion error, warns and leaves the operator to fall back to
    the fp32 model by setting `VOCALS_MODEL=UVR-MDX-NET-Voc_FT.onnx`
    (audio-separator rejects unknown filenames outright, so the fp16
    path needs `_register_vocals_fp16` to succeed; see that function
    for the dual registry injection required to make the derived file
    loadable).

    `keep_io_types=True` retains fp32 graph inputs and outputs so
    upstream (audio-separator's STFT path) and downstream (ISTFT,
    spec_utils) code can keep handing us fp32 tensors without an
    explicit cast at the call boundary. Internal weights and most
    activations move to fp16, which is where the throughput win on
    TensorCore-equipped GPUs actually comes from. Risk surface for
    MDX-Net specifically: STFT magnitude / phase reconstruction can be
    sensitive to fp16 underflow on quiet sections, producing muffled or
    NaN-tainted vocals; if a song reproducibly comes out garbled, set
    `VOCALS_MODEL=UVR-MDX-NET-Voc_FT.onnx` to switch back to fp32.
    """
    if fp16_path.exists() and fp16_path.stat().st_size > 0:
        log.info(
            "provision: %s already present, skipping fp16 conversion",
            fp16_path.name,
        )
        return
    try:
        # Lazy imports: onnxconverter_common pulls onnx graph machinery
        # we don't need on workers that never hit /lyrics/align.
        import onnx  # type: ignore[import-not-found]
        from onnxconverter_common.float16 import (  # type: ignore[import-not-found]
            convert_float_to_float16,
        )
    except Exception as exc:
        log.warning(
            "provision: onnxconverter_common unavailable (%s); leaving "
            "fp32 vocals model in place. Set VOCALS_MODEL to %s to use it.",
            exc, _VOCALS_FP32_FILENAME,
        )
        return

    # Block the fp16-fragile ops in MDX-Net's STFT-magnitude path. With a
    # bare `convert_float_to_float16(...)` the network produces all-zero
    # vocal stems on at least some songs: ReduceL2 / Sqrt / Div on quiet
    # spectral magnitudes underflow to fp16 zero, then InstanceNorm /
    # GroupNorm of zero variance cascades zeros forward, and iSTFT of a
    # zero spectrum is silence. Keeping these ops in fp32 (and inserting
    # Casts at their boundaries) costs some of the throughput win
    # vs pure fp16, but salvages numerical sanity on quiet vocals.
    fp16_op_blocklist = [
        "InstanceNormalization",
        "LayerNormalization",
        "GroupNormalization",
        "Sqrt",
        "Div",
        "ReduceMean",
        "ReduceL2",
    ]
    log.info(
        "provision: converting %s -> %s (graph-level fp16, "
        "keep_io_types=True, op_block_list=%s)",
        fp32_path.name, fp16_path.name, fp16_op_blocklist,
    )
    tmp = fp16_path.with_name(fp16_path.name + ".part")
    try:
        model = onnx.load(str(fp32_path))
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=fp16_op_blocklist,
        )
        onnx.save(model_fp16, str(tmp))
        tmp.replace(fp16_path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        log.warning(
            "provision: fp16 conversion failed (%s); leaving fp32 vocals "
            "model in place. Set VOCALS_MODEL to %s to use it.",
            exc, _VOCALS_FP32_FILENAME,
        )
        return
    log.info(
        "provision: wrote %s (%d bytes)",
        fp16_path.name, fp16_path.stat().st_size,
    )


def _audio_separator_hash(path: Path) -> str:
    """Compute the MD5 hash audio-separator uses to key `mdx_model_data.json`.

    Mirrors `audio_separator.separator.Separator.get_model_hash`: MD5 of
    the trailing 10,240,000 bytes (or the whole file when smaller). Must
    match exactly, otherwise our injected `mdx_model_data.json` entry
    won't be found when audio-separator hashes the same file at load time.
    """
    bytes_to_hash = 10_000 * 1024
    size = path.stat().st_size
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        if size > bytes_to_hash:
            fh.seek(size - bytes_to_hash)
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def _register_vocals_fp16(
    models_dir: Path, fp32_path: Path, fp16_path: Path
) -> None:
    """Make audio-separator recognize the derived fp16 vocals model.

    audio-separator's `load_model` gates on two on-disk registries:

      1. `download_checks.json::mdx_download_list` -- a
         `friendly_name -> filename` map; `download_model_files` rejects
         any `model_filename` not present here with "not found in
         supported model files".
      2. `mdx_model_data.json` -- a `md5_hash -> arch_params` map keyed
         by the MD5 of the model file's trailing ~10MB
         (`_audio_separator_hash`). Missing keys produce
         "Unsupported Model File: parameters for MD5 hash ...".

    Upstream knows only about the fp32 file. The fp16 derivative this
    module creates at startup is unknown on both axes (different
    filename, different MD5), so without injection
    `load_model(model_filename=<fp16>)` blows up. The fp16 graph
    conversion preserves topology (only weights move to fp16; IO stays
    fp32 via `keep_io_types=True`), so cloning the fp32 arch entry under
    the fp16 hash is a faithful pointer to the same MDX-Net params.

    Idempotent: rerunning with both entries already present is a no-op.
    Fails soft: missing upstream entries (registry shape drift, network
    failure on the lazy `mdx_model_data.json` fetch) log and skip
    registration, leaving the operator to switch
    `VOCALS_MODEL=UVR-MDX-NET-Voc_FT.onnx`.
    """
    if not fp16_path.exists() or fp16_path.stat().st_size == 0:
        return

    checks_path = models_dir / "download_checks.json"
    mdx_data_path = models_dir / "mdx_model_data.json"

    try:
        _download(_MDX_MODEL_DATA_URL, mdx_data_path)
    except Exception as exc:
        log.warning(
            "provision: could not fetch mdx_model_data.json (%s); "
            "skipping fp16 vocals registration.",
            exc,
        )
        return

    try:
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        mdx_data = json.loads(mdx_data_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        log.warning(
            "provision: cannot read registry files (%s); skipping fp16 "
            "vocals registration.",
            exc,
        )
        return

    mdx_list = checks.get("mdx_download_list", {})
    fp32_friendly = next(
        (k for k, v in mdx_list.items() if v == _VOCALS_FP32_FILENAME),
        None,
    )
    if fp32_friendly is None:
        log.warning(
            "provision: %s not in upstream mdx_download_list; skipping "
            "fp16 vocals registration.",
            _VOCALS_FP32_FILENAME,
        )
        return

    fp32_hash = _audio_separator_hash(fp32_path)
    fp32_entry = mdx_data.get(fp32_hash)
    if fp32_entry is None:
        log.warning(
            "provision: fp32 vocals MD5 %s missing from mdx_model_data.json; "
            "skipping fp16 vocals registration.",
            fp32_hash,
        )
        return

    fp16_hash = _audio_separator_hash(fp16_path)
    fp16_friendly = f"{fp32_friendly} (fp16, drumjot-derived)"

    if checks.setdefault("mdx_download_list", {}).get(fp16_friendly) != _VOCALS_FP16_FILENAME:
        checks["mdx_download_list"][fp16_friendly] = _VOCALS_FP16_FILENAME
        tmp = checks_path.with_name(checks_path.name + ".part")
        tmp.write_text(json.dumps(checks, indent=1), encoding="utf-8")
        tmp.replace(checks_path)
        log.info(
            "provision: injected %r -> %s into mdx_download_list",
            fp16_friendly, _VOCALS_FP16_FILENAME,
        )

    if mdx_data.get(fp16_hash) != fp32_entry:
        mdx_data[fp16_hash] = fp32_entry
        tmp = mdx_data_path.with_name(mdx_data_path.name + ".part")
        tmp.write_text(json.dumps(mdx_data, indent=1), encoding="utf-8")
        tmp.replace(mdx_data_path)
        log.info(
            "provision: injected fp16 vocals arch params under MD5 %s",
            fp16_hash,
        )


def _provision_lyrics_assets(models_dir: Path) -> None:
    """Pre-fetch the assets the /lyrics/align endpoint pulls lazily.

    All three downloads land in the bind-mounted `/models` volume, so a
    fresh deployment pays the cost once at container start (matching the
    drum-pipeline weights above) rather than on the first user-facing
    /lyrics/align call. Each fetch fails soft: a network blip or upstream
    rename of one asset shouldn't break the drum pipeline, since the
    runtime loaders will retry the lazy download themselves.

      1. Vocals separator (`settings.vocals_model`): direct HTTP fetch
         from TRvlvr's mirror. Only attempted when the configured
         filename matches the default; overrides defer to
         audio-separator's own download path on first /lyrics/align.
      2. wav2vec2 English aligner: direct HTTP fetch into
         `<models_dir>/torch/hub/checkpoints/`, where torchaudio's
         `WAV2VEC2_ASR_BASE_960H.get_model()` looks first.
      3. faster-whisper transcribe model (`settings.whisper_model`):
         delegated to `faster_whisper.download_model` so the model-name
         -> HF repo mapping and the HuggingFace cache layout match
         exactly what whisperx expects at load time.
    """
    # Pre-stage the fp32 source whenever the configured vocals model is
    # one of OUR variants of UVR-MDX-NET-Voc_FT (either the fp32 file
    # itself or our derived fp16 sibling); arbitrary overrides defer to
    # audio-separator's lazy download path on first use. The fp16
    # derivation is skipped (not just the conversion - we can't do it
    # without the source) when the fp32 download fails; the other
    # lyrics assets below still get pre-fetched on their own try blocks.
    configured = settings.vocals_model
    if configured in {_VOCALS_FP32_FILENAME, _VOCALS_FP16_FILENAME}:
        fp32_path = models_dir / _VOCALS_FP32_FILENAME
        fp32_ok = False
        try:
            _download(_VOCALS_DEFAULT_URL, fp32_path)
            fp32_ok = True
        except Exception as exc:
            log.warning(
                "provision: vocals model pre-fetch failed (%s); "
                "audio-separator will retry on first /lyrics/align call.",
                exc,
            )
        if fp32_ok and configured == _VOCALS_FP16_FILENAME:
            fp16_path = models_dir / _VOCALS_FP16_FILENAME
            _ensure_vocals_fp16(fp32_path, fp16_path)
            _register_vocals_fp16(models_dir, fp32_path, fp16_path)
    else:
        log.info(
            "provision: vocals_model=%s differs from defaults; skipping "
            "pre-fetch (audio-separator will resolve on first use).",
            configured,
        )

    try:
        wav2vec2_dest = (
            models_dir / "torch" / "hub" / "checkpoints" / _WAV2VEC2_EN_FILENAME
        )
        _download(_WAV2VEC2_EN_URL, wav2vec2_dest)
    except Exception as exc:
        log.warning(
            "provision: wav2vec2 EN aligner pre-fetch failed (%s); "
            "whisperx will retry on first English alignment.",
            exc,
        )

    try:
        # Imported lazily because faster_whisper pulls in CTranslate2.
        # We're already in the pipeline worker (which loads heavy ML
        # deps moments later), so the cost is on the same axis.
        import faster_whisper  # type: ignore[import-not-found]

        whisper_cache = models_dir / "whisperx"
        whisper_cache.mkdir(parents=True, exist_ok=True)
        log.info(
            "provision: ensuring faster-whisper model %s is cached in %s",
            settings.whisper_model, whisper_cache,
        )
        faster_whisper.download_model(
            settings.whisper_model, cache_dir=str(whisper_cache)
        )
    except Exception as exc:
        log.warning(
            "provision: faster-whisper pre-fetch failed (%s); whisperx "
            "will retry on first /lyrics/align call.",
            exc,
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
