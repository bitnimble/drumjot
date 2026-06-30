"""Two-stage drum separation: full mix -> drum stem -> per-instrument stems.

Stage `stems_all` uses **BS-Roformer SW** (jarredou's BS-ROFO-SW-Fixed) to
extract a drum stem from the full mix — a 6-stem (vocals / drums / bass /
guitar / piano / other) Band-Split RoPE Transformer chosen over htdemucs_ft
for its substantially cleaner drum stem (drums SDR ~14 vs ~10), especially
its preservation of high-frequency cymbal / hi-hat transients, which is
what Stage 2 then has to split. Stage `stems_per` uses the **jarredou
5-stem MDX23C DrumSep** model to break the drum stem into
kick / snare / toms / hi-hat / cymbals. Note: this 5-stem model merges
ride + crash into a single `cymbals` stem (see `STEM_NAME_TO_PITCH`).

Both stages run through a vendored, torch-only separation wrapper
(`pipeline/separation/`), reimplemented from `audio-separator`'s chunked
overlap-add (validated bit-exact against it) so we can drop the dependency,
surface per-chunk progress, keep stems in memory, and (next) export the model
bodies to ONNX for cross-platform GPU backends. `pipeline/provision.py`
fetches the weights on startup. The /lyrics vocals separator still uses
`audio-separator` for now (a different MDX-ONNX model).

Failure modes intentionally surface up to the caller - if the drum-piece
separator can't find a kick, we just won't emit candidates for the kick lane
and the LLM has to infer the kick pattern from context (in practice it can't,
so this is mostly a "log and let the user retry" path).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import settings
from app.debug import current_debug_sink
from app.pipeline.provision import provision_custom_models, yaml_for_ckpt
from app.pipeline.separation.loader import load_model
from app.pipeline.separation.runner import SAMPLE_RATE, ProgressCallback, SeparationRunner

log = logging.getLogger(__name__)

_BF16_SEP_PATCHED = False


def _bf16_separation_enabled() -> bool:
    """Default ON for native-bf16 Ampere+ GPUs (compute capability >= 8.0).

    bf16 MDX23C separation was validated ~equivalent to fp32 for our purposes
    (onset-position agreement F1 0.999 over 100 maps, 1/500 gate-decision flips)
    at ~1.9x; see RESULTS/commit history. Opt OUT with `DRUMJOT_SEP_BF16=0` (to
    reproduce fp32 byte-for-byte). Always off on pre-Ampere / CPU (no fast native
    bf16). NB this is the SHARED separator, so the default also applies to the
    transcriber API (RoFormer Stage 1 stays fp32 regardless; only MDX23C is bf16)."""
    if os.environ.get("DRUMJOT_SEP_BF16", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _enable_bf16_separation() -> None:
    """Run the separators' matmul-heavy layers in bf16 (~1.9x on the MDX23C path,
    measured on a 3080) while keeping STFT/iSTFT + complex ops in fp32.

    Default ON for Ampere+ (opt out with `DRUMJOT_SEP_BF16=0`). bf16 perturbs the
    spectrally-rich stems (cymbals/hat/toms ~25-40 dB from the fp32 output; kick/
    snare >40 dB; no NaNs, since bf16 keeps fp32's exponent range) but was validated
    not to move onsets or gate decisions (F1 0.999, 1/500 flips). Idempotent, and
    patches the model CLASSES so it also affects already-loaded instances.

    Scope: **MDX23C (Stage 2) only.** BS-/Mel-Band-RoFormer (Stage 1) is left fp32
    on purpose -- measured on a 3080, bf16 makes its drum-stem output deviate from
    fp32 by ~the signal's own energy (~0 to -2 dB, effectively broken, though it
    neither errors nor NaNs): the band-split rotary transformer's deep complex/
    spectral path is too precision-sensitive for bf16's 7-bit mantissa. MDX23C's
    complex STFT primitives have no bf16 kernel either, so we keep STFT fp32 and
    bf16-autocast only the conv body: fp32-guard `STFT.__call__`/`inverse`, autocast
    `TFC_TDF_net.forward`.
    """
    global _BF16_SEP_PATCHED
    if _BF16_SEP_PATCHED:
        return
    import torch

    # --- MDX23C (vendored model classes) ---
    from app.pipeline.separation.architectures import tfc_tdf as _mdx

    _call, _inv, _fwd = _mdx.STFT.__call__, _mdx.STFT.inverse, _mdx.TFC_TDF_net.forward

    def _mdx_call(self, x):
        with torch.autocast("cuda", enabled=False):
            return _call(self, x.float())

    def _mdx_inv(self, x):
        with torch.autocast("cuda", enabled=False):
            return _inv(self, x.float())

    def _mdx_fwd(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _fwd(self, x)

    _mdx.STFT.__call__, _mdx.STFT.inverse, _mdx.TFC_TDF_net.forward = _mdx_call, _mdx_inv, _mdx_fwd

    # BS-/Mel-Band-RoFormer (Stage 1) intentionally left fp32 -- bf16 breaks its
    # output (see docstring); only the MDX23C Stage-2 split runs bf16.

    _BF16_SEP_PATCHED = True
    log.info("bf16 separation ENABLED (DRUMJOT_SEP_BF16): MDX23C matmul=bf16, STFT=fp32; RoFormer stays fp32")


# Map of stem-token substrings found in the separated stem filenames ->
# Drumjot DSL pitch letter. Aligned with `src/midi/gm.ts` defaults so a
# downstream `fromMidi` would land on the same pitches.
#
# Tokens are wrapped in literal `(...)` because the separator models emit
# filenames of the shape `<title>_(Drums)_<stage1>_(<stem>)_<model>.wav`.
# Anchoring the match on the parenthesised segment avoids false-positive
# substring hits against arbitrary characters elsewhere in the filename.
@dataclass
class StemsAllResult:
    """Outputs of the `stems_all` separation stage.

    `drum_stem` feeds the downstream `stems_per` stage (and is also
    FLAC-encoded into the request's outputs folder the instant this stage
    finishes). `no_drums` is the bass+other+vocals sum, used purely as a
    deliverable (FLAC-encoded into the outputs folder + copied to debug) —
    no later stage consumes it. `None` when the sum couldn't be built (e.g.
    Demucs returned only the drum stem on a single-stem variant).
    """

    drum_stem: Path
    no_drums: Path | None


@dataclass
class StemsPerResult:
    """Outputs of the `stems_per` separation stage.

    `per_instrument` maps DSL pitch letter → isolated stem path (the five
    classes the MDX23C model recognises: kick / snare / hi-hat / cymbals
    / toms; cymbals later split into ride+crash downstream). `residual`
    is `drum_stem − sum(per_instrument)` — whatever the 5-class model
    couldn't account for: auxiliary percussion (cowbell, tambourine,
    shaker, claps, woodblock) plus the model's own separation residue
    (un-cancelled bleed, phase/reconstruction error on the supported
    pieces). Diagnostic-only — no downstream stage consumes it, but it's
    surfaced in the debug bundle so the operator can ear-check what fell
    through the seam. `None` when no per-instrument stems were recovered.
    """

    per_instrument: dict[str, Path]
    residual: Path | None


# Pitch letter → display name. Used by the filter prompt and the split
# helpers for human-readable labels in logs / prompts. `H` is a
# synthetic open-hi-hat routing pitch introduced by
# `pipeline/hihat_split.py` so the filter pass can see closed (`h`) and
# open (`H`) hits as separate lanes.
PITCH_DISPLAY_NAMES: dict[str, str] = {
    "k": "Kick",
    "s": "Snare",
    "h": "HiHat",
    "H": "Open Hi-Hat",
    "d": "Ride",
    "c": "Crash",
    "t": "Tom",
}


STEM_NAME_TO_PITCH: dict[str, str] = {
    "(kick)": "k",
    "(snare)": "s",
    "(hihat)": "h",
    "(hi-hat)": "h",
    "(hh)": "h",
    "(hat)": "h",
    # The active Stage-2 model (jarredou 5-stem DrumSep) merges ride +
    # crash into ONE `cymbals` stem, so there is no separate ride/crash
    # source here: route the merged cymbals stem to the `c` lane as its
    # carrier. `pipeline/cymbal_split.py` then splits that lane back into
    # ride (`d`) / crash (`c`) downstream (deterministic features + LLM).
    # `(ride)` / `(crash)` are kept for forward-compat if a 6-stem model
    # is ever swapped back in; first-seen-wins in `run_stems_per` keeps
    # this deterministic.
    "(cymbals)": "c",
    "(ride)": "d",  # 'd' for ride - avoids the `:r` rim-shot modifier clash
    "(crash)": "c",
    # Toms tend to ship as the plural token `(toms)` in DrumSep output;
    # keep the singular form too for forward compatibility.
    "(tom)": "t",
    "(toms)": "t",
}


# Our vendored runner returns BARE instrument names ("kick", "hh", ...), not the
# parenthesised filename tokens audio-separator emitted. Derive the bare-name ->
# pitch map from STEM_NAME_TO_PITCH so the two stay in sync.
_INSTRUMENT_TO_PITCH: dict[str, str] = {
    token.strip("()"): pitch for token, pitch in STEM_NAME_TO_PITCH.items()
}


class Separator:
    """Two-stage drum separator. Models are loaded eagerly by `load()` at
    application startup so the first `/transcribe` call doesn't pay
    model-load latency.

    Model weights are downloaded into `settings.models_dir` (mounted as a
    Docker volume so they persist across container restarts).

    The two stages are exposed as independent methods (`run_stems_all`,
    `run_stems_per`) so the pipeline runner can resume from either one
    without having to re-run the other.
    """

    def __init__(self) -> None:
        self._stems_all = None
        self._stems_per = None
        self._vocals = None

    def load(self, *, stems_all: bool = True, stems_per: bool = True) -> None:
        """Idempotently load the requested separator models.

        `stems_all` = BS-Roformer Stage-1 (drum-stem extraction); `stems_per` =
        MDX23C Stage-2 (per-instrument split). The FastAPI lifespan (and unit
        tests) load both via the no-arg default; a caller that runs only one
        stage -- e.g. the batch per-instrument data generator, which feeds
        pre-extracted drum stems straight into Stage 2 -- can load just that
        model and skip the other's ~700 MB of VRAM + load time. Called once at
        container startup and again defensively from each per-stage method.
        """
        if _bf16_separation_enabled():  # default-on bf16 (Ampere); opt out DRUMJOT_SEP_BF16=0
            _enable_bf16_separation()
        need_all = stems_all and self._stems_all is None
        need_per = stems_per and self._stems_per is None
        if not need_all and not need_per:
            return

        # Neither model is in audio-separator's registry — inject them and
        # fetch their weights BEFORE audio-separator reads the registry /
        # the local files in `load_model()` below.
        provision_custom_models()

        # Local import: pulls in heavy ML deps; only needed in worker processes.
        import torch

        # cuDNN benchmark: every chunk in a separation pass is windowed to a
        # fixed chunk_size, so input shape is fixed across the hot loop.
        # Autotune is a free win with nothing to re-benchmark mid-pass.
        torch.backends.cudnn.benchmark = True

        # TF32: lets fp32 matmuls use the Ampere+ tensor-core path (≈2× on the
        # 3080) WITHOUT changing any tensor dtype, so the models' complex STFT
        # (view_as_complex) stays fp32 and there's no range/NaN risk. A harmless
        # no-op on Turing (1660) / older cards. NB autocast is a dead end for
        # these separators: fp16 overflowed and the drum stem NaN'd out, and bf16
        # fails outright ("view_as_complex is only supported for half, float and
        # double"). TF32 is the only tensor-core path compatible with them.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        models_dir = Path(settings.models_dir)
        device = _resolve_device()

        t0 = time.perf_counter()
        if need_all:
            log.info("Loading stems_all separator (%s) ...", settings.demucs_model)
            self._stems_all = self._load_runner(settings.demucs_model, models_dir, device)
            log.info(
                "stems_all ready in %.2fs (%s)", time.perf_counter() - t0, settings.demucs_model
            )

        t1 = time.perf_counter()
        if need_per:
            log.info("Loading stems_per separator (%s) ...", settings.drum_pieces_model)
            self._stems_per = self._load_runner(settings.drum_pieces_model, models_dir, device)
            log.info(
                "stems_per ready in %.2fs (%s)",
                time.perf_counter() - t1,
                settings.drum_pieces_model,
            )
        log.info("Separator ready (total %.2fs).", time.perf_counter() - t0)

    def _load_runner(self, ckpt_filename: str, models_dir: Path, device: str) -> SeparationRunner:
        """Build a `SeparationRunner` for one custom model from its on-disk
        (ckpt, yaml) pair. `yaml_for_ckpt` resolves the paired config (the
        bare state_dict can't be instantiated without it)."""
        ckpt = models_dir / ckpt_filename
        yaml = models_dir / yaml_for_ckpt(ckpt_filename)
        loaded = load_model(ckpt, yaml, device=device)
        if _onnx_separation_enabled():
            _attach_onnx_body(loaded, ckpt, models_dir)
        else:
            _maybe_compile_model(loaded)
        return SeparationRunner(loaded, device=device)

    @staticmethod
    def _point_at(separator: object, out_dir: Path) -> None:
        """Make `audio-separator` write this call's stems into `out_dir`.

        `Separator.__init__` resolves `output_dir=None` to `os.getcwd()`
        (here `/app`, the container WORKDIR) and `load_model()` *bakes*
        that value into the loaded `model_instance` via its
        `common_params` — at startup, long before we know the per-request
        tmp dir. In `audio-separator>=0.44` the per-call
        `separator.output_dir` setter is **not** re-propagated to
        `model_instance` at `separate()` time (that re-sync line only
        exists on newer `main`), so the model keeps writing to the
        startup cwd.

        The model's output-path construction reads
        `model_instance.output_dir`, so we set it directly (plus the
        wrapper attribute, which newer versions *do* re-read). This is
        the same attribute `audio-separator` assigns internally, so it's
        stable across the contract drift.
        """
        out = str(out_dir)
        separator.output_dir = out  # type: ignore[attr-defined]
        model = getattr(separator, "model_instance", None)
        if model is not None:
            model.output_dir = out
        else:
            # Lazy-loaded by some versions on first separate(); the
            # wrapper setter above is then the only lever.
            log.warning(
                "audio-separator has no model_instance yet; relying on "
                "wrapper output_dir alone for %s",
                out,
            )

    def run_stems_all(
        self, audio_path: Path, work_dir: Path, *, build_no_drums: bool = True
    ) -> StemsAllResult:
        """Extract a drum stem from the full mix, plus a drumless mix.

        Returns a `StemsAllResult` with absolute paths to both. Also
        persists the drum stem, the drumless sum, and any sibling stems
        Demucs emits (bass / other / vocals) into the current debug sink
        under `stems_all/`, so the operator can listen back to
        intermediates while later stages are still running.

        `build_no_drums=False` skips summing the non-drum stems into the
        "music minus drums" track (returns `no_drums=None`). Batch callers that
        only want the drum stem pass this to avoid the wasted sum and, on
        drums-only input, the noisy "Failed to build drumless mix" warning the
        empty piano/guitar stems otherwise trigger.
        """
        self.load(stems_per=False)
        assert self._stems_all is not None

        out_dir = work_dir / "stems_all"
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("stems_all: extracting drum stem from %s", audio_path.name)
        sources = self._stems_all.separate(
            str(audio_path), progress_callback=_log_progress("stems_all")
        )
        if "drums" not in sources:
            raise RuntimeError(f"stems_all produced no drum stem. Got: {sorted(sources)}")

        drum_stem = out_dir / "drums.wav"
        _write_stem(drum_stem, sources["drums"])

        non_drum = {name: wave for name, wave in sources.items() if name != "drums"}

        # Sum the non-drum stems (bass / other / vocals / guitar / piano) into a
        # single drumless mix so the consumer (and the operator listening through
        # /debug) gets a ready-to-play "music minus drums" track. Summed in
        # memory, the individual non-drum stems only hit disk when a debug sink
        # is active (no consumer needs them otherwise).
        no_drums_path: Path | None = None
        if build_no_drums and non_drum:
            no_drums_path = out_dir / "no_drums.wav"
            try:
                _write_stem(no_drums_path, _sum_stems(list(non_drum.values())))
            except Exception as exc:
                log.warning("Failed to build drumless mix (%s); skipping.", exc)
                no_drums_path = None

        sink = current_debug_sink()
        if sink is not None:
            sink.copy_audio("stems_all/drum_stem", drum_stem)
            if no_drums_path is not None:
                sink.copy_audio("stems_all/no_drums", no_drums_path)
            for name, wave in non_drum.items():
                # Kept individually so the operator can audit which sub-stem
                # contains any drum bleed.
                path = out_dir / f"{name}.wav"
                _write_stem(path, wave)
                sink.copy_audio(f"stems_all/{name}", path)
        return StemsAllResult(drum_stem=drum_stem, no_drums=no_drums_path)

    def run_stems_per(
        self, drum_stem: Path, work_dir: Path, *, build_residual: bool = True
    ) -> StemsPerResult:
        """Split the drum stem into per-instrument stems keyed by pitch.

        `build_residual=False` skips the diagnostic residual track (returns
        `residual=None`). That track re-reads the drum stem + all 5 per-stems
        and writes a file -- pure overhead for callers that don't consume it
        (e.g. the batch training-data generator), where it otherwise sits on
        the GPU thread's critical path between batches.

        When built, computes a `residual` track: `drum_stem − sum(per_instrument)`.
        MDX23C is approximately source-additive on the kit classes it was
        trained on, so the residual captures (a) energy from instruments
        the 5-class model has no lane for — cowbell, tambourine, shaker,
        claps, woodblock — and (b) the model's own reconstruction error
        on the supported kit pieces. Diagnostic-only; surfaced into the
        debug bundle but not consumed by any downstream stage.
        """
        self.load(stems_all=False)
        assert self._stems_per is not None

        out_dir = work_dir / "stems_per"
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("stems_per: splitting drum stem into pieces")
        sources = self._stems_per.separate(
            str(drum_stem), progress_callback=_log_progress("stems_per")
        )

        per_instrument: dict[str, Path] = {}
        for name, wave in sources.items():
            pitch = _INSTRUMENT_TO_PITCH.get(name.lower())
            if pitch is None:
                log.info("Skipping unrecognised stem %s", name)
                continue
            if pitch in per_instrument:
                continue  # first-seen wins, for deterministic pitch routing
            path = out_dir / f"{name}.wav"
            _write_stem(path, wave)
            per_instrument[pitch] = path

        log.info("Recovered %d pitches: %s", len(per_instrument), sorted(per_instrument))

        residual_path: Path | None = None
        if build_residual and per_instrument:
            residual_path = out_dir / f"residual{drum_stem.suffix}"
            try:
                _residual_audio(
                    drum_stem,
                    list(per_instrument.values()),
                    residual_path,
                )
            except Exception as exc:
                log.warning(
                    "Failed to build per-instrument residual track (%s); skipping.",
                    exc,
                )
                residual_path = None

        sink = current_debug_sink()
        if sink is not None:
            for pitch, path in per_instrument.items():
                sink.copy_audio(f"stems_per/{pitch}", path)
            if residual_path is not None:
                sink.copy_audio("stems_per/residual", residual_path)
        return StemsPerResult(per_instrument=per_instrument, residual=residual_path)

    # ---- GPU residency control --------------------------------------
    # `park_*` / `unpark_*` move the wrapped nn.Module between CUDA
    # and CPU so the two endpoints can swap GPU ownership without
    # paying a disk-reload. Coordinated by `app.pipeline.gpu_park`;
    # callers must hold the process-wide GPU lock (see main.py) so an
    # in-flight stage isn't mid-forward through a model that's about
    # to move host-side. Each is idempotent and a no-op when the
    # wrapped audio-separator hasn't loaded a model_instance yet.
    #
    # The wrapped model lives at `model_instance.model_run`; after
    # `_maybe_compile_model` that's the torch.compile OptimizedModule,
    # which still routes `.to()` through to the underlying nn.Module.
    # We also try `.model` as a fallback in case a future audio-separator
    # version stops mutating `model_run`.

    @staticmethod
    def _inner_module(separator: object) -> object | None:
        if isinstance(separator, SeparationRunner):
            return separator.model  # vendored nn.Module (the two drum stages)
        # audio-separator wrapper (still used by the vocals / lyrics path).
        model_instance = getattr(separator, "model_instance", None)
        if model_instance is None:
            return None
        inner = getattr(model_instance, "model_run", None)
        if inner is None:
            inner = getattr(model_instance, "model", None)
        return inner

    def park_drum_models(self) -> None:
        """Park the audio-separator drum models (BS-Roformer Stage 1 +
        MDX23C Stage 2) to CPU so /lyrics/align reclaims their VRAM."""
        from app.pipeline.gpu_park import park_module

        for sep, name in (
            (self._stems_all, "stems_all"),
            (self._stems_per, "stems_per"),
        ):
            if sep is None:
                continue
            park_module(self._inner_module(sep), name)

    def unpark_drum_models(self) -> None:
        from app.pipeline.gpu_park import unpark_module

        for sep, name in (
            (self._stems_all, "stems_all"),
            (self._stems_per, "stems_per"),
        ):
            if sep is None:
                continue
            unpark_module(self._inner_module(sep), name)

    def park_vocals(self) -> None:
        """Free the vocals separator's GPU memory before the CTC aligner
        loads.

        The vocals model (`UVR-MDX-NET-Voc_FT.onnx`) is MDX/ONNX: its
        `model_instance.model_run` is an ONNX Runtime inference-session
        lambda, and ORT holds its CUDA arena (~2 GB) outside torch's
        allocator. A host memcpy (`park_module` -> `.to("cpu")`) can't
        reach that memory (and the lambda has no `.parameters()` to begin
        with), so for the ORT path we release the whole separator; the
        next `run_vocals` reloads it lazily (`_load_vocals`). A torch-
        backed inner module (the onnx2torch fallback path) still parks to
        CPU, which is cheaper than a reload.

        Idempotent: a no-op when the vocals separator was never loaded
        (e.g. a disk cache hit fed the aligner directly)."""
        if self._vocals is None:
            return
        inner = self._inner_module(self._vocals)
        if inner is not None and callable(getattr(inner, "parameters", None)):
            from app.pipeline.gpu_park import park_module

            park_module(inner, "vocals")
            return
        self._release_vocals()

    def _release_vocals(self) -> None:
        """Drop the vocals separator so its ONNX Runtime CUDA arena is
        returned to the driver. Idempotent. Runs under the process-wide
        GPU lock (its only caller is `park_vocals`)."""
        if self._vocals is None:
            return
        import gc

        self._vocals = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("gpu_park: released vocals separator (ONNX Runtime session)")

    def unpark_vocals(self) -> None:
        from app.pipeline.gpu_park import unpark_module

        if self._vocals is None:
            return
        unpark_module(self._inner_module(self._vocals), "vocals")

    def _load_vocals(self) -> None:
        """Lazily load the vocals-only separator used by /lyrics/align.

        Kept out of eager `load()` because the drum pipeline never touches
        this model; only callers hitting /lyrics/align with `mode=mix` pay
        the load cost, on first use. Idempotent.
        """
        if self._vocals is not None:
            return
        from audio_separator.separator import Separator as AS

        common = dict(
            output_dir=None,
            model_file_dir=str(settings.models_dir),
            use_autocast=False,
        )
        t0 = time.perf_counter()
        log.info("Loading vocals separator (%s) ...", settings.vocals_model)
        self._vocals = AS(**common)
        self._vocals.load_model(model_filename=settings.vocals_model)
        log.info(
            "vocals separator ready in %.2fs (%s)",
            time.perf_counter() - t0,
            settings.vocals_model,
        )

    def run_vocals(self, audio_path: Path, work_dir: Path) -> Path | None:
        """Extract a vocals stem from a full mix for CTC forced alignment.

        Uses the dedicated 2-stem `vocals_model` (not the drum pipeline's
        6-stem BS-Roformer SW) so latency is dominated by what the
        aligner actually needs. Returns the absolute path to the vocals output,
        or None when the model ran but no vocals-named output landed
        (would indicate a model swap that no longer emits a `(Vocals)`
        filename token).
        """
        self._load_vocals()
        assert self._vocals is not None

        out_dir = work_dir / "vocals"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._point_at(self._vocals, out_dir)

        log.info("vocals: extracting vocals stem from %s", audio_path.name)
        t0 = time.perf_counter()
        raw = self._vocals.separate(str(audio_path))
        stems_paths = _resolve_outputs(raw, out_dir)
        vocals_candidates = [p for p in stems_paths if "vocals" in p.stem.lower()]
        if not vocals_candidates:
            log.info(
                "vocals: separation finished in %.2fs but no vocals stem in output (%s)",
                time.perf_counter() - t0,
                settings.vocals_model,
            )
            return None
        vocals_stem = vocals_candidates[0]
        if not vocals_stem.exists():
            raise RuntimeError(
                f"vocals: separator reported {vocals_stem} but it is not on "
                f"disk (separate() returned {list(raw)!r})."
            )
        log.info(
            "vocals: extracted in %.2fs (%s)",
            time.perf_counter() - t0,
            settings.vocals_model,
        )
        return vocals_stem


def _resolve_device() -> str:
    """`settings.device` ("auto" by default) resolved to a concrete device."""
    import torch

    if settings.device and settings.device != "auto":
        return settings.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _log_progress(stage: str) -> ProgressCallback:
    """Per-chunk progress hook for a separation pass. Currently logs at DEBUG;
    the callback is the seam a future progress-bar / ETA UI plugs into."""

    def _cb(done: int, total: int) -> None:
        log.debug("%s: chunk %d/%d", stage, done, total)

    return _cb


def _write_stem(path: Path, wave: np.ndarray) -> None:
    """Write an in-memory stem (channels, samples) to `path` as 16-bit WAV.

    The runner returns (channels, samples) (audio-separator's pre-write shape);
    soundfile wants (samples, channels). PCM_16 matches the prior on-disk
    fidelity (these are FLAC-re-encoded downstream)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.ascontiguousarray(wave.T), SAMPLE_RATE, subtype="PCM_16")


def _sum_stems(stems: list[np.ndarray]) -> np.ndarray:
    """Sample-wise sum of equal-rate (channels, samples) stems, clipped to
    [-1, 1]. In memory, so the non-drum stems never need to hit disk."""
    min_len = min(s.shape[1] for s in stems)
    summed = np.zeros((stems[0].shape[0], min_len), dtype=np.float32)
    for s in stems:
        summed += s[:, :min_len]
    np.clip(summed, -1.0, 1.0, out=summed)
    return summed


def _maybe_compile_model(loaded: object) -> None:
    """Wrap the model's inner module in `torch.compile` when on CUDA.

    The separation loop calls the model in a tight, fixed-input-shape loop,
    exactly the pattern Inductor optimises best. Guarded on CUDA because
    compile cost on CPU often outweighs the win, and skipped silently on any
    compile failure so a torch/version mismatch can't break the pipeline.
    Mutates `loaded.model` in place, before the runner reads it.
    """
    import torch

    model = getattr(loaded, "model", None)
    if model is None:
        return
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type != "cuda":
        return
    log.info("Compiling %s with torch.compile", type(model).__name__)
    try:
        loaded.model = torch.compile(model, dynamic=False)
    except Exception as exc:
        log.warning("torch.compile failed (%s); continuing in eager mode.", exc)


def _onnx_separation_enabled() -> bool:
    """Route the drum-separator BODIES through onnxruntime instead of torch (the
    STFT/iSTFT stay torch/fp32). Off by default; opt in with DRUMJOT_SEP_ONNX=1.

    This is the cross-platform path: onnxruntime dispatches the body to whatever
    execution provider the installed build supports (CUDA / DirectML / CoreML /
    ROCm, else CPU). On NVIDIA the torch+bf16 path is usually faster, so the
    default stays torch; flip this on for AMD / Apple / Intel backends."""
    return os.environ.get("DRUMJOT_SEP_ONNX", "").strip().lower() in (
        "1", "true", "yes", "on", "fp16", "16", "half"
    )


def _onnx_separation_fp16() -> bool:
    """DRUMJOT_SEP_ONNX=fp16 -> export an fp16 body (~half the file, GPU tensor /
    NPU fp16 path). ~10% magnitude error vs fp32 but shape preserved (corr ~1.0);
    the STFT/iSTFT stay fp32 in torch regardless."""
    return os.environ.get("DRUMJOT_SEP_ONNX", "").strip().lower() in ("fp16", "16", "half")


def _attach_onnx_body(loaded: object, ckpt_path: Path, models_dir: Path) -> None:
    """Export (once, cached next to the ckpt) and attach the model's ONNX body so
    the runner routes body compute through onnxruntime. The first export is heavy
    (full-size graph, minutes); later loads reuse the cached `.onnx`."""
    from app.pipeline.separation.export import export_body
    from app.pipeline.separation.loader import attach_onnx_session

    fp16 = _onnx_separation_fp16()
    onnx_path = models_dir / (ckpt_path.stem + (".fp16.onnx" if fp16 else ".onnx"))
    if not onnx_path.exists():
        log.info(
            "Exporting %s body to ONNX%s (one-time, cached) ...",
            ckpt_path.name,
            " fp16" if fp16 else "",
        )
        export_body(loaded, onnx_path, fp16=fp16)  # type: ignore[arg-type]
    attach_onnx_session(loaded, onnx_path)  # type: ignore[arg-type]
    providers = loaded.onnx_session.get_providers()  # type: ignore[attr-defined]
    log.info(
        "ONNX separation ENABLED for %s (%s, providers=%s)",
        ckpt_path.name,
        "fp16" if fp16 else "fp32",
        providers,
    )


def _resolve_outputs(raw_paths: list[str], out_dir: Path) -> list[Path]:
    """Anchor `audio-separator`'s `separate()` return to absolute paths.

    With `Separator._point_at` the library writes into `out_dir`; it
    returns bare basenames relative to it (some versions return absolute
    paths). Anchor any non-absolute entry to `out_dir`; absolute paths
    pass through unchanged. The caller's `drum_stem.exists()` check
    catches the case where the library still wrote elsewhere.
    """
    resolved: list[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        resolved.append(p if p.is_absolute() else out_dir / p.name)
    return resolved


def _residual_audio(
    drum_stem: Path,
    stems: list[Path],
    out_path: Path,
) -> None:
    """Write `drum_stem − sum(stems)` to `out_path`.

    Channel/sample-rate parity with `drum_stem` is required (MDX23C
    outputs inherit both from its input, so this holds in practice). The
    write uses the drum stem's subtype so the residual sits at the same
    fidelity as the per-instrument stems. Clipping is post-mix because
    the subtracted sum is bounded by the same scale as the drum stem —
    any out-of-range excursion is separator reconstruction error, not
    musical content.
    """
    drum, sr = sf.read(str(drum_stem), always_2d=True, dtype="float32")
    subtype_ref = sf.info(str(drum_stem)).subtype
    summed: np.ndarray = np.zeros_like(drum)
    min_len = drum.shape[0]
    for p in stems:
        data, stem_sr = sf.read(str(p), always_2d=True, dtype="float32")
        if stem_sr != sr:
            raise RuntimeError(
                f"sample-rate mismatch building residual: {p} ({stem_sr}) vs drum_stem ({sr})"
            )
        if data.shape[1] != drum.shape[1]:
            raise RuntimeError(
                f"channel-count mismatch building residual: {p} "
                f"({data.shape[1]}) vs drum_stem ({drum.shape[1]})"
            )
        n = min(data.shape[0], summed.shape[0])
        summed[:n] += data[:n]
        min_len = min(min_len, data.shape[0])
    residual = drum[:min_len] - summed[:min_len]
    np.clip(residual, -1.0, 1.0, out=residual)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), residual, sr, subtype=subtype_ref)
