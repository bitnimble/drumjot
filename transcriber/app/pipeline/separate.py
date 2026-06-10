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

Both stages run via the `audio-separator` library, which dispatches
inference onto CUDA (or CPU if no GPU is available). Neither model ships
in audio-separator's registry; `pipeline/provision.py` injects them and
fetches their weights on startup (see that module for the mechanism).

Failure modes intentionally surface up to the caller - if the drum-piece
separator can't find a kick, we just won't emit candidates for the kick lane
and the LLM has to infer the kick pattern from context (in practice it can't,
so this is mostly a "log and let the user retry" path).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import settings
from app.debug import current_debug_sink
from app.pipeline.provision import provision_custom_models

log = logging.getLogger(__name__)


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


def _autocast_bf16():
    """TEMP(bf16-trial): bf16 autocast ONLY on GPUs with native bf16 tensor cores
    (Ampere+, compute capability >= 8.0, e.g. the 3080). A no-op everywhere else, CPU, and Turing cards like the GTX 1660 (CC 7.5) which have no bf16 (nor TF32); so those keep the unchanged, correct fp32 path. Wrapped around the
    audio-separator model forward so matmul/conv/attention hit the tensor cores;
    bf16 keeps fp32's exponent range, avoiding the fp16 overflow that NaN'd the
    drum stem, and mdxc_separator.demix accumulates into an fp32 buffer so the
    output upcasts and numpy conversion is unaffected."""
    import contextlib

    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _TEST_check_stem(path: Path, *, require_audio: bool) -> None:
    """TEMP(bf16-trial): assert a separated stem is sound; landed on disk,
    decodes, has no NaN/Inf, and (if `require_audio`) isn't silent. Catches the
    fp16-style failure (overflow -> NaN / silently-dropped file). REMOVE this and
    its call sites once bf16 is validated."""
    import numpy as np
    import soundfile as sf

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"[bf16 check] stem missing or empty on disk: {path}")
    y, _sr = sf.read(str(path), always_2d=True)
    if y.size == 0:
        raise RuntimeError(f"[bf16 check] stem decoded to 0 frames: {path}")
    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        raise RuntimeError(f"[bf16 check] {bad} NaN/Inf samples in {path}")
    if require_audio:
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        if rms < 1e-5:
            raise RuntimeError(f"[bf16 check] stem is silent (rms={rms:.2e}): {path}")


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
        # Opt-in LarsNet Stage-2 separator (the five U-Nets), lazily loaded
        # on first `run_stems_per_larsnet` since most requests use the
        # default MDX23C path and the five nets cost ~590 MB of VRAM. A
        # dict {stem_name: UNetWaveform}; None until first use.
        self._larsnet: dict[str, Any] | None = None
        self._larsnet_device: str = "cpu"

    def load(self) -> None:
        """Idempotently load both separator models.

        Called once at container startup from the FastAPI lifespan hook,
        and again defensively from the per-stage methods so callers that
        bypass the lifespan (e.g. unit tests) still work.
        """
        if self._stems_all is not None and self._stems_per is not None:
            return

        # Neither model is in audio-separator's registry — inject them and
        # fetch their weights BEFORE audio-separator reads the registry /
        # the local files in `load_model()` below.
        provision_custom_models()

        # Local import: pulls in heavy ML deps; only needed in worker processes.
        import torch
        from audio_separator.separator import Separator as AS

        # cuDNN benchmark: every chunk in a separation pass is windowed to
        # exactly chunk_size (see mdxc_separator.py: the tail chunk is
        # re-anchored to `mix[:; -chunk_size:]`); so input shape is fixed
        # across the hot loop — autotune is a free win and has nothing to
        # re-benchmark mid-pass.
        torch.backends.cudnn.benchmark = True

        common = dict(
            output_dir=None,  # set per-call
            model_file_dir=str(settings.models_dir),
            # fp16 autocast: kept off. Re-enabling on BS-Roformer SW
            # produced output that audio-separator logged as written but
            # never landed on disk for the drum stem (sibling stems wrote
            # fine), surfacing as the "reported … but it is not on disk"
            # error downstream. fp32 is slower but correct.
            use_autocast=False,
        )

        t0 = time.perf_counter()
        log.info("Loading stems_all separator (%s) ...", settings.demucs_model)
        self._stems_all = AS(**common)
        self._stems_all.load_model(model_filename=settings.demucs_model)
        _maybe_compile_model(self._stems_all)
        log.info(
            "stems_all ready in %.2fs (%s)",
            time.perf_counter() - t0,
            settings.demucs_model,
        )

        t1 = time.perf_counter()
        log.info("Loading stems_per separator (%s) ...", settings.drum_pieces_model)
        self._stems_per = AS(**common)
        self._stems_per.load_model(model_filename=settings.drum_pieces_model)
        _maybe_compile_model(self._stems_per)
        log.info(
            "stems_per ready in %.2fs (%s)",
            time.perf_counter() - t1,
            settings.drum_pieces_model,
        )
        log.info(
            "Separator ready (total %.2fs).",
            time.perf_counter() - t0,
        )

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

    def run_stems_all(self, audio_path: Path, work_dir: Path) -> StemsAllResult:
        """Extract a drum stem from the full mix, plus a drumless mix.

        Returns a `StemsAllResult` with absolute paths to both. Also
        persists the drum stem, the drumless sum, and any sibling stems
        Demucs emits (bass / other / vocals) into the current debug sink
        under `stems_all/`, so the operator can listen back to
        intermediates while later stages are still running.
        """
        self.load()
        assert self._stems_all is not None

        out_dir = work_dir / "stems_all"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._point_at(self._stems_all, out_dir)

        log.info("stems_all: extracting drum stem from %s", audio_path.name)
        with _autocast_bf16():  # TEMP(bf16-trial)
            raw = self._stems_all.separate(str(audio_path))
        stems_paths = _resolve_outputs(raw, out_dir)
        drum_candidates = [p for p in stems_paths if "drum" in p.stem.lower()]
        if not drum_candidates:
            raise RuntimeError(
                f"stems_all produced no drum stem. Got: {[p.name for p in stems_paths]}"
            )
        drum_stem = drum_candidates[0]
        if not drum_stem.exists():
            raise RuntimeError(
                f"stems_all: separator reported drum stem {drum_stem} but "
                f"it is not on disk (separate() returned {list(raw)!r}). "
                "audio-separator wrote elsewhere or the write failed."
            )
        _TEST_check_stem(drum_stem, require_audio=True)  # TEMP(bf16-trial)
        non_drum_stems = [p for p in stems_paths if p != drum_stem]

        # Sum bass + other + vocals into a single drumless mix so the
        # consumer (and the operator listening through /debug) gets a
        # ready-to-play "music minus drums" track without having to mix
        # the three Demucs sub-stems themselves.
        no_drums_path: Path | None = None
        if non_drum_stems:
            no_drums_path = out_dir / f"no_drums{drum_stem.suffix}"
            try:
                _sum_audio(non_drum_stems, no_drums_path)
            except Exception as exc:
                log.warning("Failed to build drumless mix (%s); skipping.", exc)
                no_drums_path = None

        sink = current_debug_sink()
        if sink is not None:
            sink.copy_audio("stems_all/drum_stem", drum_stem)
            if no_drums_path is not None:
                sink.copy_audio("stems_all/no_drums", no_drums_path)
            for path in non_drum_stems:
                # bass / other / vocals (htdemucs_ft outputs four stems).
                # Kept individually too so the operator can audit which
                # sub-stem contains any drum bleed.
                sink.copy_audio(f"stems_all/{path.stem}", path)
        return StemsAllResult(drum_stem=drum_stem, no_drums=no_drums_path)

    def run_stems_per(self, drum_stem: Path, work_dir: Path) -> StemsPerResult:
        """Split the drum stem into per-instrument stems keyed by pitch.

        Also computes a `residual` track: `drum_stem − sum(per_instrument)`.
        MDX23C is approximately source-additive on the kit classes it was
        trained on, so the residual captures (a) energy from instruments
        the 5-class model has no lane for — cowbell, tambourine, shaker,
        claps, woodblock — and (b) the model's own reconstruction error
        on the supported kit pieces. Diagnostic-only; surfaced into the
        debug bundle but not consumed by any downstream stage.
        """
        self.load()
        assert self._stems_per is not None

        out_dir = work_dir / "stems_per"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._point_at(self._stems_per, out_dir)

        log.info("stems_per: splitting drum stem into pieces")
        with _autocast_bf16():  # TEMP(bf16-trial)
            raw = self._stems_per.separate(str(drum_stem))
        piece_paths = _resolve_outputs(raw, out_dir)

        per_instrument: dict[str, Path] = {}
        for path in piece_paths:
            pitch = _pitch_for_stem_name(path.stem)
            if pitch is None:
                log.info("Skipping unrecognised stem %s", path.name)
                continue
            # First-seen wins so we deterministically pick e.g. crash over
            # china if both are present (rare with the default model).
            per_instrument.setdefault(pitch, path)

        log.info("Recovered %d pitches: %s", len(per_instrument), sorted(per_instrument))

        for _pitch, _p in per_instrument.items():  # TEMP(bf16-trial)
            _TEST_check_stem(_p, require_audio=False)  # per-piece stems can be legitimately silent

        residual_path: Path | None = None
        if per_instrument:
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

    def _resolve_larsnet_device(self) -> str:
        """Map `settings.device` onto LarsNet's torch device string.

        Mirrors `adtof_onsets._resolve_device`: `cpu`/`mps` -> "cpu";
        anything else asks for CUDA and falls back to "cpu" if torch
        reports no GPU.
        """
        import torch

        pref = (settings.device or "auto").lower()
        if pref in ("cpu", "mps"):
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_larsnet(self) -> dict[str, Any]:
        """Lazily build the five LarsNet U-Nets (or re-home them onto the
        device if a prior park moved them to CPU) and return them.

        Loaded on first use, not eagerly, since the default Stage-2 path is
        MDX23C and the five nets cost ~590 MB of VRAM. Raises
        `FileNotFoundError` (-> StageError -> HTTP 500) when the CC-BY-NC
        checkpoints weren't provisioned (`settings.provision_larsnet`).
        """
        from app.pipeline import larsnet as larsnet_pkg
        from app.pipeline.gpu_park import unpark_module

        if self._larsnet is None:
            self._larsnet_device = self._resolve_larsnet_device()
            log.info("Loading LarsNet U-Nets (device=%s) ...", self._larsnet_device)
            t0 = time.perf_counter()
            self._larsnet = larsnet_pkg.load_models(
                Path(settings.models_dir), self._larsnet_device
            )
            log.info("LarsNet ready in %.2fs", time.perf_counter() - t0)
        else:
            for stem, model in self._larsnet.items():
                unpark_module(model, f"larsnet_{stem}")
        assert self._larsnet is not None
        return self._larsnet

    def run_stems_per_larsnet(
        self, drum_stem: Path, work_dir: Path
    ) -> StemsPerResult:
        """LarsNet variant of `run_stems_per`: split the drum stem into the
        same five lanes (k/s/t/h/c) with identical debug / residual
        handling, so it's a drop-in Stage-2 alternative selected by the
        `drum_separator` request option.

        Frees the audio-separator drum models first (Stage 1 is done, and
        the MDX23C Stage-2 model is unused when LarsNet is selected) so the
        five U-Nets fit alongside everything else on a 6 GB GPU.
        """
        # Park the BS-Roformer / MDX23C separators to CPU to make room. They
        # come back at the next /transcribe entry via unpark_drum_models;
        # nothing in-flight needs them once the drum stem exists.
        self._park_audio_separators()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from app.pipeline import larsnet as larsnet_pkg

        models = self._load_larsnet()
        out_dir = work_dir / "stems_per"
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("stems_per (LarsNet): splitting drum stem into pieces")
        stems = larsnet_pkg.separate(models, drum_stem, self._larsnet_device)

        # Write each lane at the drum stem's subtype so LarsNet's outputs
        # sit at the same fidelity as the MDX23C path's per-instrument stems.
        subtype_ref = sf.info(str(drum_stem)).subtype
        per_instrument: dict[str, Path] = {}
        for stem_name, waveform in stems.items():
            pitch = larsnet_pkg.STEM_TO_PITCH.get(stem_name)
            if pitch is None:
                continue
            path = out_dir / f"{pitch}.wav"
            # waveform is [channels, samples]; soundfile wants [samples, channels].
            sf.write(
                str(path),
                waveform.numpy().T,
                larsnet_pkg.SAMPLE_RATE,
                subtype=subtype_ref,
            )
            per_instrument.setdefault(pitch, path)

        log.info("Recovered %d pitches: %s", len(per_instrument), sorted(per_instrument))

        residual_path: Path | None = None
        if per_instrument:
            residual_path = out_dir / f"residual{drum_stem.suffix}"
            try:
                _residual_audio(
                    drum_stem, list(per_instrument.values()), residual_path
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
        model_instance = getattr(separator, "model_instance", None)
        if model_instance is None:
            return None
        inner = getattr(model_instance, "model_run", None)
        if inner is None:
            inner = getattr(model_instance, "model", None)
        return inner

    def _park_audio_separators(self) -> None:
        """Park ONLY the audio-separator drum models (BS-Roformer Stage 1 +
        MDX23C Stage 2) to CPU. `run_stems_per_larsnet` uses this to free
        VRAM for LarsNet without parking LarsNet itself."""
        from app.pipeline.gpu_park import park_module

        for sep, name in (
            (self._stems_all, "stems_all"),
            (self._stems_per, "stems_per"),
        ):
            if sep is None:
                continue
            park_module(self._inner_module(sep), name)

    def park_drum_models(self) -> None:
        from app.pipeline.gpu_park import park_module

        self._park_audio_separators()
        # Also park the LarsNet U-Nets if they were ever loaded, so
        # /lyrics/align reclaims their ~590 MB. They're unparked on demand
        # in `run_stems_per_larsnet`, not by `unpark_drum_models`.
        if self._larsnet is not None:
            for stem, model in self._larsnet.items():
                park_module(model, f"larsnet_{stem}")

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


def _maybe_compile_model(separator: object) -> None:
    """Wrap the inner inference module in `torch.compile` when on CUDA.

    Both stages call `model_instance.model_run(part)` in a tight loop with
    a fixed input shape — exactly the pattern Inductor optimises best.
    Guarded on CUDA because compile cost on CPU often outweighs the win,
    and skipped silently on any compile failure so a torch/version mismatch
    can't break the pipeline.
    """
    import torch

    model_instance = getattr(separator, "model_instance", None)
    if model_instance is None:
        return
    inner = getattr(model_instance, "model_run", None)
    if inner is None:
        return
    try:
        device = next(inner.parameters()).device
    except StopIteration:
        return
    if device.type != "cuda":
        return
    log.info("Compiling %s.model_run with torch.compile", type(model_instance).__name__)
    try:
        model_instance.model_run = torch.compile(inner, dynamic=False)
    except Exception as exc:
        log.warning("torch.compile failed (%s); continuing in eager mode.", exc)


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


def _pitch_for_stem_name(stem_name: str) -> str | None:
    name = stem_name.lower()
    for needle, pitch in STEM_NAME_TO_PITCH.items():
        if needle in name:
            return pitch
    return None


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


def _sum_audio(inputs: list[Path], out_path: Path) -> None:
    """Sample-wise sum of equal-rate wavs into `out_path`.

    Demucs's four stems sum back to (approximately) the original mix, so
    bass + other + vocals is a usable "music minus drums" track without
    additional gain staging. Subtype is preserved from the first input to
    keep file size in line with the per-stem outputs.
    """
    summed: np.ndarray | None = None
    sr_ref: int | None = None
    subtype_ref: str | None = None
    for p in inputs:
        data, sr = sf.read(str(p), always_2d=True, dtype="float32")
        if summed is None:
            summed = data.copy()
            sr_ref = sr
            subtype_ref = sf.info(str(p)).subtype
            continue
        if sr != sr_ref:
            raise RuntimeError(f"sample-rate mismatch summing stems: {p} ({sr}) vs {sr_ref}")
        n = min(summed.shape[0], data.shape[0])
        summed = summed[:n] + data[:n]
    assert summed is not None and sr_ref is not None
    np.clip(summed, -1.0, 1.0, out=summed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), summed, sr_ref, subtype=subtype_ref)
