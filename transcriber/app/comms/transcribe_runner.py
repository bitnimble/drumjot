"""The `transcribe` runner.

`run()` dispatches on the input: a `.zip` debug bundle (a prior pipeline run's
output) goes to `_replay_bundle`, which extracts its predicted MIDI + per-stem
audio into the outputs dir and returns them as artifact path-refs -- exercising
the full sidecar -> protocol -> artifact-delivery path without the GPU/torch
stack. Raw audio goes to `_transcribe_live`, which drives the real
`app.pipeline.runner.run_pipeline` (needs the installed torch capability + a
GPU) on a worker thread, bridging its per-stage progress back to `emit`.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .core import Cancelled, CancelToken, EmitProgress
from .protocol import Artifact, PathRef, RequestMessage

MANIFEST_NAME = "debug.json"
MIDI_NAME = "prediction.mid"
# Mapping key for the drumless backing track (mirrors debug_bundle.NO_DRUMS_KEY).
NO_DRUMS_KEY = "no_drums"
# Pipeline stage order (mirrors app.pipeline.runner.STAGE_ORDER); kept here so
# progress fractions don't require importing the torch-heavy pipeline.
LIVE_STAGES = ["stems_all", "stems_per", "beats", "onsets", "filter", "quantise", "transcribe"]


def _outputs_dir() -> Path:
    base = os.environ.get("DRUMJOT_OUTPUTS_DIR")
    return Path(base) if base else Path(tempfile.gettempdir()) / "drumjot-outputs"


def _input_id(path: Path) -> str:
    """Content-ish id so the same input reuses its output dir."""
    st = path.stat()
    digest = hashlib.sha1(f"{path}:{st.st_size}:{int(st.st_mtime)}".encode())
    return digest.hexdigest()[:16]


def _is_debug_bundle(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return MANIFEST_NAME in zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


class TranscribeRunner:
    async def run(
        self,
        request: RequestMessage,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> list[Artifact]:
        source = request.args.audio
        if not isinstance(source, PathRef):
            raise ValueError("transcribe needs a local file path (remote upload unsupported here)")
        path = Path(source.path)
        if path.suffix == ".zip" and _is_debug_bundle(path):
            return await self._replay_bundle(path, emit, cancel)
        return await self._transcribe_live(path, request.args.params, emit, cancel)

    async def _replay_bundle(
        self,
        bundle: Path,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> list[Artifact]:
        await emit("opening", 0.1, bundle.name)
        out = _outputs_dir() / _input_id(bundle)
        out.mkdir(parents=True, exist_ok=True)
        artifacts: list[Artifact] = []
        with zipfile.ZipFile(bundle) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read(MANIFEST_NAME))
            cancel.check()

            if MIDI_NAME in names:
                await emit("midi", 0.4, None)
                midi_path = out / MIDI_NAME
                midi_path.write_bytes(zf.read(MIDI_NAME))
                artifacts.append(
                    Artifact(role="midi", ref=PathRef(kind="path", path=str(midi_path)))
                )

            # `mapping` aliases several keys to the same file (e.g. `d` -> stem_c);
            # dedup by filename so each stem is written once.
            mapping: dict[str, str] = manifest.get("mapping", {})
            audio_files = list(dict.fromkeys(mapping.values()))
            for i, filename in enumerate(audio_files):
                cancel.check()
                if filename not in names:
                    continue
                await emit("audio", 0.4 + 0.5 * (i + 1) / len(audio_files), filename)
                dest = out / filename
                dest.write_bytes(zf.read(filename))
                role = "audio" if filename == f"{NO_DRUMS_KEY}.mp3" else "stem"
                artifacts.append(Artifact(role=role, ref=PathRef(kind="path", path=str(dest))))

        await emit("done", 1.0, None)
        return artifacts

    async def _transcribe_live(
        self,
        audio_path: Path,
        params: dict[str, object],
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> list[Artifact]:
        """Run the real pipeline on raw audio. The heavy work (torch) runs off
        the event loop via a worker thread; its sync per-stage progress is
        bridged back to the async `emit`, and cooperative cancel flips the
        pipeline's `cancel_event`."""
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict | None] = asyncio.Queue()
        cancel_event = threading.Event()

        def on_progress(event: dict) -> None:  # runs on the pipeline thread
            loop.call_soon_threadsafe(events.put_nowait, event)

        async def pump() -> None:
            while True:
                event = await events.get()
                if event is None:
                    return
                if event.get("phase") == "start":
                    stage = str(event.get("stage", "running"))
                    await emit(stage, _stage_frac(stage), event.get("detail"))

        async def watch_cancel() -> None:
            while not cancel_event.is_set():
                if cancel.cancelled:
                    cancel_event.set()
                    return
                await asyncio.sleep(0.2)

        await emit("starting", 0.0, None)
        pump_task = asyncio.create_task(pump())
        cancel_task = asyncio.create_task(watch_cancel())
        try:
            result = await asyncio.to_thread(
                _run_live_pipeline, audio_path, params, on_progress, cancel_event
            )
        except Exception:
            # The pipeline raises PipelineCancelled (a plain Exception) when it
            # observes the cancel event; surface that as a clean Cancelled rather
            # than a generic "internal" error so the client treats it as an abort.
            if cancel.cancelled:
                raise Cancelled from None
            raise
        finally:
            loop.call_soon_threadsafe(events.put_nowait, None)
            # A pump failure (e.g. a broken stdout write) is already terminal for
            # the stream; don't let it mask the pipeline's real exception
            # propagating out of this finally.
            with contextlib.suppress(Exception):
                await pump_task
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)

        if cancel.cancelled:
            raise Cancelled

        try:
            out = _outputs_dir() / _input_id(audio_path)
            out.mkdir(parents=True, exist_ok=True)
            artifacts: list[Artifact] = []
            if result.midi is not None:
                await emit("midi", 0.95, None)
                midi_path = out / MIDI_NAME
                midi_path.write_bytes(result.midi)
                artifacts.append(Artifact(role="midi", ref=PathRef(kind="path", path=str(midi_path))))
            for role, label, src in result.stems:
                if not Path(src).exists():
                    continue
                dest = out / Path(src).name
                shutil.copyfile(src, dest)
                artifacts.append(
                    Artifact(role=role, name=label, ref=PathRef(kind="path", path=str(dest)))
                )
            await emit("done", 1.0, None)
            return artifacts
        finally:
            # The pipeline's scratch work dir (stems live here until copied above)
            # is no longer needed; drop it so it doesn't accumulate.
            if result.work_dir is not None:
                shutil.rmtree(result.work_dir, ignore_errors=True)


def _stage_frac(stage: str) -> float:
    try:
        return LIVE_STAGES.index(stage) / len(LIVE_STAGES)
    except ValueError:
        return 0.5


@dataclass
class LiveResult:
    midi: bytes | None
    # (artifact role, semantic name, source path) per stem, `name` lets the
    # frontend route a stem to a lane, matching SeparateRunner's labelling.
    stems: list[tuple[str, str, Path]] = field(default_factory=list)
    duration: float = 0.0
    # Scratch dir holding the stems until the caller copies them out; the caller
    # deletes it. None when the runner is monkeypatched in tests.
    work_dir: Path | None = None


def _run_live_pipeline(
    audio_path: Path,
    params: dict[str, object],
    progress: Callable[[dict], None],
    cancel_event: threading.Event,
) -> LiveResult:
    """Run the real transcription pipeline (lazy-imports the torch stack so the
    base sidecar stays light). Mirrors the HTTP /transcribe handler's setup.

    Unit tests monkeypatch this whole function to avoid importing torch / needing
    a GPU; the orchestration around it (progress bridge, output collection) is
    what they cover. A real GPU run is the one piece that can't be validated
    headlessly.
    """
    from app.config import Settings
    from app.pipeline.runner import PipelineContext, PipelineOptions, Stage, run_pipeline
    from app.pipeline.separate import Separator

    settings = Settings()
    work_dir = Path(tempfile.mkdtemp(prefix="drumjot_sidecar_"))
    try:
        ctx = PipelineContext(audio_path=audio_path, work_dir=work_dir)
        ctx.cancel_event = cancel_event
        backend = params.get("onsetBackend")
        beat_input = (
            "drum_stem"
            if (params.get("beatInput") or settings.beat_input_default) == "drum_stem"
            else "full_mix"
        )
        options = PipelineOptions(
            beat_input=beat_input,
            quantise=bool(params.get("quantise", True)),
            quantise_use_llm=bool(params.get("quantiseUseLlm", True)),
            llm_model=str(params.get("llmModel") or settings.llm_model),
            use_learned_onsets=(str(backend).lower() == "learned")
            if backend is not None
            else settings.use_learned_onsets,
            learned_onsets_checkpoint=str(settings.learned_onsets_checkpoint),
        )
        run_pipeline(
            ctx=ctx,
            start_stage=Stage.STEMS_ALL,
            separator=Separator(),
            options=options,
            sink=None,
            output_sink=None,
            progress=progress,
        )
    except BaseException:
        # On any failure/cancel the caller never gets `work_dir` back to clean, so
        # drop the scratch dir here.
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    stems: list[tuple[str, str, Path]] = []
    if ctx.drum_stem is not None:
        stems.append(("stem", "drums", ctx.drum_stem))
    for pitch, stem_path in ctx.per_instrument_stems.items():
        stems.append(("stem", pitch, stem_path))
    return LiveResult(
        midi=ctx.predicted_midi, stems=stems, duration=ctx.duration, work_dir=work_dir
    )
