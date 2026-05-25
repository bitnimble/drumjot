"""Drumjot transcriber HTTP API (FastAPI).

Endpoints:
    GET  /health             - readiness + GPU info
    POST /transcribe         - accept an audio file, return a predicted-onset MIDI
    POST /transcribe/resume  - re-run from a chosen pipeline stage using
                               a previous debug folder's intermediate
                               artifacts

The service is intentionally stateless. All temp files live in per-request
tempdirs. Models are loaded eagerly at container startup (FastAPI
lifespan) so the first /transcribe call doesn't pay model-load latency
and so orchestrators can use /health as a true readiness probe.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.debug import (
    DebugSink,
    mint_request_folder_name,
    reset_current_debug_sink,
    set_current_debug_sink,
)
from app.debug_bundle import build_debug_zip
from app.models import (
    BarSummary,
    HealthResponse,
    TranscribeMetadata,
    TranscribeResponse,
    TranscriptionSummary,
)
from app.outputs import OutputSink, make_output_sink, materialize_pending
from app.pipeline.beats import summarize_bar_for_prompt
from app.pipeline.resume import (
    find_input_audio,
    hydrate_context_from_resume,
    list_transcription_summaries,
    load_original_filename,
)
from app.pipeline.runner import (
    BeatInput,
    PipelineCancelled,
    PipelineContext,
    PipelineOptions,
    Stage,
    StageError,
    run_pipeline,
)
from app.pipeline.separate import Separator
from app.run_log import RunLog, reset_current_run_log, set_current_run_log

# Fallback debug dir used when `debug=true` is requested but `DEBUG_DIR`
# env var wasn't set. Matches the docker-compose volume mount.
DEFAULT_DEBUG_DIR = Path("/debug")

# Map stage -> HTTP status code surfaced when that stage fails. The
# `filter` and `quantise` stages are the only external dependencies
# (Anthropic LLM calls), so they get 502 (bad gateway); everything else
# (including `transcribe`, which is now a pure local render after the
# LLM bit was split out) is local compute and surfaces as 500. The
# quantise stage degrades gracefully on LLM failure (deterministic-only
# fallback) so 502 here only fires on unrecoverable errors before any
# fallback can run, schema mismatches in the deterministic pass, etc.
_STAGE_HTTP_STATUS: dict[Stage, int] = {
    Stage.STEMS_ALL: 500,
    Stage.STEMS_PER: 500,
    Stage.BEATS: 500,
    Stage.ONSETS: 500,
    Stage.FILTER: 502,
    Stage.QUANTISE: 502,
    Stage.TRANSCRIBE: 500,
}


def _require_pipeline_role() -> None:
    """Defense-in-depth: refuse heavy endpoints on the `api` worker.

    Caddy is the source of truth for routing (POSTs to /transcribe go to
    the pipeline worker); this guard only fires if someone bypasses
    Caddy and hits the api worker directly on its private port.
    """
    if settings.worker_role != "pipeline":
        raise HTTPException(
            status_code=503,
            detail=(
                f"This worker is running in '{settings.worker_role}' role "
                "and does not host the transcription pipeline."
            ),
        )


@contextmanager
def _scoped_debug_sink(sink: DebugSink | None) -> Iterator[None]:
    """Install `sink` as the request-scoped current sink for the duration
    of the `with` block, then restore the previous value.

    Deep callees (the LLM wrapper, split/filter helpers) read from a
    ContextVar so they can dump their hydrated prompts to the same
    request folder without having the sink threaded through their
    signatures.
    """
    token = set_current_debug_sink(sink)
    try:
        yield
    finally:
        reset_current_debug_sink(token)


@contextmanager
def _scoped_run_log(run_log: RunLog) -> Iterator[None]:
    """Install `run_log` as the request-scoped run log + attach its
    logging handler, then detach + restore the previous value.

    The handler is the entire point: while installed, every
    `logging.getLogger("app.*")` call routes to the run log in addition to
    its existing destinations, so the debug-bundle JSON captures the same
    stream the operator would have seen via `docker compose logs`.
    """
    run_log.install()
    token = set_current_run_log(run_log)
    try:
        yield
    finally:
        reset_current_run_log(token)
        run_log.uninstall()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The image runs two uvicorn processes (`pipeline` + `api`) behind
    # Caddy — the `api` role serves the lightweight control endpoints
    # while a transcription occupies the pipeline worker. Only the
    # pipeline role touches the GPU; the api role skips the eager model
    # load entirely so it adds ~0 VRAM. See transcriber/entrypoint.sh.
    if settings.worker_role != "pipeline":
        log.info(
            "Starting up in '%s' role: skipping separation-model load.",
            settings.worker_role,
        )
        app.state.separator = None
        yield
        log.info("Shutting down.")
        return

    # Eagerly warm the separation models so the first /transcribe call
    # doesn't pay model-load latency. The model load is blocking I/O +
    # GPU memory allocation, so we run it on a worker thread to avoid
    # blocking the event loop while uvicorn negotiates startup.
    log.info("Starting up: warming separation models...")
    started = time.perf_counter()
    separator = Separator()
    await asyncio.to_thread(separator.load)
    app.state.separator = separator
    log.info(
        "Startup complete in %.2fs - service is ready to accept requests.",
        time.perf_counter() - started,
    )
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Drumjot Transcriber",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve the per-request stem deliverables produced by `OutputSink`. The
# directory is created eagerly so the mount never points at a missing
# path; the volume mount in docker-compose maps it to the host's
# `./outputs/` for inspection and cleanup.
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/outputs",
    StaticFiles(directory=str(settings.outputs_dir)),
    name="outputs",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    gpu_name: str | None = None
    gpu_available = False
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - torch optional at runtime
        log.debug("torch GPU probe failed: %s", exc)
    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )


@app.get("/transcribe/list", response_model=list[TranscriptionSummary])
async def list_transcriptions() -> list[TranscriptionSummary]:
    """Recent /transcribe runs available for resume.

    Walks the configured debug base (`/debug` when `DEBUG_DIR` is unset)
    and emits one summary per per-request subfolder. Sorted with the
    most-recently-run folder first so the UI picker reads top-down by
    recency.
    """
    base = (settings.debug_dir or DEFAULT_DEBUG_DIR).resolve()
    return list_transcription_summaries(base)


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    include_candidates: bool = Form(default=False),
    beat_input: BeatInput = Form(default=settings.beat_input_default),
    quantise: bool = Form(default=True),
    debug: bool = Form(default=False),
) -> StreamingResponse:
    """Streaming NDJSON response: one event per pipeline stage bookend
    (and substage updates inside transcribe), terminated by a single
    `{"type": "result", "data": <TranscribeResponse>}` line.

    Event envelopes (one JSON object per line):
        {"type": "stage", "stage": "stems_all", "phase": "start"}
        {"type": "stage", "stage": "stems_all", "phase": "end", "elapsed_seconds": 5.2}
        {"type": "substage", "stage": "transcribe", "detail": "filtering 3/5 instruments (latest: snare)"}
        {"type": "result", "data": {<TranscribeResponse>}}
        {"type": "error", "status_code": 502, "message": "..."}

    Pre-validation failures (file too large) still return a 4xx with a
    JSON error body — we only switch to NDJSON once the pipeline can
    actually start.
    """
    _require_pipeline_role()
    log.info(
        "Transcribe request: %s (%s bytes) beat_input=%s quantise=%s debug=%s",
        file.filename, file.size, beat_input, quantise, debug,
    )

    if file.size is not None and file.size > 200_000_000:
        raise HTTPException(
            status_code=413, detail="Audio file too large (limit 200 MB)."
        )

    # Persist intermediates iff DEBUG_DIR env is set OR the request explicitly
    # opts in via debug=true. Per-request debug=true uses /debug as fallback
    # so the standard docker-compose volume mount works out of the box.
    debug_base: Path | None = settings.debug_dir
    if debug and debug_base is None:
        debug_base = DEFAULT_DEBUG_DIR
    # Mint a single per-request folder name so the debug and outputs
    # folders use the same slug — operators can correlate them at sight.
    folder_name = mint_request_folder_name(file.filename)
    sink = DebugSink.for_request(debug_base, file.filename, folder_name=folder_name)
    output_sink = make_output_sink(folder_name, settings.outputs_dir)
    run_log = RunLog()
    request_options = {
        "beat_input": beat_input,
        "include_candidates": include_candidates,
        "quantise": quantise,
        "debug": debug,
    }

    # Stage 1 (input write) happens up-front so the temp file is on
    # disk before we hand off to the streaming generator. Reading the
    # upload requires `await`, and `StreamingResponse` is consumed once
    # we return it — so we drain the upload here, then the generator
    # owns the temp dir for the rest of the request.
    work_dir = Path(tempfile.mkdtemp(prefix="drumjot_"))
    in_path = work_dir / (file.filename or "input.wav")
    in_path.write_bytes(await file.read())
    if sink is not None:
        sink.copy_audio("input", in_path)

    ctx = PipelineContext(audio_path=in_path, work_dir=work_dir)
    options = PipelineOptions(beat_input=beat_input, quantise=quantise)
    separator: Separator = request.app.state.separator

    async def post_run() -> None:
        # Backfill any deliverable the stage bodies didn't write before
        # the temp work_dir is torn down. Fresh full runs already wrote
        # the stems in-stage; this is mostly a no-op here but keeps the
        # two endpoints symmetric.
        materialize_pending(
            output_sink,
            drum_stem=ctx.drum_stem,
            per_instrument_stems=ctx.per_instrument_stems,
            residual_stem=ctx.residual_stem,
            predicted_midi=ctx.predicted_midi,
            scavenge_dir=sink.dir if sink is not None else None,
        )

        if sink is not None:
            sink.finalize({
                "filename": file.filename,
                "options": request_options,
                "stems_used": sorted(ctx.per_instrument_stems.keys()),
                "duration_seconds": ctx.duration,
            })

        _build_debug_zip_if_possible(
            output_sink=output_sink,
            ctx=ctx,
            original_filename=file.filename,
            options=request_options,
            run_log=run_log,
        )

    def build_final_response() -> TranscribeResponse:
        return _build_response(
            ctx=ctx,
            include_candidates=include_candidates,
            debug_dir=str(sink.dir) if sink is not None else None,
            output_sink=output_sink,
        )

    return StreamingResponse(
        _stream_pipeline(
            request=request,
            ctx=ctx,
            start_stage=Stage.STEMS_ALL,
            separator=separator,
            options=options,
            sink=sink,
            output_sink=output_sink,
            run_log=run_log,
            post_run=post_run,
            build_final_response=build_final_response,
            cleanup_dir=work_dir,
            verb="Transcribe",
        ),
        media_type="application/x-ndjson",
    )


def _resolve_resume_dir(resume_folder: str) -> Path:
    """Resolve `resume_folder` to an absolute path inside the debug base.

    Accepts either an absolute path (`/debug/20251110-123456_abc12345_song`)
    or a bare folder name resolved against the configured debug base
    (or `DEFAULT_DEBUG_DIR` when `DEBUG_DIR` is unset, matching what the
    standard docker-compose mount produces).

    Refuses paths outside the base so this endpoint can't be used to
    read arbitrary container files (defensive — the operator can already
    reach those via the debug mount, but the endpoint shouldn't widen
    its own attack surface).
    """
    base = (settings.debug_dir or DEFAULT_DEBUG_DIR).resolve()
    candidate = Path(resume_folder)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"resume_folder must be inside the debug base ({base}); "
                f"got {resolved}"
            ),
        ) from exc
    if not resolved.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"resume_folder not found: {resolved}",
        )
    return resolved


@app.post("/transcribe/resume")
async def transcribe_resume(
    request: Request,
    resume_folder: str = Form(...),
    resume_stage: Stage = Form(...),
    include_candidates: bool = Form(default=False),
    beat_input: BeatInput = Form(default=settings.beat_input_default),
    quantise: bool = Form(default=True),
) -> StreamingResponse:
    """Re-run the pipeline from `resume_stage` onward, hydrating any
    artifacts produced by earlier stages from `resume_folder`.

    `resume_stage` is one of:
        `stems_all`, `stems_per`, `beats`, `onsets`, `transcribe`.

    Required artifacts depend on which stages will be skipped:
        - `stems_per`:  `stems_all/drum_stem.<ext>`
        - `beats`:      `stems_per/*.<ext>`
        - `onsets`:     `stems_per/*.<ext>` + `beats.json`
        - `transcribe`: `beats.json` + `onsets.json` + `stems_per/*.<ext>`
    Anything missing comes back as a 400 with a stage-specific message.

    Stages from `resume_stage` onward run fresh and overwrite whichever
    of `prediction.mid`, `note_provenance.json`, `beats.json`, `onsets.json`
    they would normally produce. Upstream artifacts are left intact so
    re-resuming the same folder is idempotent.
    """
    _require_pipeline_role()
    resume_dir = _resolve_resume_dir(resume_folder)
    log.info(
        "Resume request from %s (resume_stage=%s beat_input=%s quantise=%s)",
        resume_dir, resume_stage.value, beat_input, quantise,
    )

    audio_path = find_input_audio(resume_dir) or (resume_dir / "input")
    sink = DebugSink(resume_dir)
    output_sink = make_output_sink(resume_dir.name, settings.outputs_dir)
    options = PipelineOptions(beat_input=beat_input, quantise=quantise)
    separator: Separator = request.app.state.separator
    run_log = RunLog()
    request_options = {
        "beat_input": beat_input,
        "include_candidates": include_candidates,
        "quantise": quantise,
        "resume_folder": str(resume_dir),
        "resume_stage": resume_stage.value,
    }

    # Run stem separation (if it fires) into a temp work_dir so its
    # filenames don't collide with the resume folder; the sink will
    # mirror them into `<resume_dir>/stems_all/` and `/stems_per/`.
    work_dir = Path(tempfile.mkdtemp(prefix="drumjot_resume_"))
    ctx = PipelineContext(audio_path=audio_path, work_dir=work_dir)
    try:
        hydrate_context_from_resume(ctx, resume_dir, resume_stage)
    except FileNotFoundError as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def post_run() -> None:
        # A resume that starts at `beats` or later skips stems_all /
        # stems_per, so the in-stage FLAC writes never fire. The stems
        # are hydrated into `ctx` from the resume folder, and `no_drums`
        # still sits under `<resume_dir>/stems_all/` — backfill every
        # missing deliverable from there, with no recomputation.
        materialize_pending(
            output_sink,
            drum_stem=ctx.drum_stem,
            per_instrument_stems=ctx.per_instrument_stems,
            residual_stem=ctx.residual_stem,
            predicted_midi=ctx.predicted_midi,
            scavenge_dir=resume_dir,
        )

        original_filename = load_original_filename(resume_dir)
        sink.finalize({
            "filename": original_filename,
            "options": request_options,
            "stems_used": sorted(ctx.per_instrument_stems.keys()),
            "duration_seconds": ctx.duration,
        })

        _build_debug_zip_if_possible(
            output_sink=output_sink,
            ctx=ctx,
            original_filename=original_filename,
            options=request_options,
            run_log=run_log,
        )

    def build_final_response() -> TranscribeResponse:
        return _build_response(
            ctx=ctx,
            include_candidates=include_candidates,
            debug_dir=str(resume_dir),
            output_sink=output_sink,
        )

    return StreamingResponse(
        _stream_pipeline(
            request=request,
            ctx=ctx,
            start_stage=resume_stage,
            separator=separator,
            options=options,
            sink=sink,
            output_sink=output_sink,
            run_log=run_log,
            post_run=post_run,
            build_final_response=build_final_response,
            cleanup_dir=work_dir,
            verb="Resume",
        ),
        media_type="application/x-ndjson",
    )


async def _stream_pipeline(
    *,
    request: Request,
    ctx: PipelineContext,
    start_stage: Stage,
    separator: Separator,
    options: PipelineOptions,
    sink: DebugSink | None,
    output_sink: Any,
    run_log: RunLog,
    post_run: Any,
    build_final_response: Any,
    cleanup_dir: Path,
    verb: str,
) -> AsyncIterator[bytes]:
    """Run the pipeline in a worker thread, streaming progress events
    as NDJSON. The pipeline's `progress` callback funnels into an
    asyncio.Queue via `loop.call_soon_threadsafe`; this generator pumps
    that queue out as `{"type": "stage", …}` / `{"type": "substage", …}`
    lines, then yields the final `result` (or `error`) envelope before
    closing.

    Client disconnect (Stop button -> AbortController.abort()) is
    signalled two ways:
      1. `asyncio.CancelledError` raised in the pump loop when Starlette's
         own disconnect listener cancels the response task group (this is
         the primary signal — `request.is_disconnected()` is racy because
         StreamingResponse consumes the same `receive` channel, so the
         listener almost always wins).
      2. A belt-and-suspenders `watch_disconnect` poller — harmless if it
         loses the race, useful in the rare case where it doesn't.
    Either path sets `ctx.cancel_event`. The pipeline checks the event
    between stages and inside the LLM stage's parallel pool, so a cancel
    during stage N lets stage N finish (native code is uninterruptible)
    but skips everything after it. See `pipeline.runner.PipelineCancelled`.

    The work_dir is `rmtree`d in `finally`. On Linux that races
    harmlessly with an in-flight uninterruptible stage's file writes
    (unlinking files that are still open keeps the inode alive until
    the thread's fds close). ContextVars (sink + run_log) are always
    reset.
    """
    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def emit(envelope: dict[str, Any]) -> None:
        # Called from the pipeline worker thread (for "stage"/"substage")
        # and from the async generator itself (for "result"/"error").
        # call_soon_threadsafe is the only thing safe to call on an
        # asyncio.Queue from a non-loop thread.
        loop.call_soon_threadsafe(queue.put_nowait, envelope)

    def progress(event: dict[str, Any]) -> None:
        # event already has {stage, phase?, detail?, elapsed_seconds?}.
        # Tag it as stage vs substage so the frontend can show stage
        # transitions distinctly from in-stage progress ticks.
        kind = "stage" if "phase" in event else "substage"
        emit({"type": kind, **event})

    async def watch_disconnect() -> None:
        # Poll Starlette's receive channel for an http.disconnect frame.
        # The pipeline thread can't be killed, but flipping cancel_event
        # is enough to stop it advancing past the current stage and to
        # short-circuit the LLM stage's parallel pool.
        try:
            while not ctx.cancel_event.is_set():
                if await request.is_disconnected():
                    log.info(
                        "%s: client disconnected; signalling pipeline cancel",
                        verb,
                    )
                    ctx.cancel_event.set()
                    break
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    # ContextVars must be set on the event loop side so the worker
    # thread inherits them via `asyncio.to_thread` (which uses
    # `copy_context()` under the hood).
    sink_token = set_current_debug_sink(sink)
    log_token = set_current_run_log(run_log)
    run_log.install()
    watcher: asyncio.Task[None] | None = None
    task: asyncio.Task[None] | None = None
    try:
        async def run_in_thread() -> None:
            try:
                await asyncio.to_thread(
                    run_pipeline,
                    ctx=ctx,
                    start_stage=start_stage,
                    separator=separator,
                    options=options,
                    sink=sink,
                    output_sink=output_sink,
                    progress=progress,
                )
            except PipelineCancelled as exc:
                # Client disconnected mid-pipeline. Emit an internal
                # sentinel so the streamer can log + exit without trying
                # to write a `result`/`error` to a closed socket.
                emit({
                    "type": "_cancelled",
                    "next_stage": exc.next_stage.value,
                })
            except StageError as exc:
                emit({
                    "type": "error",
                    "status_code": _STAGE_HTTP_STATUS[exc.stage],
                    "stage": exc.stage.value,
                    "message": str(exc),
                })
            except Exception as exc:
                log.exception("Pipeline crashed")
                emit({
                    "type": "error",
                    "status_code": 500,
                    "message": str(exc),
                })
            else:
                # Sentinel: tells the streamer the pipeline finished
                # cleanly — post-run + final result come next.
                emit({"type": "_done"})
            finally:
                emit(None)  # second sentinel: stream may close

        watcher = asyncio.create_task(watch_disconnect())
        task = asyncio.create_task(run_in_thread())
        pipeline_ok = False
        cancelled = False
        try:
            while True:
                envelope = await queue.get()
                if envelope is None:
                    break
                env_type = envelope.get("type")
                if env_type == "_done":
                    pipeline_ok = True
                    continue
                if env_type == "_cancelled":
                    cancelled = True
                    log.info(
                        "%s cancelled (next_stage=%s); skipping result emit",
                        verb, envelope.get("next_stage"),
                    )
                    continue
                yield (json.dumps(envelope) + "\n").encode("utf-8")
            await task  # propagate any unexpected internal failure
        except asyncio.CancelledError:
            # Starlette's disconnect listener cancelled the response.
            # Tell the pipeline to stop at its next stage boundary and
            # re-raise so cleanup runs (we don't try to yield more
            # events into a closed socket).
            ctx.cancel_event.set()
            raise

        if cancelled:
            elapsed = time.perf_counter() - started
            log.info("%s cancelled after %.2fs", verb, elapsed)
            return

        if pipeline_ok:
            try:
                await post_run()
                response = build_final_response()
                yield (
                    json.dumps({
                        "type": "result",
                        "data": response.model_dump(mode="json"),
                    })
                    + "\n"
                ).encode("utf-8")
            except HTTPException as exc:
                yield (
                    json.dumps({
                        "type": "error",
                        "status_code": exc.status_code,
                        "message": str(exc.detail),
                    })
                    + "\n"
                ).encode("utf-8")
            except Exception as exc:
                log.exception("Post-pipeline assembly failed")
                yield (
                    json.dumps({
                        "type": "error",
                        "status_code": 500,
                        "message": str(exc),
                    })
                    + "\n"
                ).encode("utf-8")
        elapsed = time.perf_counter() - started
        log.info("%s complete in %.2fs", verb, elapsed)
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(BaseException):
                await watcher
        # If the client disconnected, `task` may still be running (the
        # pipeline thread is uninterruptible mid-stage). We cancel the
        # asyncio Task to suppress "Task was destroyed but it is pending!"
        # warnings; the underlying thread continues until cancel_event
        # trips at its next stage boundary, then exits cleanly. The
        # rmtree below races with the in-flight stage's file writes but
        # on Linux unlinking files that are still open is safe — the
        # inode persists until the thread's fds close.
        if task is not None and not task.done():
            task.cancel()
        reset_current_debug_sink(sink_token)
        reset_current_run_log(log_token)
        run_log.uninstall()
        shutil.rmtree(cleanup_dir, ignore_errors=True)


def _build_debug_zip_if_possible(
    *,
    output_sink: OutputSink | None,
    ctx: PipelineContext,
    original_filename: str | None,
    options: dict[str, object],
    run_log: RunLog,
) -> None:
    """Best-effort build of the per-request debug zip. Logs and swallows
    failures — a missing bundle must never fail the actual transcribe
    response (the bundle is a debugging aid, not the contract)."""
    if output_sink is None:
        return
    metadata: dict[str, object] | None = None
    if ctx.structure is not None:
        metadata = {
            "initial_tempo": ctx.structure.initial_tempo,
            "initial_time_signature": list(ctx.structure.initial_time_signature),
            "duration_seconds": ctx.duration,
            "stems_used": sorted(ctx.per_instrument_stems.keys()),
            "has_tempo_changes": ctx.structure.has_tempo_changes,
            "has_time_sig_changes": ctx.structure.has_time_sig_changes,
        }
    # The transcribe stage emits prediction.mid; on a resume that skipped
    # transcribe `ctx.predicted_midi` is None even though prediction.mid
    # is still on disk from the prior run, so fall back to the
    # OutputSink's on-disk copy in that case.
    predicted_midi = ctx.predicted_midi
    if predicted_midi is None:
        on_disk = output_sink.existing_file("prediction.mid")
        if on_disk is not None:
            try:
                predicted_midi = on_disk.read_bytes()
            except OSError as exc:
                log.warning(
                    "debug_bundle: could not read %s: %s", on_disk, exc
                )
    # Note provenance follows the same shape — a resume from after the
    # transcribe stage skips the build but the JSON is still on disk in
    # the debug folder from the previous run. The DebugSink writes it
    # under the request folder; for resumes we scavenge from
    # `<resume_dir>/note_provenance.json`.
    note_provenance: dict[str, object] | None = ctx.note_provenance
    if note_provenance is None and ctx.audio_path.parent.name:
        # `audio_path.parent` is the debug folder when resuming (we set
        # the resume folder as the input audio's parent in
        # `transcribe_resume`); the fresh /transcribe path uses a temp
        # work_dir whose parent has no provenance, so the check below
        # gracefully no-ops there.
        candidate = ctx.audio_path.parent / "note_provenance.json"
        if candidate.is_file():
            try:
                import json as _json
                note_provenance = _json.loads(candidate.read_text())
            except (OSError, ValueError) as exc:
                log.warning(
                    "debug_bundle: could not read %s: %s", candidate, exc
                )
    try:
        build_debug_zip(
            output_sink=output_sink,
            original_filename=original_filename,
            options=options,
            metadata=metadata,
            predicted_midi=predicted_midi,
            note_provenance=note_provenance,
            per_instrument_stem_pitches=list(ctx.per_instrument_stems.keys()),
            run_log=run_log,
        )
    except Exception:
        log.exception("debug_bundle: build failed; continuing without zip")


def _build_response(
    *,
    ctx: PipelineContext,
    include_candidates: bool,
    debug_dir: str | None,
    output_sink: OutputSink | None,
) -> TranscribeResponse:
    """Assemble the HTTP response from a completed PipelineContext."""
    if ctx.structure is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed but produced no beat structure.",
        )
    if ctx.predicted_midi is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed but produced no prediction MIDI.",
        )
    bar_summaries = [
        BarSummary(**summarize_bar_for_prompt(b)) for b in ctx.structure.bars
    ]
    stems_used = sorted(ctx.per_instrument_stems.keys())
    # Surface stem URLs only when the FLACs actually exist on disk —
    # covers both fresh /transcribe (just produced) and resume that
    # skipped stems_all but found leftover FLACs from a prior run.
    drum_stem_url: str | None = None
    no_drums_url: str | None = None
    prediction_midi_url: str | None = None
    debug_zip_url: str | None = None
    if output_sink is not None:
        if output_sink.existing_path("drum_stem") is not None:
            drum_stem_url = output_sink.url_for("drum_stem")
        if output_sink.existing_path("no_drums") is not None:
            no_drums_url = output_sink.url_for("no_drums")
        if output_sink.existing_file("prediction.mid") is not None:
            prediction_midi_url = output_sink.url_for_file("prediction.mid")
        if output_sink.existing_file("debug.zip") is not None:
            debug_zip_url = output_sink.url_for_file("debug.zip")
    return TranscribeResponse(
        metadata=TranscribeMetadata(
            initial_tempo=ctx.structure.initial_tempo,
            initial_time_signature=list(ctx.structure.initial_time_signature),
            duration_seconds=ctx.duration,
            stems_used=stems_used,
            bars=bar_summaries,
            has_tempo_changes=ctx.structure.has_tempo_changes,
            has_time_sig_changes=ctx.structure.has_time_sig_changes,
        ),
        candidates=ctx.onsets_by_pitch if include_candidates else {},
        debug_dir=debug_dir,
        drum_stem_url=drum_stem_url,
        no_drums_url=no_drums_url,
        prediction_midi_url=prediction_midi_url,
        debug_zip_url=debug_zip_url,
    )
