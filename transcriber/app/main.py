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
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.cache import BlobCache
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
from app.pipeline import gpu_park
from app.pipeline.beats import summarize_bar_for_prompt
from app.pipeline.lyrics_align import InputLine, get_aligner, lines_to_json
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
from app.request_context import (
    RequestIdLogFilter,
    new_request_id,
    set_request_id,
)
from app.run_log import RunLog, reset_current_run_log, set_current_run_log
from app.scoring.score_map import score_midi, score_paradb

# Fallback debug dir used when `debug=true` is requested but `DEBUG_DIR`
# env var wasn't set. Matches the docker-compose volume mount.
DEFAULT_DEBUG_DIR = Path("/debug")

# How long the streaming endpoints will sit silent (no real progress
# event) before emitting a `{"type": "heartbeat"}` NDJSON line. The heavy
# stages (Demucs separation, beats, onsets) produce no downstream bytes
# for tens of seconds; without a keepalive an intermediary proxy with an
# idle timeout drops the connection and the client sees a broken pipe.
# The frontend (`src/transcriber.ts`) ignores unknown event types, so
# heartbeats are inert there. 10 s is comfortably under typical proxy
# idle timeouts (commonly 30-60 s) while staying low-chatter.
HEARTBEAT_INTERVAL_SECONDS = 10.0


# Process-wide GPU lock. The two heavy endpoints (/transcribe(/resume),
# /lyrics/align) take this before doing any model work so a second
# request can't move a model to CPU while the first is mid-forward
# through it. Also serialises against audio-separator's non-thread-safe
# `.separate()` state. A queued second request waits; the GPU is a
# single resource, so concurrency wouldn't make either request faster
# anyway. /transcribe(/resume) emits a `{"type": "queued"}` NDJSON line
# before awaiting on contention so the streaming UI can show a wait
# state instead of a blank stream.
_gpu_lock = asyncio.Lock()

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


# Build the single root StreamHandler by hand so we can attach the
# request-id filter to it. The format string references %(request_id)s,
# which would raise during formatting on any record lacking that
# attribute, RequestIdLogFilter always sets it, so the filter MUST live
# on this exact handler. basicConfig(handlers=[...]) installs it on the
# root logger; app.* loggers propagate up to it, so every pipeline log
# line picks up the id without per-logger wiring.
_root_handler = logging.StreamHandler()
_root_handler.addFilter(RequestIdLogFilter())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s",
    handlers=[_root_handler],
)
log = logging.getLogger(__name__)


class _DropHealthAccessLog(logging.Filter):
    """Drop uvicorn access-log lines for `GET /health`. Caddy hits the
    health endpoint on a tight liveness interval; logging every probe
    drowns out everything else in the container logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            method, path = args[1], args[2]
            if method == "GET" and isinstance(path, str) and path.startswith("/health"):
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(_DropHealthAccessLog())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The image runs two uvicorn processes (`pipeline` + `api`) behind
    # Caddy — the `api` role serves the lightweight control endpoints
    # while a transcription occupies the pipeline worker. Only the
    # pipeline role touches the GPU; the api role skips the eager model
    # load entirely so it adds ~0 VRAM. See docker/entrypoint.sh.
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
    quantise_use_llm: bool = Form(default=True),
    llm_model: str = Form(default=""),
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
    # Mint + bind the request id before the first log line so it (and
    # every later line on this request, including worker-thread LLM logs)
    # carries the same id. Bound again at the top of `_stream_pipeline`
    # because Starlette consumes the response body generator in a
    # different context than this handler, see that function's note.
    request_id = new_request_id()
    set_request_id(request_id)
    log.info(
        "Transcribe request: %s (%s bytes) beat_input=%s quantise=%s llm_model=%s debug=%s",
        file.filename, file.size, beat_input, quantise,
        llm_model or settings.llm_model, debug,
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
        "quantise_use_llm": quantise_use_llm,
        "llm_model": llm_model or settings.llm_model,
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
    options = PipelineOptions(
        beat_input=beat_input,
        quantise=quantise,
        quantise_use_llm=quantise_use_llm,
        llm_model=llm_model or settings.llm_model,
    )
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
            request_id=request_id,
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
    quantise_use_llm: bool = Form(default=True),
    llm_model: str = Form(default=""),
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
    # See `transcribe`: bind the request id before the first log line and
    # again at the top of `_stream_pipeline`.
    request_id = new_request_id()
    set_request_id(request_id)
    resume_dir = _resolve_resume_dir(resume_folder)
    resolved_model = llm_model or settings.llm_model
    log.info(
        "Resume request from %s (resume_stage=%s beat_input=%s quantise=%s llm_model=%s)",
        resume_dir, resume_stage.value, beat_input, quantise, resolved_model,
    )

    audio_path = find_input_audio(resume_dir) or (resume_dir / "input")
    sink = DebugSink(resume_dir)
    output_sink = make_output_sink(resume_dir.name, settings.outputs_dir)
    options = PipelineOptions(
        beat_input=beat_input,
        quantise=quantise,
        quantise_use_llm=quantise_use_llm,
        llm_model=resolved_model,
    )
    separator: Separator = request.app.state.separator
    run_log = RunLog()
    request_options = {
        "beat_input": beat_input,
        "include_candidates": include_candidates,
        "quantise": quantise,
        "quantise_use_llm": quantise_use_llm,
        "llm_model": resolved_model,
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
            request_id=request_id,
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


@app.post("/lyrics/align")
async def lyrics_align(
    request: Request,
    vocals: UploadFile | None = File(default=None),
    mix: UploadFile | None = File(default=None),
    lyrics: str = Form(default=""),
    language: str = Form(default=""),
) -> StreamingResponse:
    """Word-level lyrics alignment via CTC forced alignment (MMS-300m).

    Streaming NDJSON response (one JSON object per line):

        {"type": "queued"}                       # only when the GPU is busy
        {"type": "running"}                       # GPU acquired, work started
        {"type": "result", "data": {"lines": [...]}}   # terminal success
        {"type": "error", "status_code": 500, "message": "..."}  # terminal

    The `queued` envelope lets a client that arrives while /transcribe (or
    another align) holds the GPU show a wait state instead of a silent
    hang; see `_serialized_gpu_stream`. Input-validation failures still
    return a real 4xx with a JSON body, we only switch to NDJSON once the
    GPU phase can actually start.

    Exactly one audio source must be supplied:

      - `vocals`: an already-isolated vocals stem (e.g. a paradb map
        that ships its own vocals track). The aligner runs straight on
        it.
      - `mix`: a full mix. The dedicated 2-stem vocals separator
        (see `Separator.run_vocals`) runs first to extract a vocals
        stem, then the aligner.

    `lyrics` is **required**: a JSON array of `{startSec, text}` lines
    (typically the parsed LRCLIB result). The endpoint is forced-
    alignment only; it never transcribes from audio. wav2vec2 aligns
    the caller's text against the audio to produce per-word timings.

    `language` is an optional ISO-639-1 hint that forces a specific
    wav2vec2 aligner. Empty string falls back to text-based heuristic
    detection and then to a 30 s audio-based detector.
    """
    _require_pipeline_role()
    # See `transcribe`: bind the request id before the first log line and
    # again at the top of `_stream_lyrics_align` (Starlette consumes the
    # streamed body generator in a separate context).
    request_id = new_request_id()
    set_request_id(request_id)
    aligner = get_aligner()

    sources_set = sum(1 for s in (vocals, mix) if s)
    if sources_set != 1:
        raise HTTPException(
            status_code=400,
            detail="Exactly one of vocals / mix must be supplied.",
        )
    if not lyrics:
        raise HTTPException(
            status_code=400,
            detail="`lyrics` is required (JSON array of {startSec, text}).",
        )
    input_lines = _parse_lyrics_input(lyrics)

    # `_require_pipeline_role()` above ensures we're on the worker
    # that loaded the separator at startup; the None branch is purely
    # defensive (e.g. eager load failed and we somehow still got here).
    separator: Separator = request.app.state.separator
    if separator is None:
        raise HTTPException(
            status_code=503,
            detail="Separator is not loaded on this worker.",
        )

    # File I/O and the disk-cache lookup don't touch the GPU, so we do
    # them here, up front: reading the upload needs `await`, and a
    # StreamingResponse is consumed once returned, so we drain it before
    # handing the temp dir to the generator (mirrors /transcribe). The
    # GPU steps (vocals separator + CTC aligner) run inside the streamed
    # generator under the process-wide lock.
    cleanup_dir = Path(tempfile.mkdtemp(prefix="drumjot_lyrics_"))
    try:
        needs_separator = False
        vocals_key: str | None = None
        mix_path: Path | None = None
        vocals_path: Path | None = None
        if vocals is not None:
            vocals_path = cleanup_dir / (vocals.filename or "vocals.wav")
            vocals_bytes = await vocals.read()
            vocals_path.write_bytes(vocals_bytes)
        else:
            assert mix is not None
            mix_path = cleanup_dir / (mix.filename or "input.wav")
            mix_bytes = await mix.read()
            mix_path.write_bytes(mix_bytes)
            audio_hash = _hash_bytes(mix_bytes)

            # Vocals-cache check: hit means we skip the separator and feed
            # the already-isolated opus straight to the CTC aligner (which
            # decodes it through its own ffmpeg pipeline, so no manual
            # decode here).
            vocals_key = _vocals_cache_key(audio_hash)
            cached_vocals = _vocals_cache_instance().get(vocals_key)
            if cached_vocals is not None:
                log.info("lyrics_align: vocals cache HIT (%s)", vocals_key)
                vocals_path = cached_vocals
            else:
                needs_separator = True
    except Exception:
        # The generator's `finally` only runs once it starts streaming;
        # a failure during the up-front drain has to clean up itself.
        shutil.rmtree(cleanup_dir, ignore_errors=True)
        raise

    return StreamingResponse(
        _stream_lyrics_align(
            request_id=request_id,
            separator=separator,
            aligner=aligner,
            input_lines=input_lines,
            language=language or None,
            needs_separator=needs_separator,
            vocals_key=vocals_key,
            mix_path=mix_path,
            vocals_path=vocals_path,
            cleanup_dir=cleanup_dir,
        ),
        media_type="application/x-ndjson",
    )


async def _stream_lyrics_align(
    *,
    request_id: str,
    separator: Separator,
    aligner: Any,
    input_lines: list[InputLine],
    language: str | None,
    needs_separator: bool,
    vocals_key: str | None,
    mix_path: Path | None,
    vocals_path: Path | None,
    cleanup_dir: Path,
) -> AsyncIterator[bytes]:
    """Stream the GPU phase of /lyrics/align as NDJSON bytes.

    The upload drain + vocals-cache lookup already ran in the endpoint
    handler; this owns the GPU-serialised work: park the drum models,
    (optionally) run the vocals separator, park it before the CTC aligner
    loads, then run forced alignment and emit the terminal `result`.

    Wrapped in `_serialized_gpu_stream` so a request that arrives while
    another GPU request is in flight emits a `queued` envelope first and
    then waits its turn (the GPU is a single resource; serialising also
    keeps a park from moving a model host-side under an in-flight
    forward pass). Failures surface as `error` envelopes rather than
    raising into the ASGI layer. The temp dir is always cleaned up.

    A `{"type": "heartbeat"}` line is interleaved every
    HEARTBEAT_INTERVAL_SECONDS while we're waiting between envelopes (the
    vocals separator + CTC aligner are long silent GPU stages) so an
    idle-timeout proxy between us and the client doesn't drop the
    connection. The frontend ignores unknown event types, so heartbeats
    need no client handling.
    """
    # Re-bind the request id: Starlette consumes this generator inside the
    # StreamingResponse task, a different context from the endpoint handler
    # that minted the id. Binding here guarantees the id is set in the
    # context that `asyncio.to_thread(...)` snapshots for the GPU work.
    set_request_id(request_id)

    async def job() -> AsyncIterator[dict[str, Any]]:
        nonlocal vocals_path
        try:
            # park_for_lyrics frees the drum-pipeline VRAM the /transcribe
            # path holds onto; park_vocals_after_extraction (below) then
            # frees the vocals separator's VRAM before the CTC aligner
            # allocates.
            try:
                gpu_park.park_for_lyrics(separator, aligner)
            except Exception:
                log.exception("lyrics_align: park_for_lyrics failed; continuing")

            if needs_separator:
                assert mix_path is not None
                assert vocals_key is not None
                raw_vocals = await asyncio.to_thread(
                    _extract_vocals_with_separator,
                    separator, mix_path, cleanup_dir,
                )
                if raw_vocals is None:
                    yield {
                        "type": "error",
                        "status_code": 500,
                        "message": "Separator ran but produced no vocals stem.",
                    }
                    return
                # Opus-encode into the cache. Whisperx reads opus through
                # ffmpeg natively, so the cached file IS the file we feed
                # to alignment; no double-encoding, no decode step.
                opus_tmp = cleanup_dir / "vocals.opus"
                try:
                    await asyncio.to_thread(
                        _encode_vocals_to_opus, raw_vocals, opus_tmp,
                    )
                    vocals_path = _vocals_cache_instance().put_path(
                        vocals_key, opus_tmp,
                    )
                    log.info(
                        "lyrics_align: vocals cache MISS, populated (%s)",
                        vocals_key,
                    )
                except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
                    # Cache-write failure must not break alignment. Fall
                    # back to the raw separator output for this request;
                    # the cache will retry on the next call.
                    log.warning(
                        "lyrics_align: vocals cache write failed (%s); "
                        "falling back to raw separator output",
                        exc,
                    )
                    vocals_path = raw_vocals

            # Park the vocals separator before the CTC aligner loads.
            # No-op when we took the cache hit / pre-supplied vocals
            # path (the separator was never loaded into VRAM this
            # request); important when we just ran it.
            try:
                gpu_park.park_vocals_after_extraction(separator)
            except Exception:
                log.exception(
                    "lyrics_align: park_vocals_after_extraction failed; continuing"
                )

            assert vocals_path is not None
            lines = await asyncio.to_thread(
                aligner.realign_text,
                vocals_path,
                input_lines,
                language,
            )
            yield {"type": "result", "data": {"lines": lines_to_json(lines)}}
        except FileNotFoundError as exc:
            yield {"type": "error", "status_code": 404, "message": str(exc)}
        except Exception as exc:
            log.exception("lyrics_align failed")
            yield {"type": "error", "status_code": 500, "message": str(exc)}

    pending: asyncio.Task[dict[str, Any]] | None = None
    try:
        # Drive the envelope stream by hand (rather than `async for`) so we
        # can race each `__anext__()` against a heartbeat timeout: the
        # separator/aligner stages can be silent for tens of seconds, and a
        # proxy with an idle timeout would otherwise drop the connection.
        # `wait_for` cancels its inner await on timeout, so we hold the
        # awaitable (a shielded Task) across iterations to avoid dropping an
        # envelope. A heartbeat is never emitted once the stream ends:
        # StopAsyncIteration breaks the loop before any further wait.
        envelopes = _serialized_gpu_stream(_gpu_lock, job).__aiter__()
        while True:
            if pending is None:
                pending = asyncio.ensure_future(envelopes.__anext__())
            try:
                envelope = await asyncio.wait_for(
                    asyncio.shield(pending),
                    timeout=HEARTBEAT_INTERVAL_SECONDS,
                )
            except TimeoutError:
                # No envelope yet, keep the connection warm and keep
                # waiting on `pending` (shield kept it alive through the
                # wait_for cancellation).
                yield (json.dumps({"type": "heartbeat"}) + "\n").encode("utf-8")
                continue
            except StopAsyncIteration:
                break
            pending = None
            yield (json.dumps(envelope) + "\n").encode("utf-8")
    finally:
        # If we're unwound mid-wait (client disconnect), cancel the
        # in-flight pull so it doesn't leak; the job's own `finally`
        # releases the GPU lock.
        if pending is not None and not pending.done():
            pending.cancel()
        shutil.rmtree(cleanup_dir, ignore_errors=True)


@app.post("/score", response_model=None)
async def score_chart(
    request: Request,
    pack: UploadFile | None = File(default=None),
    midi: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
) -> StreamingResponse:
    """Score a drum chart's onset timing against the real drum audio.

    A development test harness for the corpus-filtering scorer (see
    `app/scoring/`). Streaming NDJSON, same envelopes as /lyrics/align
    (`queued` / `running` / `heartbeat` / `result` / `error`); the terminal
    `result.data` is an `AlignmentResult`.

    Exactly one input form:
      - `pack`: a ParaDB `.zip` map pack. The best-difficulty chart is
        scored against the pack's drums-only track (no separation) or, if
        the pack has only a song track, the separated drum stem.
      - `midi` + `audio`: a MIDI chart plus the matching audio file. The
        audio is always separated to get the drum stem.
    """
    _require_pipeline_role()
    request_id = new_request_id()
    set_request_id(request_id)

    if pack is not None and (midi is not None or audio is not None):
        raise HTTPException(
            status_code=400, detail="Supply either `pack`, or `midi` + `audio`, not both."
        )
    if pack is None and not (midi is not None and audio is not None):
        raise HTTPException(
            status_code=400, detail="Supply a ParaDB `pack`, or both `midi` and `audio`."
        )

    separator: Separator = request.app.state.separator
    if separator is None:
        raise HTTPException(status_code=503, detail="Separator is not loaded on this worker.")

    # Drain uploads up front (needs await; the StreamingResponse body is
    # consumed once returned). The GPU work runs inside the streamed
    # generator under the process-wide lock, mirroring /lyrics/align.
    cleanup_dir = Path(tempfile.mkdtemp(prefix="drumjot_score_"))
    try:
        pack_bytes: bytes | None = None
        midi_bytes: bytes | None = None
        audio_path: Path | None = None
        if pack is not None:
            pack_bytes = await pack.read()
        else:
            assert midi is not None and audio is not None
            midi_bytes = await midi.read()
            audio_path = cleanup_dir / (audio.filename or "audio.wav")
            audio_path.write_bytes(await audio.read())
    except Exception:
        shutil.rmtree(cleanup_dir, ignore_errors=True)
        raise

    return StreamingResponse(
        _stream_score(
            request_id=request_id,
            separator=separator,
            pack_bytes=pack_bytes,
            midi_bytes=midi_bytes,
            audio_path=audio_path,
            cleanup_dir=cleanup_dir,
        ),
        media_type="application/x-ndjson",
    )


async def _stream_score(
    *,
    request_id: str,
    separator: Separator,
    pack_bytes: bytes | None,
    midi_bytes: bytes | None,
    audio_path: Path | None,
    cleanup_dir: Path,
) -> AsyncIterator[bytes]:
    """Stream the GPU phase of /score as NDJSON. Parks the lyrics-side
    models (this uses the drum models), runs the scorer on a worker thread,
    and emits the terminal `result` / `error`. Heartbeats keep the
    connection warm through the silent separation + ADTOF stages."""
    set_request_id(request_id)

    async def job() -> AsyncIterator[dict[str, Any]]:
        try:
            try:
                gpu_park.park_for_transcribe(separator, get_aligner())
            except Exception:
                log.exception("score: park_for_transcribe failed; continuing")

            if pack_bytes is not None:
                result = await asyncio.to_thread(
                    score_paradb, pack_bytes, work_dir=cleanup_dir, separator=separator
                )
            else:
                assert midi_bytes is not None and audio_path is not None
                result = await asyncio.to_thread(
                    score_midi,
                    midi_bytes,
                    audio_path=audio_path,
                    work_dir=cleanup_dir,
                    separator=separator,
                )
            yield {"type": "result", "data": result.model_dump()}
        except ValueError as exc:
            yield {"type": "error", "status_code": 400, "message": str(exc)}
        except FileNotFoundError as exc:
            yield {"type": "error", "status_code": 404, "message": str(exc)}
        except Exception as exc:
            log.exception("score failed")
            yield {"type": "error", "status_code": 500, "message": str(exc)}

    pending: asyncio.Task[dict[str, Any]] | None = None
    try:
        envelopes = _serialized_gpu_stream(_gpu_lock, job).__aiter__()
        while True:
            if pending is None:
                pending = asyncio.ensure_future(envelopes.__anext__())
            try:
                envelope = await asyncio.wait_for(
                    asyncio.shield(pending), timeout=HEARTBEAT_INTERVAL_SECONDS
                )
            except TimeoutError:
                yield (json.dumps({"type": "heartbeat"}) + "\n").encode("utf-8")
                continue
            except StopAsyncIteration:
                break
            pending = None
            yield (json.dumps(envelope) + "\n").encode("utf-8")
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        shutil.rmtree(cleanup_dir, ignore_errors=True)


def _parse_lyrics_input(raw: str) -> list[InputLine]:
    """Decode the `lyrics` form field into {@link InputLine}s.

    The frontend sends a JSON array of `{startSec, text}` matching its
    in-memory `LyricLine` shape (minus `words`, which we recompute).
    Validates shape eagerly so the caller gets a 400 with a specific
    message instead of a 500 from inside the worker thread later.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"`lyrics` is not valid JSON: {exc}",
        ) from exc
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=400,
            detail="`lyrics` must be a JSON array of {startSec, text} objects.",
        )
    out: list[InputLine] = []
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=400,
                detail=f"`lyrics[{i}]` must be an object with startSec + text.",
            )
        start = entry.get("startSec")
        text = entry.get("text")
        if not isinstance(start, (int, float)) or not isinstance(text, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"`lyrics[{i}]` requires numeric startSec and string text."
                ),
            )
        out.append(InputLine(start_sec=float(start), text=text))
    return out


def _extract_vocals_with_separator(
    separator: Separator, mix_path: Path, work_dir: Path,
) -> Path | None:
    """Run the dedicated vocals separator (fast 2-stem MDX-Net) on
    `mix_path` and return the vocals stem path for CTC forced alignment.
    Avoids the drum pipeline's 6-stem BS-Roformer SW pass; the aligner
    doesn't need that quality and the throughput cost was untenable.

    Returns None when the separator finished but no vocals-named output
    landed (model swap that no longer emits a `(Vocals)` token).
    """
    return separator.run_vocals(mix_path, work_dir)


# ---------------------------------------------------------------------------
# /lyrics/align vocals cache
# ---------------------------------------------------------------------------
#
# `settings.cache_dir/vocals/<sha256>__sep-<vocals_model_id>.opus`, # only the `mix` flow populates / reads it. Caching the separated
# vocals stem lets repeat alignments against the same mix skip the
# 5-10 s separation pass. The alignment result itself is not cached
# since each call's output depends on caller-provided text + language.

_KEY_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")

_vocals_cache: BlobCache | None = None
_cache_init_lock = threading.Lock()


def _vocals_cache_instance() -> BlobCache:
    """Lazy singleton for the vocals stem cache. Each worker process
    (pipeline + api) instantiates its own BlobCache against the shared
    on-disk directory."""
    global _vocals_cache
    if _vocals_cache is not None:
        return _vocals_cache
    with _cache_init_lock:
        if _vocals_cache is None:
            _vocals_cache = BlobCache(
                settings.cache_dir / "vocals",
                cap_bytes=settings.cache_vocals_cap_bytes,
            )
        return _vocals_cache


def _sanitize_id(s: str) -> str:
    """Strip filename-unsafe characters from a model id so it can ride
    in a cache filename. Anything outside `[A-Za-z0-9._-]` becomes `_`."""
    return _KEY_SAFE_CHARS.sub("_", s)


def _vocals_model_id() -> str:
    """Identifier for the vocals separator output. Burnt into the cache
    key so a model swap (e.g. switching `vocals_model`) auto-invalidates
    every cached vocals stem."""
    return _sanitize_id(settings.vocals_model)


def _vocals_cache_key(audio_hash: str) -> str:
    return f"{audio_hash}__sep-{_vocals_model_id()}.opus"


def _hash_bytes(data: bytes) -> str:
    """SHA-256 hex of `data`, matching `hashlib.sha256(bytes).hexdigest()`."""
    return hashlib.sha256(data).hexdigest()


def _encode_vocals_to_opus(src: Path, dest: Path) -> None:
    """ffmpeg-encode `src` to 16 kHz mono Opus at 24 kbps into `dest`.

    16 kHz mono matches what the CTC aligner's `load_audio` resamples to
    anyway, so doing the downmix + downsample at cache-write time shrinks the
    on-disk artifact ~50x vs FLAC with zero impact on alignment quality.
    `-application voip` biases libopus toward speech intelligibility at
    low bitrate. Raises CalledProcessError on encoder failure so the
    caller can decide whether to fall back to running uncached."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; required to populate the vocals cache."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-nostdin",
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "24k",
        "-application", "voip",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


async def _serialized_gpu_stream(
    lock: asyncio.Lock,
    job: Callable[[], AsyncIterator[dict[str, Any]]],
) -> AsyncIterator[dict[str, Any]]:
    """Serialise `job` behind the process-wide GPU `lock`, yielding NDJSON
    envelope dicts.

    Emits ``{"type": "queued"}`` first iff the lock is already held, so a
    client blocked behind another in-flight request can render a wait
    state instead of a silent hang. The queued envelope is yielded
    *before* awaiting acquisition, so it reaches the client immediately
    rather than after the wait. Once this stream owns the lock it emits
    ``{"type": "running"}`` and then forwards every envelope `job()`
    yields (typically a terminal ``result`` / ``error``).

    The lock is always released, even if `job` raises. `job` owns its own
    error handling and cleanup; it should yield a terminal envelope
    rather than raise, but a stray exception still unwinds cleanly here.

    This is the /lyrics/align analogue of the queued-event handling baked
    into `_stream_pipeline`; that streamer keeps its inline form because
    it also juggles a worker-thread queue, disconnect watching, and
    post-run assembly that don't generalise.
    """
    if lock.locked():
        yield {"type": "queued"}
    await lock.acquire()
    try:
        yield {"type": "running"}
        async for envelope in job():
            yield envelope
    finally:
        lock.release()


async def _stream_pipeline(
    *,
    request: Request,
    request_id: str,
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

    Serialised against /lyrics/align via `_gpu_lock`: a request that
    arrives while the other endpoint is mid-flight waits its turn so
    the park/unpark can't move a model to CPU under an in-flight
    forward pass. If the lock is contended we emit a `queued` event
    first so the streaming UI shows a wait state instead of a blank
    stream while we await acquisition.

    Re-binds the request id (the endpoint handler already set it, but
    Starlette consumes this body generator in a separate context). This
    binding is the one that matters: it runs in the context that
    `asyncio.to_thread(run_pipeline, ...)` snapshots via `copy_context()`,
    so the worker thread, and the LLM stages' own thread pools, which
    copy *their* submitting context, all log under the same id.

    During the long silent stages a `{"type": "heartbeat"}` line is
    emitted every HEARTBEAT_INTERVAL_SECONDS so an idle-timeout proxy
    between us and the client doesn't drop the connection.
    """
    # Bind in the generator's context (see docstring) before the to_thread
    # hop so the worker thread inherits the id.
    set_request_id(request_id)
    lock_held = False
    if _gpu_lock.locked():
        log.info("%s: GPU lock contended; queued behind another request", verb)
        yield (json.dumps({"type": "queued"}) + "\n").encode("utf-8")
    await _gpu_lock.acquire()
    lock_held = True
    try:
        gpu_park.park_for_transcribe(separator, get_aligner())
    except Exception:
        log.exception("%s: park_for_transcribe failed; continuing", verb)

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
                try:
                    envelope = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_INTERVAL_SECONDS
                    )
                except TimeoutError:
                    # No real progress event for HEARTBEAT_INTERVAL_SECONDS
                    # (a long silent stage is in flight). Emit a keepalive
                    # without consuming a real event, then keep draining.
                    # This can never fire after the terminal event: the
                    # terminal `result`/`error` line is yielded after this
                    # loop, and the `None` sentinel that ends the loop is
                    # enqueued in the same worker-thread `finally` as every
                    # terminal envelope, so once they're queued
                    # `queue.get()` returns them immediately rather than
                    # timing out.
                    yield (
                        json.dumps({"type": "heartbeat"}) + "\n"
                    ).encode("utf-8")
                    continue
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
        if lock_held:
            _gpu_lock.release()


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
