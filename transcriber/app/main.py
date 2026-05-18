"""Drumjot transcriber HTTP API (FastAPI).

Endpoints:
    GET  /health             - readiness + GPU info
    POST /transcribe         - accept an audio file, return Drumjot DSL
    POST /transcribe/resume  - re-run from a chosen pipeline stage using
                               a previous debug folder's intermediate
                               artifacts

The service is intentionally stateless. All temp files live in per-request
tempdirs. Models are loaded eagerly at container startup (FastAPI
lifespan) so the first /transcribe call doesn't pay model-load latency
and so orchestrators can use /health as a true readiness probe.
"""
from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.debug import (
    DebugSink,
    mint_request_folder_name,
    reset_current_debug_sink,
    set_current_debug_sink,
)
from app.models import (
    BarSummary,
    HealthResponse,
    TranscribeMetadata,
    TranscribeResponse,
)
from app.outputs import OutputSink, make_output_sink
from app.pipeline.beats import summarize_bar_for_prompt
from app.pipeline.resume import (
    find_input_audio,
    hydrate_context_from_resume,
    load_original_filename,
)
from app.pipeline.runner import (
    BeatInput,
    PipelineContext,
    PipelineOptions,
    Stage,
    StageError,
    run_pipeline,
)
from app.pipeline.separate import Separator

# Fallback debug dir used when `debug=true` is requested but `DEBUG_DIR`
# env var wasn't set. Matches the docker-compose volume mount.
DEFAULT_DEBUG_DIR = Path("/debug")

# Map stage -> HTTP status code surfaced when that stage fails. The
# transcribe stage is the only external dependency (Anthropic), so it
# gets 502 (bad gateway); everything else is local compute and surfaces
# as 500.
_STAGE_HTTP_STATUS: dict[Stage, int] = {
    Stage.STEMS_ALL: 500,
    Stage.STEMS_PER: 500,
    Stage.BEATS: 500,
    Stage.ONSETS: 500,
    Stage.TRANSCRIBE: 502,
    Stage.REFINE: 500,
}


@contextmanager
def _scoped_debug_sink(sink: DebugSink | None) -> Iterator[None]:
    """Install `sink` as the request-scoped current sink for the duration
    of the `with` block, then restore the previous value.

    Deep callees (the LLM wrapper, refinement helpers) read from a
    ContextVar so they can dump their hydrated prompts to the same
    request folder without having the sink threaded through their
    signatures.
    """
    token = set_current_debug_sink(sink)
    try:
        yield
    finally:
        reset_current_debug_sink(token)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly warm the separation models so the first /transcribe call
    # doesn't pay model-load latency. The model load is blocking I/O +
    # GPU memory allocation, so we run it on a worker thread to avoid
    # blocking the event loop while uvicorn negotiates startup.
    import asyncio
    import time

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


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    include_candidates: bool = Form(default=False),
    refine: bool = Form(default=settings.refine_by_default),
    lint: bool = Form(default=settings.lint_by_default),
    best_of_k: int = Form(default=settings.best_of_k_default),
    beat_input: BeatInput = Form(default=settings.beat_input_default),
    debug: bool = Form(default=False),
) -> TranscribeResponse:
    started = time.perf_counter()
    log.info(
        "Transcribe request: %s (%s bytes) refine=%s lint=%s best_of_k=%d "
        "beat_input=%s debug=%s",
        file.filename, file.size, refine, lint, best_of_k, beat_input, debug,
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

    with tempfile.TemporaryDirectory(prefix="drumjot_") as tmp_str, _scoped_debug_sink(sink):
        work = Path(tmp_str)
        in_path = work / (file.filename or "input.wav")
        in_path.write_bytes(await file.read())
        if sink is not None:
            sink.copy_audio("input", in_path)

        ctx = PipelineContext(audio_path=in_path, work_dir=work)
        options = PipelineOptions(
            refine=refine, lint=lint, best_of_k=best_of_k, beat_input=beat_input,
        )
        sep: Separator = request.app.state.separator
        try:
            run_pipeline(
                ctx=ctx,
                start_stage=Stage.STEMS_ALL,
                separator=sep,
                options=options,
                sink=sink,
                output_sink=output_sink,
            )
        except StageError as exc:
            raise HTTPException(
                status_code=_STAGE_HTTP_STATUS[exc.stage],
                detail=str(exc),
            ) from exc

        if sink is not None:
            sink.finalize({
                "filename": file.filename,
                "options": {
                    "refine": refine,
                    "lint": lint,
                    "best_of_k": best_of_k,
                    "beat_input": beat_input,
                    "include_candidates": include_candidates,
                    "debug": debug,
                },
                "scores": {
                    "initial": (
                        ctx.refinement_log.initial_score
                        if ctx.refinement_log else None
                    ),
                    "final": (
                        ctx.refinement_log.final_score
                        if ctx.refinement_log else None
                    ),
                },
                "stems_used": sorted(ctx.per_instrument_stems.keys()),
                "duration_seconds": ctx.duration,
            })

    elapsed = time.perf_counter() - started
    log.info("Transcribe complete in %.2fs (refined=%s)", elapsed, refine)

    return _build_response(
        ctx=ctx,
        include_candidates=include_candidates,
        debug_dir=str(sink.dir) if sink is not None else None,
        output_sink=output_sink,
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


@app.post("/transcribe/resume", response_model=TranscribeResponse)
async def transcribe_resume(
    request: Request,
    resume_folder: str = Form(...),
    resume_stage: Stage = Form(...),
    include_candidates: bool = Form(default=False),
    refine: bool = Form(default=settings.refine_by_default),
    lint: bool = Form(default=settings.lint_by_default),
    best_of_k: int = Form(default=settings.best_of_k_default),
    beat_input: BeatInput = Form(default=settings.beat_input_default),
) -> TranscribeResponse:
    """Re-run the pipeline from `resume_stage` onward, hydrating any
    artifacts produced by earlier stages from `resume_folder`.

    `resume_stage` is one of:
        `stems_all`, `stems_per`, `beats`, `onsets`, `transcribe`, `refine`.

    Required artifacts depend on which stages will be skipped:
        - `stems_per`: `stems_all/drum_stem.<ext>`
        - `beats`:     `stems_per/*.<ext>`
        - `onsets`:    `stems_per/*.<ext>` + `beats.json`
        - `transcribe`: `beats.json` + `onsets.json` + `stems_per/*.<ext>`
        - `refine`:    `initial.jot` + the four above
    Anything missing comes back as a 400 with a stage-specific message.

    Stages from `resume_stage` onward run fresh and overwrite whichever
    of `initial.jot`, `final.jot`, `best_of_k.json`, `refinement.json`,
    `beats.json`, `onsets.json` they would normally produce. Upstream
    artifacts are left intact so subsequent resumes from this folder
    remain idempotent.
    """
    started = time.perf_counter()
    resume_dir = _resolve_resume_dir(resume_folder)
    log.info(
        "Resume request from %s (resume_stage=%s refine=%s lint=%s best_of_k=%d)",
        resume_dir, resume_stage.value, refine, lint, best_of_k,
    )

    audio_path = find_input_audio(resume_dir) or (resume_dir / "input")
    sink = DebugSink(resume_dir)
    # Reuse the resume folder's basename as the outputs folder name so
    # any FLACs the original /transcribe run wrote are still surfaced in
    # the resumed response (and a resume-from-stems_all overwrites them
    # in place rather than orphaning the originals).
    output_sink = make_output_sink(resume_dir.name, settings.outputs_dir)
    options = PipelineOptions(
        refine=refine, lint=lint, best_of_k=best_of_k, beat_input=beat_input,
    )
    sep: Separator = request.app.state.separator

    # Run stem separation (if it fires) into a temp work_dir so its
    # filenames don't collide with the resume folder; the sink will
    # mirror them into `<resume_dir>/stems_all/` and `/stems_per/`.
    with (
        tempfile.TemporaryDirectory(prefix="drumjot_resume_") as tmp_str,
        _scoped_debug_sink(sink),
    ):
        work = Path(tmp_str)
        ctx = PipelineContext(audio_path=audio_path, work_dir=work)
        try:
            hydrate_context_from_resume(ctx, resume_dir, resume_stage)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            run_pipeline(
                ctx=ctx,
                start_stage=resume_stage,
                separator=sep,
                options=options,
                sink=sink,
                output_sink=output_sink,
            )
        except StageError as exc:
            raise HTTPException(
                status_code=_STAGE_HTTP_STATUS[exc.stage],
                detail=str(exc),
            ) from exc

        sink.finalize({
            "filename": load_original_filename(resume_dir),
            "options": {
                "refine": refine,
                "lint": lint,
                "best_of_k": best_of_k,
                "beat_input": beat_input,
                "include_candidates": include_candidates,
                "resume_folder": str(resume_dir),
                "resume_stage": resume_stage.value,
            },
            "scores": {
                "initial": (
                    ctx.refinement_log.initial_score
                    if ctx.refinement_log else None
                ),
                "final": (
                    ctx.refinement_log.final_score
                    if ctx.refinement_log else None
                ),
            },
            "stems_used": sorted(ctx.per_instrument_stems.keys()),
            "duration_seconds": ctx.duration,
        })

    elapsed = time.perf_counter() - started
    log.info("Resume complete in %.2fs (refined=%s)", elapsed, refine)

    return _build_response(
        ctx=ctx,
        include_candidates=include_candidates,
        debug_dir=str(resume_dir),
        output_sink=output_sink,
    )


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
    if ctx.final_jot is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed but produced no final jot.",
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
    if output_sink is not None:
        if output_sink.existing_path("drum_stem") is not None:
            drum_stem_url = output_sink.url_for("drum_stem")
        if output_sink.existing_path("no_drums") is not None:
            no_drums_url = output_sink.url_for("no_drums")
    return TranscribeResponse(
        jot_dsl=ctx.final_jot,
        metadata=TranscribeMetadata(
            initial_tempo=ctx.structure.initial_tempo,
            initial_time_signature=list(ctx.structure.initial_time_signature),
            duration_seconds=ctx.duration,
            stems_used=stems_used,
            bars=bar_summaries,
            has_tempo_changes=ctx.structure.has_tempo_changes,
            has_time_sig_changes=ctx.structure.has_time_sig_changes,
        ),
        refinement=ctx.refinement_log,
        best_of_k=ctx.best_of_k_log,
        candidates=ctx.onsets_by_pitch if include_candidates else {},
        debug_dir=debug_dir,
        drum_stem_url=drum_stem_url,
        no_drums_url=no_drums_url,
    )
