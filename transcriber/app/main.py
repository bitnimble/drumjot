"""Drumjot transcriber HTTP API (FastAPI).

Endpoints:
    GET  /health        - readiness + GPU info
    POST /transcribe    - accept an audio file, return Drumjot DSL

    The service is intentionally stateless. All temp files live in per-request
    tempdirs. Models are loaded eagerly at container startup (FastAPI
    lifespan) so the first /transcribe call doesn't pay model-load latency
    and so orchestrators can use /health as a true readiness probe.
"""
from __future__ import annotations

import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import librosa
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.debug import DebugSink, beats_dump, onsets_dump
from app.models import (
    BarSummary,
    BestOfKLog,
    HealthResponse,
    RefinementIteration,
    RefinementLog,
    TranscribeMetadata,
    TranscribeResponse,
)
from app.pipeline.beats import (
    analyze_beats,
    detect_feel_for_bars,
    summarize_bar_for_prompt,
)
from app.pipeline.llm import (
    transcribe_to_jot,
    transcribe_to_jot_best_of_k,
)
from app.pipeline.onsets import attach_beat_positions, detect_onsets
from app.pipeline.refine import RefineLevel, refine_jot
from app.pipeline.separate import Separator
from app.pipeline.title import inject_title, title_from_filename

# Fallback debug dir used when `debug=true` is requested but `DEBUG_DIR`
# env var wasn't set. Matches the docker-compose volume mount.
DEFAULT_DEBUG_DIR = Path("/debug")

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
    best_of_k: int = Form(
        default=settings.best_of_k_default
    ),
    debug: bool = Form(default=False),
) -> TranscribeResponse:
    started = time.perf_counter()
    log.info(
        "Transcribe request: %s (%s bytes) refine=%s lint=%s best_of_k=%d debug=%s",
        file.filename, file.size, refine, lint, best_of_k, debug,
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
    sink = DebugSink.for_request(debug_base, file.filename)

    with tempfile.TemporaryDirectory(prefix="drumjot_") as tmp_str:
        work = Path(tmp_str)
        in_path = work / (file.filename or "input.wav")
        in_path.write_bytes(await file.read())

        if sink is not None:
            sink.copy_audio("input", in_path)

        # 1) Stem separation
        sep: Separator = request.app.state.separator
        try:
            stems = sep.separate(in_path, work)
        except Exception as exc:
            log.exception("Separation failed")
            raise HTTPException(
                status_code=500,
                detail=f"Stem separation failed: {exc}",
            ) from exc

        if sink is not None:
            sink.copy_audio("stage1/drum_stem", stems.drum_stem)
            for pitch, path in stems.per_instrument.items():
                sink.copy_audio(f"stage2/{pitch}", path)

        # Compute song duration up front so beat tracking can pad
        # synthetic bars forward to cover the full audio (otherwise
        # onsets that fire after the last detected beat get jammed onto
        # the last real bar with out-of-range beat_in_bar values).
        try:
            import soundfile as sf

            with sf.SoundFile(str(in_path)) as f:
                duration = len(f) / f.samplerate
        except Exception:
            duration = 0.0

        # 2) Beat / downbeat / time-sig tracking over the FULL mix. We
        #    run on the original audio (not the drum stem) so beat
        #    tracking has the full spectral context to work with.
        try:
            structure = analyze_beats(
                in_path, duration_seconds=duration if duration > 0 else None
            )
        except Exception as exc:
            log.exception("Beat tracking failed")
            raise HTTPException(
                status_code=500,
                detail=f"Beat tracking failed: {exc}",
            ) from exc

        # 3) Onset detection per stem (high recall)
        raw_onsets = {
            pitch: detect_onsets(path)
            for pitch, path in stems.per_instrument.items()
        }
        # Attach (bar, beat_in_bar) positions using the beat structure.
        onsets_by_pitch = {
            pitch: attach_beat_positions(cands, structure)
            for pitch, cands in raw_onsets.items()
        }

        # 4) Per-bar feel detection using the flat onset stream.
        flat_times = [c.time for cs in onsets_by_pitch.values() for c in cs]
        detect_feel_for_bars(structure, flat_times)

        if sink is not None:
            sink.write_json("beats.json", beats_dump(structure))
            sink.write_json("onsets.json", onsets_dump(onsets_by_pitch))

        # Fallback duration if the soundfile probe above failed - use the
        # latest detected onset across all stems as a lower bound for the
        # metadata payload.
        if duration <= 0:
            duration = max(
                (c.time for cs in onsets_by_pitch.values() for c in cs),
                default=0.0,
            )

        # 5) Initial transcription. Best-of-K sampling if requested.
        best_of_k_log: BestOfKLog | None = None
        try:
            if best_of_k > 1:
                jot_dsl, scores = transcribe_to_jot_best_of_k(
                    candidates_by_pitch=onsets_by_pitch,
                    structure=structure,
                    samples=best_of_k,
                )
                chosen_idx = (
                    int(max(range(len(scores)), key=lambda i: scores[i]))
                    if scores
                    else 0
                )
                best_of_k_log = BestOfKLog(
                    samples=best_of_k,
                    scores=scores,
                    chosen_index=chosen_idx,
                )
            else:
                jot_dsl = transcribe_to_jot(
                    candidates_by_pitch=onsets_by_pitch,
                    structure=structure,
                )
        except Exception as exc:
            log.exception("LLM transcription failed")
            raise HTTPException(
                status_code=502,
                detail=f"LLM transcription failed: {exc}",
            ) from exc

        initial_dsl = jot_dsl
        if sink is not None:
            sink.write_text("initial.jot", initial_dsl)
            if best_of_k_log is not None:
                sink.write_json(
                    "best_of_k.json", best_of_k_log.model_dump()
                )

        # 6) Optional refinement (multi-level convergence loop).
        # `lint` and `refine` are independent toggles: lint enables the
        # deterministic instrument/performance fix-up pass; refine enables
        # the F1-gated tempo / structure / onset / velocity loop. The
        # canonical level order is enforced inside refine_jot, so we just
        # build a set here.
        refinement_levels: list[RefineLevel] = []
        if lint:
            refinement_levels.append(RefineLevel.LINT)
        if refine:
            refinement_levels.extend([
                RefineLevel.MACRO,
                RefineLevel.STRUCTURE,
                RefineLevel.ONSETS,
                RefineLevel.VELOCITY,
            ])

        refinement_log: RefinementLog | None = None
        if refinement_levels:
            try:
                stem_audios: dict[str, tuple[Any, int]] = {}
                for pitch, path in stems.per_instrument.items():
                    audio, sr = librosa.load(str(path), sr=44100, mono=True)
                    stem_audios[pitch] = (audio, sr)

                refined_dsl, log_obj = refine_jot(
                    initial_dsl=jot_dsl,
                    stem_onsets=onsets_by_pitch,
                    stem_audios=stem_audios,
                    structure=structure,
                    levels=refinement_levels,
                )
                jot_dsl = refined_dsl
                refinement_log = RefinementLog(
                    initial_score=log_obj.initial_score,
                    final_score=log_obj.final_score,
                    elapsed_seconds=log_obj.elapsed_seconds,
                    iterations=[
                        RefinementIteration(**vars(it)) for it in log_obj.iterations
                    ],
                )
            except Exception:
                log.exception("Refinement failed; returning unrefined Jot")

        # Inject the title deterministically from the uploaded filename.
        # The few-shot examples don't include `title:` because LLM-invented
        # titles occasionally tripped Anthropic's output content filter on
        # benign drum audio.
        jot_dsl = inject_title(jot_dsl, title_from_filename(file.filename))

        if sink is not None:
            sink.write_text("final.jot", jot_dsl)
            if refinement_log is not None:
                sink.write_json("refinement.json", refinement_log.model_dump())

    elapsed = time.perf_counter() - started
    log.info("Transcribe complete in %.2fs (refined=%s)", elapsed, refine)

    bar_summaries = [
        BarSummary(**summarize_bar_for_prompt(b)) for b in structure.bars
    ]

    if sink is not None:
        sink.finalize({
            "filename": file.filename,
            "options": {
                "refine": refine,
                "lint": lint,
                "best_of_k": best_of_k,
                "include_candidates": include_candidates,
                "debug": debug,
            },
            "scores": {
                "initial": refinement_log.initial_score if refinement_log else None,
                "final": refinement_log.final_score if refinement_log else None,
            },
            "stems_used": sorted(stems.per_instrument.keys()),
            "duration_seconds": duration,
        })

    return TranscribeResponse(
        jot_dsl=jot_dsl,
        metadata=TranscribeMetadata(
            initial_tempo=structure.initial_tempo,
            initial_time_signature=list(structure.initial_time_signature),
            duration_seconds=duration,
            stems_used=sorted(stems.per_instrument.keys()),
            bars=bar_summaries,
            has_tempo_changes=structure.has_tempo_changes,
            has_time_sig_changes=structure.has_time_sig_changes,
        ),
        refinement=refinement_log,
        best_of_k=best_of_k_log,
        candidates=onsets_by_pitch if include_candidates else {},
        debug_dir=str(sink.dir) if sink is not None else None,
    )
