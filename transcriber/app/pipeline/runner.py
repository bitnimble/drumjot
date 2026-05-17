"""Pipeline runner: dispatches the six named stages in order.

Both `/transcribe` (full run from a fresh upload) and `/transcribe/resume`
(skip to a chosen stage using a previous debug folder) go through this
single function. The caller hydrates a `PipelineContext` with whatever
upstream artifacts they already have, picks a `start_stage`, and the
runner walks `STAGE_ORDER[start_stage:]` mutating the context.

Stage dependencies (output of stage -> consumers):

    stems_all  -> drum_stem               (consumed by stems_per)
    stems_per  -> per_instrument_stems    (consumed by onsets, refine)
    beats      -> structure               (consumed by onsets, transcribe, refine)
    onsets     -> onsets_by_pitch         (consumed by transcribe, refine)
    transcribe -> initial_jot             (consumed by refine)
    refine     -> final_jot

If a stage's required upstream artifact is missing, the corresponding
`_do_*` function raises with a message naming which stage should have
produced it; the HTTP layer wraps that into a 400.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import librosa

from app.debug import DebugSink, beats_dump, onsets_dump
from app.models import (
    BestOfKLog,
    OnsetCandidate,
    RefinementIteration,
    RefinementLog,
)
from app.pipeline.beats import (
    BeatStructure,
    analyze_beats,
    detect_feel_for_bars,
)
from app.pipeline.llm import (
    transcribe_to_jot,
    transcribe_to_jot_best_of_k,
)
from app.pipeline.onsets import attach_beat_positions, detect_onsets
from app.pipeline.onsets_midi import onsets_to_midi_bytes
from app.pipeline.refine import RefineLevel, refine_jot
from app.pipeline.separate import Separator

log = logging.getLogger(__name__)


class Stage(StrEnum):
    """Named pipeline stages, ordered by data dependency."""

    STEMS_ALL = "stems_all"
    STEMS_PER = "stems_per"
    BEATS = "beats"
    ONSETS = "onsets"
    TRANSCRIBE = "transcribe"
    REFINE = "refine"


STAGE_ORDER: list[Stage] = [
    Stage.STEMS_ALL,
    Stage.STEMS_PER,
    Stage.BEATS,
    Stage.ONSETS,
    Stage.TRANSCRIBE,
    Stage.REFINE,
]


def stage_index(s: Stage) -> int:
    return STAGE_ORDER.index(s)


class StageError(Exception):
    """Raised when a pipeline stage fails. Carries the stage that failed so
    the HTTP layer can map it to a meaningful status code (e.g. transcribe
    → 502 because it's an external LLM call, vs other stages → 500).
    """

    def __init__(self, stage: Stage, original: Exception) -> None:
        self.stage = stage
        self.original = original
        super().__init__(f"{stage.value} failed: {original}")


BeatInput = Literal["full_mix", "drum_stem"]


@dataclass
class PipelineOptions:
    refine: bool
    lint: bool
    best_of_k: int
    beat_input: BeatInput = "full_mix"


@dataclass
class PipelineContext:
    """Mutable bag of artifacts that flows through the pipeline.

    Each stage reads the fields produced by its upstream dependencies and
    writes its own. Resume hydration pre-populates fields whose stage
    will be skipped; fresh runs leave them at their defaults and let
    each stage fill them in.
    """

    audio_path: Path
    work_dir: Path
    duration: float = 0.0
    drum_stem: Path | None = None
    per_instrument_stems: dict[str, Path] = field(default_factory=dict)
    structure: BeatStructure | None = None
    onsets_by_pitch: dict[str, list[OnsetCandidate]] = field(default_factory=dict)
    initial_jot: str | None = None
    best_of_k_log: BestOfKLog | None = None
    final_jot: str | None = None
    refinement_log: RefinementLog | None = None


def run_pipeline(
    *,
    ctx: PipelineContext,
    start_stage: Stage,
    separator: Separator,
    options: PipelineOptions,
    sink: DebugSink | None,
) -> PipelineContext:
    """Run stages from `start_stage` onward, mutating and returning `ctx`."""
    if ctx.duration <= 0 and ctx.audio_path.exists():
        ctx.duration = _probe_duration(ctx.audio_path)

    start_idx = stage_index(start_stage)
    for stage in STAGE_ORDER[start_idx:]:
        log.info("Running stage: %s", stage.value)
        try:
            _run_stage(stage, ctx, separator, options, sink)
        except StageError:
            raise
        except Exception as exc:
            log.exception("Stage %s failed", stage.value)
            raise StageError(stage, exc) from exc
    return ctx


def _run_stage(
    stage: Stage,
    ctx: PipelineContext,
    separator: Separator,
    options: PipelineOptions,
    sink: DebugSink | None,
) -> None:
    if stage is Stage.STEMS_ALL:
        _do_stems_all(ctx, separator)
    elif stage is Stage.STEMS_PER:
        _do_stems_per(ctx, separator)
    elif stage is Stage.BEATS:
        _do_beats(ctx, options, sink)
    elif stage is Stage.ONSETS:
        _do_onsets(ctx, sink)
    elif stage is Stage.TRANSCRIBE:
        _do_transcribe(ctx, options, sink)
    elif stage is Stage.REFINE:
        _do_refine(ctx, options, sink)


def _do_stems_all(ctx: PipelineContext, separator: Separator) -> None:
    if not ctx.audio_path.exists():
        raise RuntimeError(
            f"stems_all: input audio missing at {ctx.audio_path}"
        )
    ctx.drum_stem = separator.run_stems_all(ctx.audio_path, ctx.work_dir)


def _do_stems_per(ctx: PipelineContext, separator: Separator) -> None:
    if ctx.drum_stem is None or not ctx.drum_stem.exists():
        raise RuntimeError(
            "stems_per: drum stem missing (expected stems_all/drum_stem.<ext> "
            "from a previous run, or resume_stage<=stems_all to regenerate)."
        )
    ctx.per_instrument_stems = separator.run_stems_per(
        ctx.drum_stem, ctx.work_dir,
    )


def _do_beats(
    ctx: PipelineContext, options: PipelineOptions, sink: DebugSink | None
) -> None:
    if options.beat_input == "drum_stem":
        if ctx.drum_stem is None or not ctx.drum_stem.exists():
            raise RuntimeError(
                "beats: beat_input='drum_stem' requested but drum stem missing "
                "(expected stems_all/drum_stem.<ext> from a previous run, or "
                "resume_stage<=stems_all to regenerate)."
            )
        beat_audio = ctx.drum_stem
    else:
        if not ctx.audio_path.exists():
            raise RuntimeError(
                f"beats: input audio missing at {ctx.audio_path}"
            )
        beat_audio = ctx.audio_path
    log.info("Beat tracking using audio: %s (mode=%s)", beat_audio.name, options.beat_input)
    duration = ctx.duration if ctx.duration > 0 else None

    # Snap each detected beat to the strongest drum onset within ±50 ms.
    # Neural trackers report beat times at the activation peak, which lags
    # the transient by ~50 ms; without this, downbeat kicks register at
    # beat_in_bar=1.18 instead of 1.00 and the LLM either drops them or
    # quantizes them to the wrong slot.
    align_onsets: list[tuple[float, float]] | None = None
    if ctx.drum_stem is not None and ctx.drum_stem.exists():
        drum_onsets = detect_onsets(ctx.drum_stem)
        align_onsets = [(c.time, c.strength) for c in drum_onsets]

    ctx.structure = analyze_beats(
        beat_audio,
        duration_seconds=duration,
        align_onsets=align_onsets,
    )
    if sink is not None:
        sink.write_json("beats.json", beats_dump(ctx.structure))


def _do_onsets(ctx: PipelineContext, sink: DebugSink | None) -> None:
    if not ctx.per_instrument_stems:
        raise RuntimeError(
            "onsets: per-instrument stems missing (expected stems_per/*.<ext> "
            "from a previous run, or resume_stage<=stems_per to regenerate)."
        )
    if ctx.structure is None:
        raise RuntimeError(
            "onsets: beat structure missing (expected beats.json from a "
            "previous run, or resume_stage<=beats to regenerate)."
        )
    raw_onsets = {
        pitch: detect_onsets(path)
        for pitch, path in ctx.per_instrument_stems.items()
    }
    ctx.onsets_by_pitch = {
        pitch: attach_beat_positions(cands, ctx.structure)
        for pitch, cands in raw_onsets.items()
    }
    flat_times = [c.time for cs in ctx.onsets_by_pitch.values() for c in cs]
    detect_feel_for_bars(ctx.structure, flat_times)
    # Fallback duration probe (when the soundfile probe in run_pipeline
    # failed because audio_path didn't exist at that point).
    if ctx.duration <= 0:
        ctx.duration = max(
            (c.time for cs in ctx.onsets_by_pitch.values() for c in cs),
            default=0.0,
        )
    if sink is not None:
        sink.write_json("onsets.json", onsets_dump(ctx.onsets_by_pitch))
        # Re-write beats.json now that detect_feel_for_bars has mutated
        # the structure in-place with per-bar feel labels.
        sink.write_json("beats.json", beats_dump(ctx.structure))
        # Auto-emit a "what the detector heard" MIDI file. One note per
        # raw onset, no LLM filtering — purely a diagnostic so the
        # operator can ear-check whether onsets.json is reasonable
        # without booting a JSON viewer + audio editor side-by-side.
        try:
            midi_bytes = onsets_to_midi_bytes(
                ctx.onsets_by_pitch,
                initial_tempo_bpm=ctx.structure.initial_tempo,
            )
            sink.write_bytes("onsets_only.mid", midi_bytes)
        except Exception as exc:
            log.warning("onsets -> MIDI render failed (%s); skipping", exc)


def _do_transcribe(
    ctx: PipelineContext,
    options: PipelineOptions,
    sink: DebugSink | None,
) -> None:
    if ctx.structure is None:
        raise RuntimeError(
            "transcribe: beat structure missing (expected beats.json or "
            "resume_stage<=beats)."
        )
    if not ctx.onsets_by_pitch:
        raise RuntimeError(
            "transcribe: onsets missing (expected onsets.json or "
            "resume_stage<=onsets)."
        )
    if options.best_of_k > 1:
        jot_dsl, scores = transcribe_to_jot_best_of_k(
            candidates_by_pitch=ctx.onsets_by_pitch,
            structure=ctx.structure,
            samples=options.best_of_k,
        )
        chosen_idx = (
            int(max(range(len(scores)), key=lambda i: scores[i]))
            if scores
            else 0
        )
        ctx.best_of_k_log = BestOfKLog(
            samples=options.best_of_k,
            scores=scores,
            chosen_index=chosen_idx,
        )
    else:
        jot_dsl = transcribe_to_jot(
            candidates_by_pitch=ctx.onsets_by_pitch,
            structure=ctx.structure,
        )
    ctx.initial_jot = jot_dsl
    if sink is not None:
        sink.write_text("initial.jot", jot_dsl)
        if ctx.best_of_k_log is not None:
            sink.write_json("best_of_k.json", ctx.best_of_k_log.model_dump())


def _do_refine(
    ctx: PipelineContext,
    options: PipelineOptions,
    sink: DebugSink | None,
) -> None:
    if ctx.initial_jot is None:
        raise RuntimeError(
            "refine: initial.jot missing (expected initial.jot from a "
            "previous run, or resume_stage<=transcribe to regenerate)."
        )
    if ctx.structure is None:
        raise RuntimeError("refine: beat structure missing")

    refinement_levels: list[RefineLevel] = []
    if options.lint:
        refinement_levels.append(RefineLevel.LINT)
    if options.refine:
        refinement_levels.extend([
            RefineLevel.MACRO,
            RefineLevel.STRUCTURE,
            RefineLevel.ONSETS,
            RefineLevel.VELOCITY,
        ])

    if not refinement_levels:
        # No levels requested; final == initial. Still emit final.jot
        # so downstream tooling has a stable artifact name to read.
        ctx.final_jot = ctx.initial_jot
        if sink is not None:
            sink.write_text("final.jot", ctx.initial_jot)
        return

    if not ctx.per_instrument_stems:
        raise RuntimeError(
            "refine: per-instrument stems missing (refinement re-loads them "
            "via librosa for score sampling)."
        )

    stem_audios: dict[str, tuple[Any, int]] = {}
    for pitch, path in ctx.per_instrument_stems.items():
        audio, sr = librosa.load(str(path), sr=44100, mono=True)
        stem_audios[pitch] = (audio, sr)

    try:
        refined_dsl, log_obj = refine_jot(
            initial_dsl=ctx.initial_jot,
            stem_onsets=ctx.onsets_by_pitch,
            stem_audios=stem_audios,
            structure=ctx.structure,
            levels=refinement_levels,
        )
        ctx.final_jot = refined_dsl
        ctx.refinement_log = RefinementLog(
            initial_score=log_obj.initial_score,
            final_score=log_obj.final_score,
            elapsed_seconds=log_obj.elapsed_seconds,
            iterations=[
                RefinementIteration(**vars(it)) for it in log_obj.iterations
            ],
        )
    except Exception:
        log.exception("Refinement failed; returning unrefined Jot")
        ctx.final_jot = ctx.initial_jot

    if sink is not None:
        sink.write_text("final.jot", ctx.final_jot)
        if ctx.refinement_log is not None:
            sink.write_json(
                "refinement.json", ctx.refinement_log.model_dump(),
            )


def _probe_duration(audio_path: Path) -> float:
    try:
        import soundfile as sf

        with sf.SoundFile(str(audio_path)) as f:
            return float(len(f) / f.samplerate)
    except Exception:
        return 0.0
