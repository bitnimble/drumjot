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
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import librosa

from app.debug import DebugSink, beats_dump, onsets_dump
from app.run_log import current_run_log
from app.models import (
    BestOfKLog,
    OnsetCandidate,
    RefinementIteration,
    RefinementLog,
)
from app.outputs import OutputSink
from app.pipeline.adtof_onsets import detect_onsets_adtof_or_librosa
from app.pipeline.beats import (
    BeatStructure,
    analyze_beats,
    detect_feel_for_bars,
)
from app.pipeline.cymbal_split import split_cymbal_onsets
from app.pipeline.hihat_split import split_hihat_onsets
from app.pipeline.filter_llm import filter_onsets_all_instruments
from app.pipeline.format import format_dsl
from app.pipeline.llm import transcribe_all_instruments
from app.pipeline.note_provenance import build_note_provenance
from app.pipeline.onsets import attach_beat_positions, detect_onsets
from app.pipeline.onsets_midi import onsets_to_midi_bytes
from app.pipeline.recompose import FEET_PITCHES, recompose
from app.pipeline.refine import (
    RefineLevel,
    refine_jot,
    refine_jot_per_instrument,
)
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
    # "dsl" = LLM emits Drumjot DSL, recompose + refine (legacy path).
    # "filter" = LLM only rejects artifact onsets; kept onsets render
    # straight to MIDI with original times. No Jot/recompose/refine.
    transcribe_mode: Literal["dsl", "filter"] = "dsl"
    # Per-request onset backend (the `onset_backend` form param).
    # "librosa" = the legacy spectral-flux detector; "adtof" = ADTOF
    # CRNN per stem with automatic fallback to librosa if ADTOF or its
    # weights are unavailable / the model errors.
    onset_backend: Literal["librosa", "adtof"] = "librosa"


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
    # Set only by the `filter` transcribe path: the rendered MIDI of the
    # LLM-kept onsets. Materialised to `prediction.mid` (debug + output
    # sink) and surfaced as `prediction_midi_url` on the response.
    predicted_midi: bytes | None = None
    # Set only by the `filter` transcribe path: per-note debug provenance
    # JSON listing every detected onset (kept and rejected) for the UI.
    # Shipped inside the debug bundle as `note_provenance.json`; absent
    # for DSL-mode runs (and absent in resume runs that didn't reach
    # `transcribe`). See `note_provenance.build_note_provenance`.
    note_provenance: dict[str, Any] | None = None
    # Per-instrument monophonic fragments produced by the transcribe
    # stage, kept so the refine stage can re-run the per-instrument loop
    # on fresh runs. Empty on a resume-from-refine (only the merged
    # initial.jot is on disk) — `_do_refine` falls back accordingly.
    initial_lines_by_pitch: dict[str, str] = field(default_factory=dict)
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
    output_sink: OutputSink | None = None,
) -> PipelineContext:
    """Run stages from `start_stage` onward, mutating and returning `ctx`."""
    if ctx.duration <= 0 and ctx.audio_path.exists():
        ctx.duration = _probe_duration(ctx.audio_path)

    start_idx = stage_index(start_stage)
    run_log = current_run_log()
    for stage in STAGE_ORDER[start_idx:]:
        log.info("Running stage: %s", stage.value)
        stage_start_wall = time.time()
        stage_start_perf = time.perf_counter()
        try:
            _run_stage(stage, ctx, separator, options, sink, output_sink)
        except StageError:
            raise
        except Exception as exc:
            log.exception("Stage %s failed", stage.value)
            raise StageError(stage, exc) from exc
        finally:
            elapsed = time.perf_counter() - stage_start_perf
            log.info("Stage %s finished in %.2fs", stage.value, elapsed)
            if run_log is not None:
                run_log.record_stage(stage.value, stage_start_wall, time.time())
    return ctx


def _run_stage(
    stage: Stage,
    ctx: PipelineContext,
    separator: Separator,
    options: PipelineOptions,
    sink: DebugSink | None,
    output_sink: OutputSink | None,
) -> None:
    if stage is Stage.STEMS_ALL:
        _do_stems_all(ctx, separator, output_sink)
    elif stage is Stage.STEMS_PER:
        _do_stems_per(ctx, separator, output_sink)
    elif stage is Stage.BEATS:
        _do_beats(ctx, options, sink)
    elif stage is Stage.ONSETS:
        _do_onsets(ctx, options, sink)
    elif stage is Stage.TRANSCRIBE:
        if options.transcribe_mode == "filter":
            _do_transcribe_filter(ctx, options, sink)
        else:
            _do_transcribe(ctx, options, sink)
    elif stage is Stage.REFINE:
        if options.transcribe_mode == "filter":
            # Filter mode has no Jot to refine, and the F1-vs-detected-
            # onsets gate is degenerate for a pure filter (see
            # transcriber/docs/filter-mode-proxy-reference.md). Skip.
            log.info("REFINE skipped (transcribe_mode=filter)")
            return
        _do_refine(ctx, options, sink)


def _do_stems_all(
    ctx: PipelineContext, separator: Separator, output_sink: OutputSink | None,
) -> None:
    if not ctx.audio_path.exists():
        raise RuntimeError(
            f"stems_all: input audio missing at {ctx.audio_path}"
        )
    result = separator.run_stems_all(ctx.audio_path, ctx.work_dir)
    ctx.drum_stem = result.drum_stem
    if output_sink is not None:
        output_sink.save_flac_from_wav("drum_stem", result.drum_stem)
        if result.no_drums is not None:
            output_sink.save_flac_from_wav("no_drums", result.no_drums)


def _do_stems_per(
    ctx: PipelineContext, separator: Separator, output_sink: OutputSink | None,
) -> None:
    if ctx.drum_stem is None or not ctx.drum_stem.exists():
        raise RuntimeError(
            "stems_per: drum stem missing (expected stems_all/drum_stem.<ext> "
            "from a previous run, or resume_stage<=stems_all to regenerate)."
        )
    ctx.per_instrument_stems = separator.run_stems_per(
        ctx.drum_stem, ctx.work_dir,
    )
    # Export the per-instrument stems as soon as splitting is done — they
    # are the second batch of deliverables, available long before the
    # (slow) beats/onsets/transcribe/refine stages run.
    if output_sink is not None:
        for pitch, path in ctx.per_instrument_stems.items():
            output_sink.save_flac_from_wav(f"stem_{pitch}", path)


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


def _do_onsets(
    ctx: PipelineContext, options: PipelineOptions, sink: DebugSink | None,
) -> None:
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
    if options.onset_backend == "adtof":
        # Noisy lanes (hihat/cymbal) detect off the in-distribution drum
        # stem inside detect_onsets_adtof; pass it through. None on a
        # resume that didn't cache it -> automatic per-stem fallback.
        raw_onsets = {
            pitch: detect_onsets_adtof_or_librosa(
                path, pitch, drum_stem_path=ctx.drum_stem
            )
            for pitch, path in ctx.per_instrument_stems.items()
        }
    else:
        raw_onsets = {
            pitch: detect_onsets(path)
            for pitch, path in ctx.per_instrument_stems.items()
        }
    ctx.onsets_by_pitch = {
        pitch: attach_beat_positions(cands, ctx.structure)
        for pitch, cands in raw_onsets.items()
    }
    # The Stage-2 separator merges ride + crash into one `cymbals` stem
    # (pitch `c`); split that lane into ride (`d`) / crash (`c`) here so
    # the downstream per-instrument pathways see two real lanes. No-op
    # when there is no cymbals stem / onsets.
    ctx.onsets_by_pitch = split_cymbal_onsets(
        ctx.onsets_by_pitch, ctx.per_instrument_stems, ctx.structure,
    )
    # The hi-hat stem mixes closed and open hi-hat hits; classify each
    # onset and split into closed (`h`) and synthetic open (`H`) lanes so
    # the per-instrument transcribe pass handles them independently. No-op
    # when there is no hi-hat stem / onsets. See `hihat_split` docs for
    # the `H` synthetic-pitch caveat (folded back into `h:o` is a TODO).
    ctx.onsets_by_pitch = split_hihat_onsets(
        ctx.onsets_by_pitch, ctx.per_instrument_stems, ctx.structure,
    )
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
    # One LLM call per instrument (parallel), each emitting a monophonic
    # line; recompose merges them deterministically into one Jot.
    lines_by_pitch, scores_by_pitch = transcribe_all_instruments(
        candidates_by_pitch=ctx.onsets_by_pitch,
        structure=ctx.structure,
        samples=options.best_of_k,
    )
    if not lines_by_pitch:
        raise RuntimeError(
            "transcribe: every per-instrument LLM call failed or produced "
            "no usable line."
        )
    ctx.initial_lines_by_pitch = lines_by_pitch
    jot_dsl = format_dsl(recompose(lines_by_pitch, ctx.structure, FEET_PITCHES))
    ctx.initial_jot = jot_dsl

    if options.best_of_k > 1 and scores_by_pitch:
        per_instrument: dict[str, BestOfKLog] = {}
        for pitch, scores in scores_by_pitch.items():
            chosen = (
                int(max(range(len(scores)), key=lambda i: scores[i]))
                if scores
                else 0
            )
            per_instrument[pitch] = BestOfKLog(
                samples=options.best_of_k,
                scores=scores,
                chosen_index=chosen,
            )
        ctx.best_of_k_log = BestOfKLog(
            samples=options.best_of_k,
            scores=[],
            chosen_index=0,
            per_instrument=per_instrument,
        )

    if sink is not None:
        for pitch, fragment in lines_by_pitch.items():
            sink.write_text(f"initial_{pitch}.jot", fragment)
        sink.write_text("initial.jot", jot_dsl)
        if ctx.best_of_k_log is not None:
            sink.write_json("best_of_k.json", ctx.best_of_k_log.model_dump())


def _do_transcribe_filter(
    ctx: PipelineContext,
    options: PipelineOptions,
    sink: DebugSink | None,
) -> None:
    """Filter-mode transcribe: LLM rejects artifact onsets per
    instrument; the kept onsets render straight to MIDI with their
    original (un-quantized) times. No Jot, no recompose, no refine.
    """
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

    kept_by_pitch = filter_onsets_all_instruments(
        ctx.onsets_by_pitch, ctx.structure,
    )
    if not kept_by_pitch:
        raise RuntimeError(
            "transcribe: filter kept no onsets for any instrument."
        )

    midi_bytes = onsets_to_midi_bytes(
        kept_by_pitch,
        initial_tempo_bpm=ctx.structure.initial_tempo,
        structure=ctx.structure,
    )
    ctx.predicted_midi = midi_bytes
    # Per-note debug provenance covering every detected onset (kept and
    # rejected) — shipped in the debug bundle for the UI to surface in
    # the selection label + render rejected onsets as ghost overlays.
    # Built from the pre-filter candidate set so the rejected branch is
    # actually populated; identity is by `id(c)` against `kept_by_pitch`.
    ctx.note_provenance = build_note_provenance(
        all_onsets_by_pitch=ctx.onsets_by_pitch,
        kept_by_pitch=kept_by_pitch,
        structure=ctx.structure,
        onset_backend=options.onset_backend,
    )
    # Filter mode produces no Jot. Set empty strings so the shared
    # response builder (which requires a non-None final_jot) and the
    # outputs backfill stay happy without special-casing every caller.
    ctx.initial_jot = ""
    ctx.final_jot = ""

    if sink is not None:
        sink.write_bytes("prediction.mid", midi_bytes)
        sink.write_json(
            "filter/kept_onsets.json",
            {
                pitch: [
                    {
                        "time": c.time,
                        "strength": c.strength,
                        "bar": c.bar,
                        "beat_in_bar": c.beat_in_bar,
                    }
                    for c in cands
                ]
                for pitch, cands in kept_by_pitch.items()
            },
        )
        sink.write_json("note_provenance.json", ctx.note_provenance)


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

    # STRUCTURE (cross-instrument pattern factoring) is intentionally
    # dropped: per-instrument transcription emits flat monophonic lines
    # with no shared bar patterns to factor, so the level has nothing to
    # act on. ONSETS/VELOCITY run per instrument; LINT/MACRO run once on
    # the recomposed Jot (see refine_jot_per_instrument).
    refinement_levels: list[RefineLevel] = []
    if options.lint:
        refinement_levels.append(RefineLevel.LINT)
    if options.refine:
        refinement_levels.extend([
            RefineLevel.MACRO,
            RefineLevel.ONSETS,
            RefineLevel.VELOCITY,
        ])

    if not refinement_levels:
        # No levels requested; final == initial. Still emit final.jot
        # so downstream tooling has a stable artifact name to read.
        ctx.final_jot = format_dsl(
            _inject_start_offset(ctx.initial_jot, ctx.structure)
        )
        if sink is not None:
            sink.write_text("final.jot", ctx.final_jot)
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
        if ctx.initial_lines_by_pitch:
            # Fresh run: per-instrument ONSETS/VELOCITY loop, then
            # recompose, then LINT/MACRO on the merged Jot.
            refined_dsl, log_obj = refine_jot_per_instrument(
                lines_by_pitch=ctx.initial_lines_by_pitch,
                stem_onsets=ctx.onsets_by_pitch,
                stem_audios=stem_audios,
                structure=ctx.structure,
                levels=refinement_levels,
                feet_pitches=FEET_PITCHES,
            )
        else:
            # Resume-from-refine: only the merged initial.jot is on
            # disk; refine it directly (no per-instrument fragments).
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

    # Stamp the lead-in (audio time of the first detected beat) onto the
    # DSL after refinement so the LLM can't accidentally drop it during
    # revision. Browser playback reads this off `globalMetadata` to delay
    # its schedule and match the original recording's offset.
    ctx.final_jot = format_dsl(
        _inject_start_offset(ctx.final_jot, ctx.structure)
    )

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


def _inject_start_offset(dsl: str, structure: BeatStructure) -> str:
    """Prepend a `{{ startOffset: X }}` block to `dsl`.

    `X` is the audio time (seconds) of the first detected beat in the
    source recording — i.e. how much silence / non-drum intro preceded
    bar 0 of the transcription. The TS player honours this on
    `globalMetadata` to delay browser playback so the rendered drums
    hit at the same wall-clock offset as in the original.

    A separate top-level block is prepended (rather than merged into
    the LLM-emitted block) so we don't have to parse and rewrite the
    LLM's metadata; the TS parser merges successive `{{...}}` blocks
    into one `globalMetadata` dict so the order doesn't matter.

    No-op when no beats were detected or the offset is non-positive
    (≤ ~0 means the song starts at audio time 0, no lead-in worth
    encoding).
    """
    if not structure.bars or not structure.bars[0].beats:
        return dsl
    offset = float(structure.bars[0].start_time)
    if offset <= 1e-3:
        return dsl
    return f"{{{{ startOffset: {offset:.3f} }}}}\n{dsl}"
