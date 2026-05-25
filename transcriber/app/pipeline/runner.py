"""Pipeline runner: dispatches the five named stages in order.

Both `/transcribe` (full run from a fresh upload) and `/transcribe/resume`
(skip to a chosen stage using a previous debug folder) go through this
single function. The caller hydrates a `PipelineContext` with whatever
upstream artifacts they already have, picks a `start_stage`, and the
runner walks `STAGE_ORDER[start_stage:]` mutating the context.

Stage dependencies (output of stage -> consumers):

    stems_all  -> drum_stem               (consumed by stems_per)
    stems_per  -> per_instrument_stems    (consumed by onsets)
    beats      -> structure               (consumed by onsets, filter, transcribe)
    onsets     -> onsets_by_pitch         (consumed by filter, transcribe)
    filter     -> kept_by_pitch           (consumed by quantise, transcribe)
    quantise   -> kept_by_pitch (mutated) (consumed by transcribe)
    transcribe -> predicted_midi          (kept-onsets MIDI deliverable)

The split between `filter` (parallel per-instrument LLM calls that
reject artifact onsets) and `transcribe` (the deterministic
`onsets_to_midi_bytes` + `build_note_provenance` render) lets the
operator iterate on the MIDI / provenance code without paying for LLM
calls; resume from `transcribe` re-hydrates `kept_by_pitch` from
`filter/kept_onsets.json` (and `quantise/shifts.json` if the prior run
had quantise enabled) and re-runs only the render. Re-running the
filter LLM itself means resuming from `filter`.

`quantise` is enabled by default and can be disabled per-request via
the `quantise` form param. It snaps kept onsets to a 1/48 grid
deterministically and then runs a Haiku LLM pass for jitter-class
shifts in cross-instrument context. See `pipeline/quantise.py`.

The legacy `dsl`/`refine` pathway (LLM-emitted Drumjot DSL + F1-gated
refinement loop) was removed in May 2026; see
`transcriber/docs/ai-midi-to-jot-notes.md` for the techniques captured
from it for any future AI-assisted MIDI -> Jot work. The backend now
produces MIDI only; converting that MIDI to a Drumjot Jot lives on the
frontend via `src/midi/from_midi.ts`.

If a stage's required upstream artifact is missing, the corresponding
`_do_*` function raises with a message naming which stage should have
produced it; the HTTP layer wraps that into a 400.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from app.debug import DebugSink, beats_dump, onsets_dump
from app.models import OnsetCandidate
from app.outputs import OutputSink
from app.pipeline.adtof_onsets import (
    detect_drum_onsets_for_alignment,
    detect_onsets_adtof,
)
from app.pipeline.beats import (
    BeatStructure,
    analyze_beats,
    detect_feel_for_bars,
)
from app.pipeline.cymbal_split import split_cymbal_onsets
from app.pipeline.filter_llm import filter_onsets_all_instruments
from app.pipeline.hihat_split import split_hihat_onsets
from app.pipeline.note_provenance import build_note_provenance
from app.pipeline.onsets_midi import onsets_to_midi_bytes
from app.pipeline.quantise import quantise_kept_onsets
from app.pipeline.separate import Separator
from app.run_log import current_run_log

log = logging.getLogger(__name__)


class Stage(StrEnum):
    """Named pipeline stages, ordered by data dependency."""

    STEMS_ALL = "stems_all"
    STEMS_PER = "stems_per"
    BEATS = "beats"
    ONSETS = "onsets"
    FILTER = "filter"
    QUANTISE = "quantise"
    TRANSCRIBE = "transcribe"


STAGE_ORDER: list[Stage] = [
    Stage.STEMS_ALL,
    Stage.STEMS_PER,
    Stage.BEATS,
    Stage.ONSETS,
    Stage.FILTER,
    Stage.QUANTISE,
    Stage.TRANSCRIBE,
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


class PipelineCancelled(Exception):
    """Raised between stages when the client has disconnected (the HTTP
    layer set `ctx.cancel_event`). The current stage runs to completion —
    native code in separation/beats/onsets is uninterruptible — but
    subsequent stages are skipped.
    """

    def __init__(self, next_stage: Stage) -> None:
        self.next_stage = next_stage
        super().__init__(f"cancelled before stage {next_stage.value}")


BeatInput = Literal["full_mix", "drum_stem"]


# Progress event payload published by the pipeline as stages start/end
# and (optionally) as substage milestones tick over (e.g. "3/5
# instruments filtered"). The HTTP layer wraps this in `{"type": ...}`
# NDJSON envelopes for the streaming response; here we keep it as a
# plain dict so the runner doesn't pull in HTTP types.
#
# Fields:
#   stage         the StrEnum value of the current stage
#   phase         "start" | "end" — emitted at the bookends of each stage
#   detail        free-form human-readable substage label
#                 (e.g. "filtering snare", "iteration 2 / 3")
#   elapsed_seconds  set only on "end" events
ProgressEvent = dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]


def _safe_progress(progress: ProgressCallback | None, event: ProgressEvent) -> None:
    """Invoke `progress` while swallowing exceptions.

    Progress is a debugging / UX aid — a misbehaving callback (or a
    closed client connection on the HTTP side) must never propagate
    into the pipeline and abort the actual transcribe.
    """
    if progress is None:
        return
    try:
        progress(event)
    except Exception:  # pragma: no cover - progress is best-effort
        log.exception("progress callback raised; ignoring")


@dataclass
class PipelineOptions:
    beat_input: BeatInput = "full_mix"
    # Whether to run the optional `quantise` stage. Enabled by default;
    # set False to skip both the deterministic joint-snap pass and the
    # LLM residual pass, leaving `kept_by_pitch` as the filter stage
    # produced it (raw seconds; frontend's 1/48 snap in
    # `src/midi/from_midi.ts` handles quantisation).
    quantise: bool = True


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
    # Diagnostic audio track: `drum_stem − sum(per_instrument_stems)`.
    # Captures auxiliary percussion (cowbell, tambourine, etc.) plus the
    # separator's reconstruction error. Surfaced in the debug bundle only;
    # no downstream stage consumes it.
    residual_stem: Path | None = None
    structure: BeatStructure | None = None
    onsets_by_pitch: dict[str, list[OnsetCandidate]] = field(default_factory=dict)
    # Hi-hat onsets the unified ternary classifier (or its open-tail
    # backstop) rejected as artifacts. Kept off `onsets_by_pitch` so the
    # rest of the pipeline sees a clean split into closed (`h`) / open
    # (`H`); the runner merges them back into the provenance input at
    # the `note_provenance` boundary so the UI's "Show filtered" overlay
    # can surface them as ghosts. Empty when resuming from `transcribe`
    #; discards live only in memory and `hihat_split/decision.json`,
    # not in `onsets.json`. See `pipeline/hihat_split.py`.
    hihat_discarded: list[OnsetCandidate] = field(default_factory=list)
    # Cymbal onsets the unified ternary classifier rejected as
    # artifacts. Same shape and lifecycle as `hihat_discarded`: kept off
    # `onsets_by_pitch` (which holds the clean ride / crash split) and
    # merged into `all_onsets_by_pitch[c]` at the provenance boundary
    # only. See `pipeline/cymbal_split.py`.
    cymbal_discarded: list[OnsetCandidate] = field(default_factory=list)
    # Output of the `filter` stage (parallel per-instrument LLM calls
    # rejecting artifact onsets). The keys here are the subset of
    # `onsets_by_pitch`'s keys for which at least one onset survived
    # the filter; the `OnsetCandidate` instances are the SAME objects
    # carried through from `onsets_by_pitch` so that `build_note_provenance`
    # can match kept-vs-rejected by `id(c)`. Persisted to
    # `filter/kept_onsets.json` for resume; `pipeline/resume.py` re-
    # threads the identity against `onsets_by_pitch` on hydration.
    kept_by_pitch: dict[str, list[OnsetCandidate]] = field(default_factory=dict)
    # The rendered MIDI of the LLM-kept onsets. Materialised to
    # `prediction.mid` (debug + output sink) and surfaced as
    # `prediction_midi_url` on the response.
    predicted_midi: bytes | None = None
    # Per-note debug provenance JSON listing every detected onset (kept
    # and rejected) for the UI. Shipped inside the debug bundle as
    # `note_provenance.json`. See `note_provenance.build_note_provenance`.
    note_provenance: dict[str, Any] | None = None
    # Set by the HTTP layer when the client disconnects (Stop button).
    # Checked between stages and inside the LLM stage's parallel pool;
    # the current stage cannot be interrupted (native code) but the
    # pipeline stops advancing once it returns. See PipelineCancelled.
    cancel_event: threading.Event = field(default_factory=threading.Event)


def run_pipeline(
    *,
    ctx: PipelineContext,
    start_stage: Stage,
    separator: Separator,
    options: PipelineOptions,
    sink: DebugSink | None,
    output_sink: OutputSink | None = None,
    progress: ProgressCallback | None = None,
) -> PipelineContext:
    """Run stages from `start_stage` onward, mutating and returning `ctx`.

    When `progress` is provided it receives one event at each stage
    bookend (`phase="start"` / `phase="end"`) plus any substage events
    emitted by `_do_<stage>` helpers (e.g. per-instrument LLM progress
    inside `transcribe`). Callback exceptions are caught and logged so a
    misbehaving sink can never abort the actual pipeline.
    """
    if ctx.duration <= 0 and ctx.audio_path.exists():
        ctx.duration = _probe_duration(ctx.audio_path)

    start_idx = stage_index(start_stage)
    run_log = current_run_log()
    for stage in STAGE_ORDER[start_idx:]:
        # Client-disconnect / Stop-button cancellation check. Fires between
        # stages only — native code in the current stage cannot be
        # interrupted, but skipping the remaining stages saves the rest
        # of the pipeline (most importantly the LLM stage).
        if ctx.cancel_event.is_set():
            log.info("Pipeline cancelled before stage %s", stage.value)
            raise PipelineCancelled(stage)
        log.info("Running stage: %s", stage.value)
        _safe_progress(progress, {"stage": stage.value, "phase": "start"})
        stage_start_wall = time.time()
        stage_start_perf = time.perf_counter()
        try:
            _run_stage(stage, ctx, separator, options, sink, output_sink, progress)
        except StageError:
            raise
        except PipelineCancelled:
            raise
        except Exception as exc:
            log.exception("Stage %s failed", stage.value)
            raise StageError(stage, exc) from exc
        finally:
            elapsed = time.perf_counter() - stage_start_perf
            log.info("Stage %s finished in %.2fs", stage.value, elapsed)
            if run_log is not None:
                run_log.record_stage(stage.value, stage_start_wall, time.time())
            _safe_progress(
                progress,
                {"stage": stage.value, "phase": "end", "elapsed_seconds": elapsed},
            )
    return ctx


def _run_stage(
    stage: Stage,
    ctx: PipelineContext,
    separator: Separator,
    options: PipelineOptions,
    sink: DebugSink | None,
    output_sink: OutputSink | None,
    progress: ProgressCallback | None,
) -> None:
    if stage is Stage.STEMS_ALL:
        _do_stems_all(ctx, separator, output_sink)
    elif stage is Stage.STEMS_PER:
        _do_stems_per(ctx, separator, output_sink)
    elif stage is Stage.BEATS:
        _do_beats(ctx, options, sink)
    elif stage is Stage.ONSETS:
        _do_onsets(ctx, sink)
    elif stage is Stage.FILTER:
        _do_filter(ctx, sink, progress)
    elif stage is Stage.QUANTISE:
        _do_quantise(ctx, options, sink)
    elif stage is Stage.TRANSCRIBE:
        _do_transcribe(ctx, sink)
    # Re-check cancel after the (possibly long-running) stage finishes so
    # PipelineCancelled fires at the next stage boundary without one
    # final "phase=end" event sneaking in for a stage that the user has
    # already cancelled.


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
    result = separator.run_stems_per(ctx.drum_stem, ctx.work_dir)
    ctx.per_instrument_stems = result.per_instrument
    ctx.residual_stem = result.residual
    # Export the per-instrument stems as soon as splitting is done — they
    # are the second batch of deliverables, available long before the
    # (slow) beats/onsets/transcribe stages run.
    if output_sink is not None:
        for pitch, path in ctx.per_instrument_stems.items():
            output_sink.save_flac_from_wav(f"stem_{pitch}", path)
        if ctx.residual_stem is not None:
            output_sink.save_flac_from_wav("residual", ctx.residual_stem)


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

    # Snap the whole beat grid to the strongest drum transients within
    # ±50 ms. Neural trackers report beat times at the activation peak,
    # which lags the transient by ~50 ms; without this, downbeat kicks
    # register at beat_in_bar=1.18 instead of 1.00 and the LLM either
    # drops them or quantizes them to the wrong slot. We deliberately
    # reuse the ADTOF backend on the full drum stem — its CRNN gives the
    # most stable transient peaks for the grid-alignment median, and
    # keeping the alignment detector independent of the per-stem onsets
    # the next stage runs avoids any circular dependency between them.
    #
    # Pool onsets across all five drum lanes (kick / snare / toms /
    # hi-hat / cymbal) rather than just kick. The alignment median's
    # coverage gate rejected songs whose kicks weren't on every beat;
    # adding snare/hat/cymbal multiplies the per-beat hit rate without
    # biasing the offset — `align_beats_to_onsets` picks the strongest
    # nearby onset, so a louder kick still wins when one is around.
    align_onsets: list[tuple[float, float]] | None = None
    if ctx.drum_stem is not None and ctx.drum_stem.exists():
        try:
            align_onsets = detect_drum_onsets_for_alignment(ctx.drum_stem)
        except Exception as exc:
            log.warning(
                "beats: ADTOF onset detection on drum stem failed (%s); "
                "skipping grid alignment.", exc,
            )
            align_onsets = []

    ctx.structure = analyze_beats(
        beat_audio,
        duration_seconds=duration,
        align_onsets=align_onsets,
    )
    if sink is not None:
        sink.write_json("beats.json", beats_dump(ctx.structure))


def _do_onsets(
    ctx: PipelineContext, sink: DebugSink | None,
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
    # ADTOF runs the noisy lanes (hihat / merged cymbal) on the
    # in-distribution drum stem; pass it through. None on a resume that
    # didn't cache it — `detect_onsets_adtof` falls back to the isolated
    # stem when the drum stem is absent.
    raw_onsets = {
        pitch: detect_onsets_adtof(
            path, pitch, drum_stem_path=ctx.drum_stem
        )
        for pitch, path in ctx.per_instrument_stems.items()
    }
    ctx.onsets_by_pitch = {
        pitch: _attach_beat_positions(cands, ctx.structure)
        for pitch, cands in raw_onsets.items()
    }
    # The Stage-2 separator merges ride + crash into one `cymbals` stem
    # (pitch `c`); split that lane into ride (`d`) / crash (`c`) AND a
    # discard set (sizzle re-triggers in long crash tails, bleed,
    # double-triggers) rejected by the unified ternary classifier
    # upstream of the filter LLM. No-op when there is no cymbals stem /
    # onsets. The filter LLM is skipped for `c` / `d` in `_do_transcribe`
    # below for the same reason as the hi-hat lanes.
    ctx.onsets_by_pitch, ctx.cymbal_discarded = split_cymbal_onsets(
        ctx.onsets_by_pitch, ctx.per_instrument_stems, ctx.structure,
    )
    # The hi-hat stem mixes closed and open hi-hat hits; classify each
    # onset and split into closed (`h`) and synthetic open (`H`) lanes
    # AND a discard set (sizzle re-triggers, bleed, double-triggers)
    # rejected by the unified ternary classifier upstream of the filter
    # LLM. No-op when there is no hi-hat stem / onsets. See `hihat_split`
    # docs for the `H` synthetic-pitch caveat (folded back into `h:o` is
    # a TODO) and for why the filter LLM is skipped for `h` / `H` in
    # `_do_transcribe` below.
    ctx.onsets_by_pitch, ctx.hihat_discarded = split_hihat_onsets(
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


def _attach_beat_positions(
    candidates: list[OnsetCandidate],
    structure: BeatStructure,
) -> list[OnsetCandidate]:
    """Annotate each candidate with `(bar, beat_in_bar)` using `structure`.

    Candidates whose timestamps fall outside the tracked range are kept
    with `bar=-1, beat_in_bar=-1.0` and downstream code should treat
    them as "out of song" / drop.
    """
    out: list[OnsetCandidate] = []
    for c in candidates:
        pos = structure.position(c.time)
        if pos is None:
            out.append(
                OnsetCandidate(time=c.time, strength=c.strength, bar=-1, beat_in_bar=-1.0)
            )
            continue
        bar, beat = pos
        out.append(
            OnsetCandidate(
                time=c.time, strength=c.strength, bar=int(bar), beat_in_bar=float(beat)
            )
        )
    return out


def _do_filter(
    ctx: PipelineContext,
    sink: DebugSink | None,
    progress: ProgressCallback | None = None,
) -> None:
    """Per-instrument LLM filter that rejects artifact onsets.

    Runs one Anthropic call per drum pitch in parallel, populating
    `ctx.kept_by_pitch` with the surviving onsets. Pre-vetted lanes
    (`h` / `H` / `c` / `d`, already filtered upstream by their unified
    ternary classifiers) skip the LLM and pass through verbatim.

    Persists `filter/kept_onsets.json` so resuming from `transcribe`
    can re-hydrate `kept_by_pitch` without paying for the LLM calls
    again.
    """
    if ctx.structure is None:
        raise RuntimeError(
            "filter: beat structure missing (expected beats.json or "
            "resume_stage<=beats)."
        )
    if not ctx.onsets_by_pitch:
        raise RuntimeError(
            "filter: onsets missing (expected onsets.json or "
            "resume_stage<=onsets)."
        )

    # The progress callback receives one substage update per *completed*
    # instrument so the UI can show "filtering N/M instruments" without
    # flickering between parallel in-flight pitches.
    def on_instrument_done(pitch: str, done: int, total: int) -> None:
        _safe_progress(
            progress,
            {
                "stage": Stage.FILTER.value,
                "detail": f"filtering {done}/{total} instruments (latest: {pitch})",
            },
        )

    # `h` / `H` / `c` / `d` are filtered upstream by their unified
    # ternary classifiers (`hihat_split` and `cymbal_split`). Re-running
    # the per-instrument filter LLM here would duplicate work and risk
    # double-rejecting soft real hits, so we skip those pitches in the
    # pool and re-attach the pre-vetted lanes verbatim afterwards.
    kept_by_pitch = filter_onsets_all_instruments(
        ctx.onsets_by_pitch,
        ctx.structure,
        on_complete=on_instrument_done,
        cancel_event=ctx.cancel_event,
        skip_pitches={"h", "H", "c", "d"},
    )
    if ctx.cancel_event.is_set():
        # The pool exited early because the client disconnected. Surface
        # this so the runner stops at the next stage boundary check (the
        # filter pass may have partial results but persisting and
        # advancing into `transcribe` would be wasted work).
        raise PipelineCancelled(Stage.FILTER)
    for p in ("h", "H", "c", "d"):
        vetted = ctx.onsets_by_pitch.get(p)
        if vetted:
            in_range = [c for c in vetted if c.bar >= 0]
            if in_range:
                kept_by_pitch[p] = in_range
    if not kept_by_pitch:
        raise RuntimeError(
            "filter: kept no onsets for any instrument."
        )

    ctx.kept_by_pitch = kept_by_pitch

    if sink is not None:
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


def _do_quantise(
    ctx: PipelineContext,
    options: PipelineOptions,
    sink: DebugSink | None,
) -> None:
    """Snap kept onsets to a 1/48 grid: deterministic joint snap +
    optional LLM residual pass.

    Mutates `ctx.kept_by_pitch` candidates in place, each shifted onset
    gets `quantised_time` / `quantised_shift_slots` populated, leaving
    `time` / `beat_in_bar` as the raw detector hit for provenance.
    Persists `quantise/shifts.json` for resume + inspection. Skipped
    entirely when `options.quantise` is False (no-op; downstream sees
    raw filter output).
    """
    if not options.quantise:
        log.info("quantise: skipped (options.quantise=False)")
        return
    if ctx.structure is None:
        raise RuntimeError(
            "quantise: beat structure missing (expected beats.json or "
            "resume_stage<=beats)."
        )
    if not ctx.kept_by_pitch:
        raise RuntimeError(
            "quantise: kept_by_pitch missing (expected "
            "filter/kept_onsets.json or resume_stage<=filter)."
        )

    summary = quantise_kept_onsets(
        ctx.kept_by_pitch,
        ctx.structure,
        use_llm=True,
        cancel_event=ctx.cancel_event,
    )
    log.info(
        "quantise: deterministic shifted %d, LLM shifted %d (llm_status=%s)",
        summary.get("deterministic_shifted", 0),
        summary.get("llm_shifted", 0),
        summary.get("llm_status", "?"),
    )
    if sink is not None:
        sink.write_json("quantise/shifts.json", summary)


def _do_transcribe(
    ctx: PipelineContext,
    sink: DebugSink | None,
) -> None:
    """Deterministic MIDI render + per-note provenance.

    Consumes `ctx.kept_by_pitch` (from the `filter` stage) and writes
    `prediction.mid` + `note_provenance.json`. No LLM calls; resuming
    from this stage costs nothing externally.
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
    if not ctx.kept_by_pitch:
        raise RuntimeError(
            "transcribe: kept_by_pitch missing (expected "
            "filter/kept_onsets.json or resume_stage<=filter)."
        )

    midi_bytes = onsets_to_midi_bytes(
        ctx.kept_by_pitch,
        initial_tempo_bpm=ctx.structure.initial_tempo,
        structure=ctx.structure,
    )
    ctx.predicted_midi = midi_bytes
    # Per-note debug provenance covering every detected onset (kept and
    # rejected); shipped in the debug bundle for the UI to surface in
    # the selection label + render rejected onsets as ghost overlays.
    # Built from the pre-filter candidate set so the rejected branch is
    # actually populated; identity is by `id(c)` against `kept_by_pitch`.
    # Hi-hat and cymbal discards from their unified ternary classifiers
    # are spliced back into their source pitches (`h` and `c`
    # respectively; those were the merged input lanes) *only at this
    # boundary* so they appear in `all_onsets_by_pitch` (= rendered as
    # ghosts) without ever entering `kept_by_pitch` (= surviving into
    # MIDI). Identity-by-`id(c)` naturally keeps them out of the kept
    # set.
    all_onsets_for_provenance = ctx.onsets_by_pitch
    if ctx.hihat_discarded or ctx.cymbal_discarded:
        all_onsets_for_provenance = dict(ctx.onsets_by_pitch)
        if ctx.hihat_discarded:
            all_onsets_for_provenance["h"] = sorted(
                list(ctx.onsets_by_pitch.get("h", [])) + ctx.hihat_discarded,
                key=lambda c: (c.bar, c.beat_in_bar),
            )
        if ctx.cymbal_discarded:
            all_onsets_for_provenance["c"] = sorted(
                list(ctx.onsets_by_pitch.get("c", [])) + ctx.cymbal_discarded,
                key=lambda c: (c.bar, c.beat_in_bar),
            )
    ctx.note_provenance = build_note_provenance(
        all_onsets_by_pitch=all_onsets_for_provenance,
        kept_by_pitch=ctx.kept_by_pitch,
        structure=ctx.structure,
        beat_alignment_offset_sec=ctx.structure.align_offset_sec,
        rejected_by_pitch={
            "h": "hihat_split",
            "H": "hihat_split",
            "c": "cymbal_split",
            "d": "cymbal_split",
        },
    )

    if sink is not None:
        sink.write_bytes("prediction.mid", midi_bytes)
        sink.write_json("note_provenance.json", ctx.note_provenance)


def _probe_duration(audio_path: Path) -> float:
    try:
        import soundfile as sf

        with sf.SoundFile(str(audio_path)) as f:
            return float(len(f) / f.samplerate)
    except Exception:
        return 0.0
