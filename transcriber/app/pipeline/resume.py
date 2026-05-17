"""Hydrate a `PipelineContext` from a previous /transcribe debug folder.

The runner is stage-driven: skipping a stage means whatever artifact that
stage would have produced has to come from somewhere else. For
`/transcribe/resume`, that "somewhere else" is the on-disk debug folder
of an earlier run.

This module is the inverse of the per-stage writes in
`pipeline/runner.py` plus `app.debug.beats_dump` / `app.debug.onsets_dump`.
It's intentionally separate from `app.debug` so that module stays
write-only (the debug sink shouldn't need to know how to read anything).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.models import BestOfKLog, OnsetCandidate
from app.pipeline.beats import BarInfo, BeatStructure, BeatTick
from app.pipeline.runner import PipelineContext, Stage, stage_index

log = logging.getLogger(__name__)


def find_input_audio(folder: Path) -> Path | None:
    """Return the cached input audio in a debug folder, if any.

    `DebugSink.copy_audio("input", src)` writes it as `input.<ext>` — we
    look for any file whose stem is exactly `input`.
    """
    for child in folder.iterdir():
        if child.is_file() and child.stem == "input":
            return child
    return None


def find_drum_stem(folder: Path) -> Path | None:
    """Return the cached drum stem from a previous `stems_all` run, if any."""
    out_dir = folder / "stems_all"
    if not out_dir.is_dir():
        return None
    for child in out_dir.iterdir():
        if child.is_file() and child.stem == "drum_stem":
            return child
    return None


def find_per_instrument_stems(folder: Path) -> dict[str, Path]:
    """Return cached per-instrument stems keyed by pitch letter."""
    out_dir = folder / "stems_per"
    if not out_dir.is_dir():
        return {}
    return {p.stem: p for p in out_dir.iterdir() if p.is_file()}


def hydrate_context_from_resume(
    ctx: PipelineContext,
    folder: Path,
    start_stage: Stage,
) -> None:
    """Populate `ctx` with artifacts produced by stages strictly before
    `start_stage`. Raises `FileNotFoundError` (with a stage-specific
    message) if any required artifact is missing.

    Stages strictly after the last skipped one are expected to run, and
    will write their own outputs — this function never touches their
    fields.
    """
    start_idx = stage_index(start_stage)

    if stage_index(Stage.STEMS_ALL) < start_idx:
        # Only required as input to stems_per. If stems_per is itself
        # being skipped (start_stage > stems_per), we don't actually
        # need drum_stem on disk.
        if stage_index(Stage.STEMS_PER) < start_idx:
            ctx.drum_stem = find_drum_stem(folder)  # may be None; tolerated
        else:
            drum_stem = find_drum_stem(folder)
            if drum_stem is None:
                raise FileNotFoundError(
                    f"resume({start_stage.value}): expected drum stem under "
                    f"{folder}/stems_all/drum_stem.<ext> from the previous "
                    "run, but it's missing. Re-run with "
                    "resume_stage=stems_all to regenerate it."
                )
            ctx.drum_stem = drum_stem

    if stage_index(Stage.STEMS_PER) < start_idx:
        per = find_per_instrument_stems(folder)
        if not per:
            raise FileNotFoundError(
                f"resume({start_stage.value}): expected per-instrument stems "
                f"under {folder}/stems_per/*.<ext> from the previous run, "
                "but none were found. Re-run with resume_stage=stems_per "
                "to regenerate them."
            )
        ctx.per_instrument_stems = per

    if stage_index(Stage.BEATS) < start_idx:
        beats_path = folder / "beats.json"
        if not beats_path.exists():
            raise FileNotFoundError(
                f"resume({start_stage.value}): expected {beats_path} from "
                "the previous run, but it's missing. Re-run with "
                "resume_stage=beats to regenerate it."
            )
        ctx.structure = _load_beats(beats_path)
        if ctx.duration <= 0:
            ctx.duration = _infer_duration(ctx.structure)

    if stage_index(Stage.ONSETS) < start_idx:
        onsets_path = folder / "onsets.json"
        if not onsets_path.exists():
            raise FileNotFoundError(
                f"resume({start_stage.value}): expected {onsets_path} from "
                "the previous run, but it's missing. Re-run with "
                "resume_stage=onsets to regenerate it."
            )
        ctx.onsets_by_pitch = _load_onsets(onsets_path)

    if stage_index(Stage.TRANSCRIBE) < start_idx:
        initial_path = folder / "initial.jot"
        if not initial_path.exists():
            raise FileNotFoundError(
                f"resume({start_stage.value}): expected {initial_path} from "
                "the previous run, but it's missing. Re-run with "
                "resume_stage=transcribe to regenerate it."
            )
        ctx.initial_jot = initial_path.read_text(encoding="utf-8")
        # best_of_k.json is optional — only present when K>1 was used.
        bk_path = folder / "best_of_k.json"
        if bk_path.exists():
            try:
                payload = json.loads(bk_path.read_text(encoding="utf-8"))
                ctx.best_of_k_log = BestOfKLog(**payload)
            except (OSError, ValueError, TypeError) as exc:
                log.warning(
                    "Could not load existing best_of_k.json (%s); "
                    "continuing without best-of-K metadata.", exc,
                )


def load_original_filename(folder: Path) -> str | None:
    """Recover the uploaded filename from `request.json` if present.

    Falls back to the folder slug (the third underscore-separated chunk
    of the folder name, per `debug.DebugSink.for_request`) when no
    summary was written - typical for runs that errored out mid-pipeline,
    which is exactly when resume is most useful.
    """
    request_path = folder / "request.json"
    if request_path.exists():
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            filename = payload.get("filename")
            if isinstance(filename, str) and filename:
                return filename
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s: %s", request_path, exc)
    parts = folder.name.split("_", 2)
    if len(parts) >= 3 and parts[2]:
        return parts[2]
    return None


def _load_beats(path: Path) -> BeatStructure:
    data = json.loads(path.read_text(encoding="utf-8"))
    bars: list[BarInfo] = []
    for bd in data.get("bars", []):
        bar_idx = int(bd["index"])
        # `beats_dump` drops `beat_in_bar` / `bar_index` from per-bar
        # BeatTicks (only the absolute times survive). Reconstruct them
        # from list order, which is sufficient for the downstream LLM /
        # refinement path that only reads BarInfo-level fields.
        beats_in_bar = [
            BeatTick(time=float(t), beat_in_bar=i + 1, bar_index=bar_idx)
            for i, t in enumerate(bd.get("beats", []))
        ]
        bars.append(
            BarInfo(
                index=bar_idx,
                start_time=float(bd["start_time"]),
                end_time=float(bd["end_time"]),
                beats=beats_in_bar,
                time_signature=tuple(bd["time_signature"]),  # type: ignore[arg-type]
                tempo_bpm=float(bd["tempo_bpm"]),
                feel=bd.get("feel", "straight16"),
            )
        )
    all_beats: list[BeatTick] = [
        BeatTick(
            time=float(bt["time"]),
            beat_in_bar=int(bt["beat_in_bar"]),
            bar_index=int(bt["bar_index"]),
        )
        for bt in data.get("beats", [])
    ]
    return BeatStructure(
        beats=all_beats,
        bars=bars,
        initial_tempo=float(data.get("initial_tempo", 120.0)),
        initial_time_signature=tuple(  # type: ignore[arg-type]
            data.get("initial_time_signature", [4, 4])
        ),
        has_tempo_changes=bool(data.get("has_tempo_changes", False)),
        has_time_sig_changes=bool(data.get("has_time_sig_changes", False)),
    )


def _load_onsets(path: Path) -> dict[str, list[OnsetCandidate]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        pitch: [OnsetCandidate(**row) for row in rows]
        for pitch, rows in data.items()
    }


def _infer_duration(structure: BeatStructure) -> float:
    if structure.bars:
        return float(structure.bars[-1].end_time)
    if structure.beats:
        return float(structure.beats[-1].time)
    return 0.0
