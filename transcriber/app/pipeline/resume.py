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
from datetime import UTC, datetime
from pathlib import Path

from app.models import OnsetCandidate, TranscriptionSummary
from app.pipeline.beats import BarInfo, BeatStructure, BeatTick
from app.pipeline.runner import STAGE_ORDER, PipelineContext, Stage, stage_index
from app.pipeline.separate import PITCH_DISPLAY_NAMES

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
    """Return cached per-instrument stems keyed by pitch letter.

    Filenames whose stem is not a known DSL pitch letter (per
    `PITCH_DISPLAY_NAMES`) are ignored — most importantly this excludes
    `residual.<ext>`, which lives alongside the per-instrument stems
    but is a diagnostic-only track, not an instrument.
    """
    out_dir = folder / "stems_per"
    if not out_dir.is_dir():
        return {}
    return {
        p.stem: p
        for p in out_dir.iterdir()
        if p.is_file() and p.stem in PITCH_DISPLAY_NAMES
    }


def find_residual_stem(folder: Path) -> Path | None:
    """Return the cached residual track from a previous `stems_per` run."""
    out_dir = folder / "stems_per"
    if not out_dir.is_dir():
        return None
    for child in out_dir.iterdir():
        if child.is_file() and child.stem == "residual":
            return child
    return None


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
        # Residual is diagnostic-only; tolerated when missing (older
        # debug folders predate it).
        ctx.residual_stem = find_residual_stem(folder)

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

    if stage_index(Stage.FILTER) < start_idx:
        kept_path = folder / "filter" / "kept_onsets.json"
        if not kept_path.exists():
            raise FileNotFoundError(
                f"resume({start_stage.value}): expected {kept_path} from "
                "the previous run, but it's missing. Re-run with "
                "resume_stage=filter to regenerate it."
            )
        ctx.kept_by_pitch = _load_kept_onsets(kept_path, ctx.onsets_by_pitch)


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


def _parse_folder_timestamp(name: str) -> str | None:
    """Parse the `<YYYYMMDD-HHMMSS>` prefix minted by
    `debug.mint_request_folder_name` back into an ISO-8601 string.

    Returns None when the prefix doesn't parse — likely an old folder or
    one created by hand. Callers fall back to file mtimes in that case.

    The folder-name stamp is UTC (see `mint_request_folder_name`); emit
    it with a `Z` suffix so the UI can convert to the operator's local
    time unambiguously.
    """
    stamp = name.split("_", 1)[0]
    try:
        dt = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _resumable_stages(folder: Path) -> list[str]:
    """Return the stages whose `start_stage=…` would succeed for this
    folder, ordered by `STAGE_ORDER`.

    Mirrors the per-stage hydration checks in
    `hydrate_context_from_resume`: a stage is resumable iff every
    artifact strictly *before* it exists on disk. STEMS_ALL is always
    listed when the input audio is cached (it's the from-scratch resume
    that re-uses the upload), regardless of any later artifacts.
    """
    out: list[str] = []
    # STEMS_ALL = re-run everything from the cached upload.
    if find_input_audio(folder) is not None:
        out.append(Stage.STEMS_ALL.value)
    # STEMS_PER needs the drum stem.
    has_drum_stem = find_drum_stem(folder) is not None
    if has_drum_stem:
        out.append(Stage.STEMS_PER.value)
    # BEATS needs per-instrument stems.
    has_per = bool(find_per_instrument_stems(folder))
    if has_per:
        out.append(Stage.BEATS.value)
    has_beats = (folder / "beats.json").is_file()
    has_onsets = (folder / "onsets.json").is_file()
    has_filter = (folder / "filter" / "kept_onsets.json").is_file()
    if has_per and has_beats:
        out.append(Stage.ONSETS.value)
    if has_per and has_beats and has_onsets:
        out.append(Stage.FILTER.value)
    if has_per and has_beats and has_onsets and has_filter:
        out.append(Stage.TRANSCRIBE.value)
    # Preserve STAGE_ORDER so the UI can render the picker without
    # re-sorting (and so a future stage insertion lands where it
    # should).
    order = {s.value: i for i, s in enumerate(STAGE_ORDER)}
    out.sort(key=lambda s: order[s])
    return out


def list_transcription_summaries(base: Path) -> list[TranscriptionSummary]:
    """Build the GET /transcribe/list payload.

    Tolerant of partial folders (mid-pipeline errors, hand-created
    directories): each folder is independently parsed and any failure
    is logged and skipped rather than failing the whole listing.
    Sorted with the most-recently-run folder first.
    """
    if not base.exists() or not base.is_dir():
        return []
    summaries: list[TranscriptionSummary] = []
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        try:
            summary = _summarize_folder(folder)
        except OSError as exc:
            log.warning("Could not summarize %s: %s", folder, exc)
            continue
        if summary is not None:
            summaries.append(summary)
    # Sort by `last_run_at` desc, with folders that have no recorded
    # run time tucked at the bottom (they're least useful to resume).
    summaries.sort(
        key=lambda s: (s.last_run_at or "", s.requested_at),
        reverse=True,
    )
    return summaries


def _summarize_folder(folder: Path) -> TranscriptionSummary | None:
    """Read a single debug folder into its summary row. Returns None
    when the folder is too empty to be a real run (no input audio AND
    no request.json AND no stage artifacts) — keeps the listing free of
    noise."""
    request_payload: dict[str, object] | None = None
    request_path = folder / "request.json"
    last_run_at: str | None = None
    if request_path.is_file():
        try:
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s: %s", request_path, exc)
            request_payload = None
        try:
            mtime = request_path.stat().st_mtime
            last_run_at = datetime.fromtimestamp(mtime, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except OSError:
            last_run_at = None

    last_resume_stage: str | None = None
    if isinstance(request_payload, dict):
        options = request_payload.get("options")
        if isinstance(options, dict):
            stage_val = options.get("resume_stage")
            if isinstance(stage_val, str) and stage_val:
                last_resume_stage = stage_val

    resumable = _resumable_stages(folder)
    requested_at = _parse_folder_timestamp(folder.name)

    # Skip folders that look empty (no input, no artifacts, no
    # request.json). They're either pre-mature requests that aborted
    # before any stage wrote anything, or unrelated directories.
    if request_payload is None and not resumable:
        return None

    if requested_at is None:
        # Fall back to mtime so the row still has a usable timestamp.
        try:
            stat = folder.stat()
            requested_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except OSError:
            requested_at = ""

    return TranscriptionSummary(
        folder=folder.name,
        original_filename=load_original_filename(folder),
        requested_at=requested_at,
        last_run_at=last_run_at,
        last_resume_stage=last_resume_stage,
        resumable_stages=resumable,
    )


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
    # Older beats.json files (pre-alignment-tracking) won't have the
    # field, and earlier code wrote `null` on rejected alignments —
    # default both to 0.0 so the rest of the pipeline always sees a
    # number. New runs always persist a numeric value.
    raw_offset = data.get("align_offset_sec")
    align_offset_sec = (
        float(raw_offset) if isinstance(raw_offset, (int, float)) else 0.0
    )
    return BeatStructure(
        beats=all_beats,
        bars=bars,
        initial_tempo=float(data.get("initial_tempo", 120.0)),
        initial_time_signature=tuple(  # type: ignore[arg-type]
            data.get("initial_time_signature", [4, 4])
        ),
        has_tempo_changes=bool(data.get("has_tempo_changes", False)),
        has_time_sig_changes=bool(data.get("has_time_sig_changes", False)),
        align_offset_sec=align_offset_sec,
    )


def _load_onsets(path: Path) -> dict[str, list[OnsetCandidate]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        pitch: [OnsetCandidate(**row) for row in rows]
        for pitch, rows in data.items()
    }


def _load_kept_onsets(
    path: Path,
    onsets_by_pitch: dict[str, list[OnsetCandidate]],
) -> dict[str, list[OnsetCandidate]]:
    """Load `filter/kept_onsets.json` and re-thread identity against
    `onsets_by_pitch` so `build_note_provenance`'s `id(c)` kept-vs-
    rejected match still works after a resume.

    `filter/kept_onsets.json` stores only the value fields (time / bar /
    beat / strength), so a naive `OnsetCandidate(**row)` would mint new
    objects whose `id()` doesn't match anything in `onsets_by_pitch`; every candidate would then appear as "rejected" in the provenance
    even though it was kept. We look each row up by
    `(time, bar, beat_in_bar)` against the matching pitch in
    `onsets_by_pitch` and re-attach the same instance.

    Falls back to constructing a fresh `OnsetCandidate` only when the
    row can't be matched (e.g. `onsets.json` was regenerated more
    recently than `filter/kept_onsets.json` and the candidate counts
    drifted). The fallback candidate won't be present in
    `all_onsets_by_pitch` so the provenance will mark it as "kept" but
    won't surface a rejected ghost for it; acceptable degradation.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    kept: dict[str, list[OnsetCandidate]] = {}
    # Round float keys so a tiny serialization wobble (JSON's str(float)
    # repr) doesn't break the lookup.
    def _key(c_or_row: OnsetCandidate | dict) -> tuple[float, int, float]:
        if isinstance(c_or_row, dict):
            return (
                round(float(c_or_row["time"]), 6),
                int(c_or_row["bar"]),
                round(float(c_or_row["beat_in_bar"]), 6),
            )
        return (
            round(float(c_or_row.time), 6),
            int(c_or_row.bar),
            round(float(c_or_row.beat_in_bar), 6),
        )

    for pitch, rows in data.items():
        source = onsets_by_pitch.get(pitch, [])
        by_key: dict[tuple[float, int, float], OnsetCandidate] = {
            _key(c): c for c in source
        }
        matched: list[OnsetCandidate] = []
        unmatched = 0
        for row in rows:
            existing = by_key.get(_key(row))
            if existing is not None:
                matched.append(existing)
            else:
                unmatched += 1
                matched.append(OnsetCandidate(**row))
        if unmatched:
            log.warning(
                "resume filter/kept_onsets.json: %d/%d entries for pitch %r "
                "did not match an onset in onsets.json (likely re-derived "
                "after the filter ran); rejected-ghost rendering may be "
                "incomplete for those candidates.",
                unmatched, len(rows), pitch,
            )
        kept[pitch] = matched
    return kept


def _infer_duration(structure: BeatStructure) -> float:
    if structure.bars:
        return float(structure.bars[-1].end_time)
    if structure.beats:
        return float(structure.beats[-1].time)
    return 0.0
