"""Pydantic request / response schemas for the transcriber HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BarSummary(BaseModel):
    """Per-bar metadata recovered by beat tracking. Exposed to the
    frontend so users can see when the pipeline detected tempo or
    time-signature changes.
    """

    bar: int  # 0-indexed
    time_signature: str  # e.g. "4/4"
    tempo_bpm: float
    feel: str            # "straight16", "triplet", "shuffle", ...
    start_time: float


class TranscribeMetadata(BaseModel):
    initial_tempo: float
    initial_time_signature: list[int]
    duration_seconds: float
    stems_used: list[str]
    bars: list[BarSummary] = Field(default_factory=list)
    has_tempo_changes: bool = False
    has_time_sig_changes: bool = False


class OnsetCandidate(BaseModel):
    """One detected onset before LLM filtering, annotated with the
    beat-relative position produced by `pipeline/beats.py`.

    `bar` is 0-indexed (matches `TranscribeMetadata.bars`). `beat_in_bar`
    is 1-indexed and floating-point: integer part = beat number, fraction
    = how far into that beat (so 2.333 = "1/3 into beat 2").

    Both fields are -1 / -1.0 for onsets that fall outside the
    beat-tracked region.
    """

    time: float
    strength: float
    bar: int = -1
    beat_in_bar: float = -1.0


class RefinementIteration(BaseModel):
    level: str
    iteration: int
    issues_detected: int
    issues_sent_to_llm: int
    score_before: float
    score_after: float
    accepted: bool
    note: str = ""


class RefinementLog(BaseModel):
    """Summary of the refinement loop, surfaced to the UI for transparency."""

    initial_score: float
    final_score: float
    elapsed_seconds: float
    iterations: list[RefinementIteration] = Field(default_factory=list)


class BestOfKLog(BaseModel):
    """Best-of-K sampling log.

    Transcription is per-instrument, so best-of-K is applied per
    instrument: `per_instrument[pitch]` carries that instrument's K
    sample scores + chosen index. The top-level `scores`/`chosen_index`
    are unused in the per-instrument case (left empty / 0) and retained
    for the per-instrument *sub*-logs and backward-compatible decoding
    of older `best_of_k.json` artifacts.
    """

    samples: int
    scores: list[float] = Field(default_factory=list)
    chosen_index: int = 0
    per_instrument: dict[str, BestOfKLog] | None = None


class TranscribeResponse(BaseModel):
    """Returned by POST /transcribe.

    `jot_dsl` is the canonical Drumjot DSL string. The Drumjot frontend
    parses it via `src/parser` and loads the result.

    `debug_dir` is the absolute path inside the service container where
    intermediate artifacts (drum stems, per-instrument stems, beats.json,
    onsets.json, initial.jot, final.jot, refinement.json) were written,
    or `None` if debug persistence was disabled for this request. When
    running under docker-compose with the default `./debug:/debug` mount,
    this path corresponds 1:1 with a host folder.

    `drum_stem_url` and `no_drums_url` are paths (with leading `/`) to
    FLAC-encoded stem deliverables — the isolated drum mix and the
    bass+other+vocals "music minus drums" mix respectively. They are
    served by the same transcriber service under `/outputs/...` and
    are intended to be composed against the caller's transcriber base
    URL (e.g. `/api` in dev, `https://...` in prod). Either can be
    `None` if the corresponding stem couldn't be produced (e.g. resume
    skipping stems_all and no cached FLAC on disk).
    """

    jot_dsl: str
    metadata: TranscribeMetadata
    refinement: RefinementLog | None = None
    best_of_k: BestOfKLog | None = None
    candidates: dict[str, list[OnsetCandidate]] = Field(default_factory=dict)
    debug_dir: str | None = None
    drum_stem_url: str | None = None
    no_drums_url: str | None = None
    # Set only by the `filter` transcribe path: URL path (no host, served
    # under `/outputs/...`) of the predicted onsets rendered as a MIDI
    # file. The `filter` path produces this *instead of* `jot_dsl`
    # (which is then empty); accuracy is scored directly on this MIDI.
    prediction_midi_url: str | None = None
    # URL path (no host) to the debug zip bundle for this run. The zip
    # holds the final.jot, MP3-encoded per-stem + drumless audio, and a
    # JSON manifest with stage timings + the full captured log stream so
    # the operator can inspect what happened end-to-end. Designed to be
    # downloaded and re-loaded in the web UI to reconstitute the score +
    # audio tracks + debug info offline. None if bundling failed.
    debug_zip_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str | None = None


class TranscriptionSummary(BaseModel):
    """One entry in the GET /transcribe/list response.

    Built from a single per-request debug folder (`/debug/<folder>/`).
    The folder name is the stable identifier the resume endpoint takes
    as `resume_folder`; the other fields are diagnostic, surfaced so the
    UI can show a useful picker without having to fetch each folder
    separately.

    `requested_at` is parsed from the folder name's `<YYYYMMDD-HHMMSS>`
    stamp (set once when the original /transcribe run minted the folder
    via `debug.mint_request_folder_name`). `last_run_at` is the
    modification time of `request.json`, which `DebugSink.finalize`
    overwrites at the end of every run (initial or resume), so it
    captures the most-recent run that produced artifacts. When that
    most-recent run was a resume, `last_resume_stage` carries which
    stage it restarted from (echoed by `_request_options` into
    `request.json`).
    """

    folder: str
    original_filename: str | None = None
    requested_at: str
    last_run_at: str | None = None
    last_resume_stage: str | None = None
    # Stages whose required prior artifacts exist on disk, so a resume
    # request starting from that stage can succeed without a 400.
    # Ordered by `STAGE_ORDER`.
    resumable_stages: list[str] = Field(default_factory=list)
