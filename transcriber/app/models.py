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
    samples: int
    scores: list[float] = Field(default_factory=list)
    chosen_index: int = 0


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
    """

    jot_dsl: str
    metadata: TranscribeMetadata
    refinement: RefinementLog | None = None
    best_of_k: BestOfKLog | None = None
    candidates: dict[str, list[OnsetCandidate]] = Field(default_factory=dict)
    debug_dir: str | None = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str | None = None
