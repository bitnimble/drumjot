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

    `quantised_time` is set by the `quantise` pipeline stage (when
    enabled) to the snap-corrected absolute time. When None, downstream
    consumers fall back to `time`. The original `time` / `beat_in_bar`
    fields are intentionally left untouched so per-note provenance can
    still report the original detector hit.
    """

    time: float
    strength: float
    bar: int = -1
    beat_in_bar: float = -1.0
    quantised_time: float | None = None
    # Integer 1/48-slot shift the `quantise` stage applied (sum of the
    # deterministic joint-snap pass and the LLM residual pass). Inspection
    # / debug only; the canonical post-quantise time is `quantised_time`.
    quantised_shift_slots: int | None = None


class TranscribeResponse(BaseModel):
    """Returned by POST /transcribe.

    The score is the predicted MIDI at `prediction_midi_url`; the
    frontend's deterministic `src/midi/from_midi.ts` converts that to a
    Drumjot Jot. `metadata` carries the beat-tracker output for the UI's
    status pill / debug surfaces.

    `debug_dir` is the absolute path inside the service container where
    intermediate artifacts (drum stems, per-instrument stems, beats.json,
    onsets.json, prediction.mid, note_provenance.json) were written, or
    `None` if debug persistence was disabled for this request. When
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

    metadata: TranscribeMetadata
    candidates: dict[str, list[OnsetCandidate]] = Field(default_factory=dict)
    debug_dir: str | None = None
    drum_stem_url: str | None = None
    no_drums_url: str | None = None
    # URL path (no host, served under `/outputs/...`) of the kept
    # onsets rendered as a MIDI file. The score itself; the frontend
    # converts it to a Jot via `src/midi/from_midi.ts`.
    prediction_midi_url: str | None = None
    # URL path (no host) to the debug zip bundle for this run. The zip
    # holds the prediction MIDI, per-note provenance, MP3-encoded
    # per-stem + drumless audio, and a JSON manifest with stage timings
    # + the full captured log stream so the operator can inspect what
    # happened end-to-end. Designed to be downloaded and re-loaded in
    # the web UI to reconstitute the score + audio tracks + debug info
    # offline. None if bundling failed.
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
