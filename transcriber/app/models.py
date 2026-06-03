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
    # Peak audio amplitude (|sample| in [0, 1]) in a small window around
    # `time`, measured on the source stem before model-input normalisation.
    # Drives MIDI velocity in `onsets_midi.py::_build_velocity_lookup`:
    # per-pitch p10/p90 percentile-normalised so each lane's loudness
    # spread maps cleanly to the [VEL_FLOOR, VEL_CEIL] range.
    #
    # Distinct from `strength`, which is the ADTOF model's per-frame
    # confidence (in [0, 1]) at the peak, a "is this a hit?" signal.
    # The two diverge often enough to matter: a quiet but clean snare
    # attack reads as confident (high `strength`), while a louder hit
    # with unusual spectral content (or bleed) can read as uncertain.
    # `strength` continues to feed the filter LLM's `(beat_in_bar,
    # strength)` prompt block, where confidence is what we want; only
    # the velocity mapping switched to `amplitude`.
    #
    # `None` for non-ADTOF detection paths (none in production today)
    # and for re-loaded legacy debug bundles; velocity_lookup falls back
    # to `strength` in that case.
    amplitude: float | None = None
    bar: int = -1
    beat_in_bar: float = -1.0
    # ADTOF's raw model peak time, BEFORE `_refine_peak_times_audio`'s
    # envelope-local-max snap (`time` above is the post-refine value).
    # Populated by `detect_onsets_adtof` only, non-ADTOF code paths leave
    # this None. Surfaced in `note_provenance.json` so the per-note debug
    # popup can show the envelope-refine shift as its own stage; consumers
    # that just want "where did the detector say the onset was" should
    # keep reading `time`.
    raw_model_time: float | None = None
    quantised_time: float | None = None
    # Integer slot shift the `quantise` stage applied (sum of the geometric
    # snap and the LLM residual pass). Inspection / debug only; the
    # canonical post-quantise time is `quantised_time`.
    quantised_shift_slots: int | None = None
    # Per-pass slot shifts within the `quantise` stage, surfaced separately
    # so the per-note debug popup can attribute every quantise-induced
    # movement to a specific pass instead of showing one collapsed sum.
    # `0` means "the pass processed this onset but didn't shift it",
    # `None` means "the pass didn't run / didn't see this onset" (e.g.
    # envelope pass skipped because no envelope was supplied; LLM pass
    # skipped or rejected by the monotonic-injective guard). The sum of
    # the four equals `quantised_shift_slots` for any onset that ran the
    # full chain.
    geometric_shift_slots: int | None = None
    envelope_shift_slots: int | None = None
    grid_shift_slots: int | None = None
    llm_shift_slots: int | None = None
    # Signed sub-slot residual from the geometric snap: the fractional part
    # of the onset's natural slot position, i.e. how far (in slots, range
    # (-0.5, +0.5]) the raw detector timing sat from the integer slot it was
    # rounded to. + = the hit was late of its slot, - = early. None for
    # off-grid onsets (no rounded slot) and for onsets the quantise stage
    # never processed. Informational provenance carried to the musical-grid
    # pass and the LLM residual pass; NOT a correctness gate (a performer
    # consistently a full slot off rounds *cleanly* onto the wrong slot, so
    # a residual near 0 does not mean the slot is right).
    quantised_residual_slots: float | None = None
    # True when the geometric snap deliberately left this onset off-grid
    # (band-rejected: no free slot within the match band). Off-grid onsets
    # keep `quantised_time = None` so the MIDI emitter uses their raw
    # `time`; the frontend then records the sub-slot residual as the note's
    # `offset` (swing / ghost-flam / push-pull feel). Distinct from a
    # `quantised_time = None` that merely means "no shift was needed".
    off_grid: bool = False

    # --- Acoustic features (populated by the split passes) ---------------
    #
    # The cymbal / hi-hat split LLMs decide ride-vs-crash and
    # closed-vs-open by reading these per-onset acoustic features off
    # the stem audio. Each field is populated only by the pass(es) that
    # measure it; `None` everywhere else (other pitches' candidates,
    # bundles produced before this field existed, etc.). Surfaced
    # through `note_provenance.json` so the per-note debug popup's
    # "Acoustic properties" subsection can show the same numbers the
    # classifier saw, useful when a crash is mis-labelled as ride, or
    # an open hat as closed.
    #
    # Set by both cymbal_split and hihat_split:
    #   decay_s, flatness, centroid_hz, gap_s
    # Set only by hihat_split (open/closed-discriminating envelope):
    #   attack_s, attack_flux, late_rms, pre_rms, tail_end_s
    decay_s: float | None = None
    flatness: float | None = None
    centroid_hz: float | None = None
    gap_s: float | None = None
    attack_s: float | None = None
    # Hi-hat-only: peak onset-strength flux at the strike / stem-median
    # flux. A real strike (even a soft one on a loud ring) produces a fresh
    # spectral-flux spike; a sizzle re-trigger inside a ring does not.
    # Drives the open-within-open drop in `hihat_split.py` and is shown to
    # the split LLM; surfaced for the "Acoustic properties" popup.
    attack_flux: float | None = None
    # Hi-hat-only: fraction of occupied-band (~200 Hz-14 kHz) energy that
    # sits in the low band (~200-1500 Hz). Low for a real hi-hat (high-band
    # noise), high for snare/kick bleed (low-mid body). The discard-rescue's
    # bleed guard reads this; surfaced for the "Acoustic properties" popup.
    lowband_ratio: float | None = None
    late_rms: float | None = None
    pre_rms: float | None = None
    # Hi-hat-only: seconds from the onset to the point where its ring is
    # considered over (per the `_TAIL_END_FRAC` / `_TAIL_MIN_S` rule in
    # `hihat_split.py`). Consulted by the open-tail post-filter; surfaced
    # for visibility.
    tail_end_s: float | None = None


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
