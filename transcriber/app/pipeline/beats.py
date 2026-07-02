"""Beat tracking + downbeat detection + per-bar feel analysis.

This module replaces the older `tempo.py` + `quantize.py` pair. Rather
than assuming a constant tempo + fixed grid (1/16 by default), we use
Beat This! (ISMIR 2024; DBN-free, meter-agnostic) to extract per-beat
anchors + downbeats from the audio. Everything downstream then works in
**beat-relative** time -
that is, "1/3 of the way through beat 2 of bar 3" rather than
"slot 47 of a 1/16 grid".

Why it matters:

- Tempo changes are handled naturally: each beat has its own absolute
  time, so a song that accelerates or hits a click drop still maps
  onsets onto the right beats.
- Time-signature changes are detected from the downbeat-classifier
  output (gaps of 4 beats between downbeats -> 4/4; 7 beats -> 7/4 or
  7/8 depending on tempo).
- Triplet and swing feel become a property of intra-beat fractions and
  are encoded as the bar's `feel` field; the LLM can then emit
  `(...)_N` groups vs straight grids per bar.
- The grid is implicit and per-bar, not a global constant - the LLM
  decides how to position each onset in DSL space based on the bar's
  feel and the onset's beat fraction.

This file is the stable public facade. The implementation is split across
cohesive focused modules; everything is re-exported here so existing call
sites (`from app.pipeline.beats import …`, `beats.park_model()`) are unchanged:

- `beats_types`     - `BeatTick` / `BarInfo` / `TempoSegment` / `BeatStructure`
                      + the `Feel` label and shared constants.
- `beats_structure` - raw->structure assembly + robust global summary.
- `beats_meter`     - odd-meter downbeat recovery + downbeat smoothing.
- `beats_tempo`     - tempo-segment fitting + grid regularization + padding.
- `beats_align`     - coarse envelope + fine onset grid alignment.
- `beats_detect`    - Beat This! model + `analyze_beats` orchestration.
- `beats_feel`      - per-bar feel detection + LLM-friendly summary.
"""
# ruff: noqa: F401  -- this module is a re-export facade; the "unused" imports
# are the point (they keep `from app.pipeline.beats import …` call sites intact).
from __future__ import annotations

from app.pipeline.beats_align import (
    COARSE_CENTER_PENALTY,
    COARSE_ENV_HOP,
    COARSE_MAX_SHIFT_BEATS,
    COARSE_PROMINENCE,
    COARSE_SEARCH_STEP_SEC,
    MIN_ALIGN_COVERAGE,
    _coarse_offset_from_envelope,
    align_beats_to_envelope,
    align_beats_to_onsets,
    detect_envelope_onsets_for_alignment,
)
from app.pipeline.beats_detect import (
    _beat_onnx_enabled,
    _beat_this_beats,
    _beat_this_model,
    _librosa_fallback,
    _onnx_providers,
    analyze_beats,
    beat_engine_name,
    park_model,
    unpark_model,
)
from app.pipeline.beats_feel import (
    _FEEL_GRIDS,
    CandidateOnset,
    _intra_beat_fraction,
    _score_feel,
    candidates_with_beat_positions,
    detect_feel_for_bars,
    summarize_bar_for_prompt,
)
from app.pipeline.beats_meter import (
    _METER_CANDIDATES,
    _METER_FUNDAMENTAL_FRAC,
    _METER_MIN_AUTOCORR,
    _METER_MIN_BARS,
    _METER_MIN_BEATS,
    _METER_TRUST_DOM_FRAC,
    _bar_length_from_autocorr,
    _beat_sync_strength,
    _beats_downbeats_to_raw,
    _best_downbeat_phase,
    _dominant_gap_fraction,
    _merge_fragment_bars,
    _recover_bar_length_if_incoherent,
    _smooth_downbeats,
)
from app.pipeline.beats_structure import (
    _choose_time_signature,
    _finalize_bar,
    _has_sustained_meter_change,
    _modal_time_signature,
    _raw_to_structure,
    _rebuild_bar_fields,
    _reference_bars,
    _robust_initial_tempo,
    _summarize,
)
from app.pipeline.beats_tempo import (
    _BOUNDARY_REFINE_HALF,
    _BOUNDARY_REFINE_PROMINENCE,
    _DRIFT_DEADBAND_SEC,
    _DRIFT_SMOOTHING_WINDOW,
    _MIN_SEG_BEATS,
    _SEGMENT_FIT_TOL_BEATS,
    _enforce_monotonic_times,
    _finalize_bar_tempos,
    _fit_range,
    _greedy_segment_ranges,
    _pad_trailing_bars,
    _quad_resid,
    _refine_step_boundaries,
    _regularize_beats_quadratic,
    _resample_beats_uniform,
    _segment_beats,
)
from app.pipeline.beats_types import (
    _RAMP_EPS_BPM,
    BEATS_PER_BAR_CANDIDATES,
    BarInfo,
    BeatStructure,
    BeatTick,
    Feel,
    TempoSegment,
)

__all__ = [
    "BEATS_PER_BAR_CANDIDATES",
    "Feel",
    "BeatTick",
    "BarInfo",
    "TempoSegment",
    "BeatStructure",
    "CandidateOnset",
    "analyze_beats",
    "beat_engine_name",
    "park_model",
    "unpark_model",
    "detect_envelope_onsets_for_alignment",
    "align_beats_to_envelope",
    "align_beats_to_onsets",
    "detect_feel_for_bars",
    "candidates_with_beat_positions",
    "summarize_bar_for_prompt",
    "MIN_ALIGN_COVERAGE",
    "COARSE_MAX_SHIFT_BEATS",
    "COARSE_SEARCH_STEP_SEC",
    "COARSE_ENV_HOP",
    "COARSE_CENTER_PENALTY",
    "COARSE_PROMINENCE",
]
