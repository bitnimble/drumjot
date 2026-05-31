"""Result schema for the alignment scorer (also the `POST /score` response).

`score_corrected` is the headline filter metric: the rigid soft-F1 score
after global offset+tempo alignment, so notation faithfulness isn't tanked
by a fixable global drift. `score` is the pre-correction number; a large gap
between them flags gross drift, which `offset_sec` / `tempo_ratio` quantify.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class LaneScoreOut(BaseModel):
    soft_f1: float
    soft_precision: float
    soft_recall: float
    n_chart: int
    n_audio: int


class AlignmentResult(BaseModel):
    score: int  # round(100 * f1_weighted), pre-correction
    score_corrected: int  # headline filter metric, post global-align
    f1_macro: float  # corrected
    f1_weighted: float  # corrected (basis of score_corrected)
    f1_weighted_raw: float
    # Per-lane breakdown at the corrected positions.
    per_lane: dict[str, LaneScoreOut] = Field(default_factory=dict)
    offset_sec: float  # tier-0 b in t' = a*t + b
    tempo_ratio: float  # tier-1 a (1.0 = no tempo correction)
    matched_pairs: int  # pairs the affine fit used (trust signal)
    # Chart onset seconds after t' = a*t + b: a batch run applies these to
    # emit cleaned (audio, chart) training pairs.
    corrected_onsets_by_lane: dict[str, list[float]] = Field(default_factory=dict)
    unmapped_notes: int = 0
    audio_reference: str = "separated"  # "drum_track" | "separated"
    separation_skipped: bool = False
