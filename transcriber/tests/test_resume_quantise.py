"""Resume path: re-attaching a prior run's quantise results.

`_apply_quantise_shifts` hydrates `quantise/shifts.json` back onto kept
onsets when resuming from a post-quantise stage (so the MIDI render is
reproduced without re-running the quantise LLM). It must restore the
`off_grid` flag too, not just `quantised_time` / `quantised_shift_slots`,
or a band-rejected onset comes back looking like an ordinary on-grid hit.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.models import OnsetCandidate
from app.pipeline.resume import _apply_quantise_shifts


def test_apply_quantise_shifts_restores_off_grid_and_times(tmp_path: Path) -> None:
    shifts = {
        "per_pitch": {
            "k": [
                # Band-rejected onset: off-grid, no quantised time.
                {"idx": 0, "off_grid": True, "quantised_time": None, "total_shift": 0},
                # Ordinary shifted onset.
                {"idx": 1, "off_grid": False, "quantised_time": 0.5, "total_shift": 1},
            ]
        }
    }
    path = tmp_path / "shifts.json"
    path.write_text(json.dumps(shifts), encoding="utf-8")

    kept = {
        "k": [
            OnsetCandidate(time=0.10, strength=1.0, bar=0, beat_in_bar=1.0),
            OnsetCandidate(time=0.49, strength=1.0, bar=0, beat_in_bar=2.0),
        ]
    }
    _apply_quantise_shifts(path, kept)

    # Off-grid onset: flag restored, time stays raw (None).
    assert kept["k"][0].off_grid is True
    assert kept["k"][0].quantised_time is None
    # On-grid onset: time + shift restored, off_grid stays False.
    assert kept["k"][1].off_grid is False
    assert kept["k"][1].quantised_time == 0.5
    assert kept["k"][1].quantised_shift_slots == 1
