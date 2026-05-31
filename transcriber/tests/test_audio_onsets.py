"""Tests for the audio-reference path.

`detect_all_lanes_adtof` (in app.pipeline.adtof_onsets) runs ONE ADTOF
inference and peak-picks all five lanes; `detect_reference_onsets` (in
app.scoring.audio_onsets) decides drum-track-vs-separate and returns the
reference. ADTOF inference + Demucs separation are the heavy external
dependencies and are the only things mocked here; the lane-mapping,
peak-pick, and decision logic run for real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.pipeline import adtof_onsets as ao
from app.scoring.audio_onsets import detect_reference_onsets


def test_detect_all_lanes_adtof_maps_lanes_and_picks_peaks(monkeypatch) -> None:
    frames = 100
    acts = np.zeros((frames, 5))
    acts[10, 0] = 0.99  # kick  -> 0.10 s
    acts[20, 1] = 0.99  # snare -> 0.20 s
    acts[30, 2] = 0.99  # toms  -> 0.30 s
    acts[40, 3] = 0.99  # hat   -> 0.40 s
    acts[50, 4] = 0.99  # cymbal lane 4 -> "cy" -> 0.50 s

    monkeypatch.setattr(ao, "_load_mono_audio", lambda p: np.ones(1000, dtype=np.float32))
    monkeypatch.setattr(ao, "_load_model", lambda: (None, "cpu"))
    monkeypatch.setattr(ao, "_adtof_activations", lambda m, d, a: (acts, 100.0))
    monkeypatch.setattr(
        ao, "_refine_peak_times_audio", lambda path, times, window_sec: list(times)
    )

    out = ao.detect_all_lanes_adtof(Path("/fake/drum.wav"))
    assert out["k"] == pytest.approx([0.10])
    assert out["s"] == pytest.approx([0.20])
    assert out["t"] == pytest.approx([0.30])
    assert out["h"] == pytest.approx([0.40])
    assert out["cy"] == pytest.approx([0.50])


def test_reference_uses_drum_stem_directly_and_skips_separation(tmp_path) -> None:
    drum = tmp_path / "drums.wav"
    drum.write_bytes(b"x")
    seen: dict[str, Path] = {}

    def fake_detect(path: Path) -> dict[str, list[float]]:
        seen["path"] = path
        return {"k": [0.0, 1.0]}

    ref = detect_reference_onsets(
        drum_audio_path=drum, mix_audio_path=None, work_dir=tmp_path, detect=fake_detect
    )
    assert ref.separation_skipped is True
    assert ref.onsets_by_lane == {"k": [0.0, 1.0]}
    assert seen["path"] == drum  # ADTOF ran on the pre-isolated drum track


def test_reference_separates_mix_when_no_drum_track(tmp_path) -> None:
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"x")
    produced_stem = tmp_path / "drum_stem.wav"
    produced_stem.write_bytes(b"y")

    class FakeStemsAll:
        drum_stem = produced_stem

    class FakeSeparator:
        def run_stems_all(self, audio_path: Path, work_dir: Path) -> FakeStemsAll:
            assert audio_path == mix
            return FakeStemsAll()

    ref = detect_reference_onsets(
        drum_audio_path=None,
        mix_audio_path=mix,
        work_dir=tmp_path,
        separator=FakeSeparator(),
        detect=lambda path: {"s": [0.5]} if path == produced_stem else {},
    )
    assert ref.separation_skipped is False
    assert ref.onsets_by_lane == {"s": [0.5]}


def test_reference_requires_some_audio(tmp_path) -> None:
    with pytest.raises(ValueError, match="audio"):
        detect_reference_onsets(
            drum_audio_path=None, mix_audio_path=None, work_dir=tmp_path, detect=lambda p: {}
        )
