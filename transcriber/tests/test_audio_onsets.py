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


def test_merge_audio_onsets_dedupes_and_unions() -> None:
    # ADTOF found frames 10, 50; audio supplement found 11 (dup of 10,
    # within dedup), 30 (new), 52 (dup of 50), 200 (out of range), -1
    # (out of range). Expect 10, 30, 50, 30 added, dups + OOR dropped.
    adtof = np.array([10, 50], dtype=int)
    audio = np.array([11, 30, 52, 200, -1], dtype=int)
    out = ao._merge_audio_onsets(adtof, n_frames=100, audio_frames=audio, dedup_frames=3)
    assert out.tolist() == [10, 30, 50]


def test_merge_audio_onsets_empty_audio_is_noop() -> None:
    adtof = np.array([5, 9], dtype=int)
    out = ao._merge_audio_onsets(adtof, n_frames=100,
                                 audio_frames=np.empty(0, dtype=int), dedup_frames=4)
    assert out.tolist() == [5, 9]


def _cand(amp):
    from app.models import OnsetCandidate
    return OnsetCandidate(time=0.0, strength=0.5, amplitude=amp)


def test_amplitude_floor_drops_near_silent_phantoms() -> None:
    # 8 real hits at ~1.0 + 2 phantoms at 0.1 -> median ~1.0, floor 0.25 drops
    # the phantoms, keeps the real hits.
    cands = [_cand(1.0) for _ in range(8)] + [_cand(0.1), _cand(0.1)]
    kept, dropped = ao._apply_amplitude_floor(cands, frac=0.25, min_onsets=8)
    assert dropped == 2
    assert all(c.amplitude == 1.0 for c in kept)


def test_amplitude_floor_skips_when_too_few_onsets() -> None:
    # Below min_onsets the median is unstable -> keep everything.
    cands = [_cand(1.0), _cand(0.01)]
    kept, dropped = ao._apply_amplitude_floor(cands, frac=0.25, min_onsets=8)
    assert dropped == 0 and len(kept) == 2


def test_amplitude_floor_disabled_when_frac_zero() -> None:
    cands = [_cand(1.0) for _ in range(8)] + [_cand(0.01)]
    kept, dropped = ao._apply_amplitude_floor(cands, frac=0.0, min_onsets=8)
    assert dropped == 0 and len(kept) == 9


def test_amplitude_floor_never_drops_none_amplitude() -> None:
    cands = [_cand(1.0) for _ in range(8)] + [_cand(None), _cand(0.01)]
    kept, dropped = ao._apply_amplitude_floor(cands, frac=0.25, min_onsets=8)
    # the 0.01 phantom drops; the None (no signal to judge) is kept.
    assert dropped == 1
    assert any(c.amplitude is None for c in kept)


# --- _bloom_amplitude (cymbal loudness = the bloom, not the attack) -----

def _crash_like(sr: int) -> np.ndarray:
    # A crash: quiet stick attack at 0.5s, loud wash bloom 100ms later.
    a = np.zeros(int(1.0 * sr), dtype=np.float32)
    a[int(0.5 * sr)] = 0.2   # attack
    a[int(0.6 * sr)] = 1.0   # bloom
    return a


def test_bloom_amplitude_catches_late_bloom() -> None:
    sr = 44100
    a = _crash_like(sr)
    # The ±20ms attack window sees only the quiet strike; the forward
    # bloom window catches the loud wash 100ms later.
    assert ao._peak_amplitude(a, 0.5, sr) == pytest.approx(0.2)
    assert ao._bloom_amplitude(a, 0.5, next_time_sec=2.0, sample_rate=sr) == pytest.approx(1.0)


def test_bloom_amplitude_caps_at_next_onset() -> None:
    sr = 44100
    a = _crash_like(sr)
    # A next onset at 0.55s caps the window before the 0.6s bloom, so the
    # value stays at the attack level (can't borrow the next hit's energy).
    assert ao._bloom_amplitude(a, 0.5, next_time_sec=0.55, sample_rate=sr) == pytest.approx(0.2)


def test_bloom_amplitude_empty_window_is_zero() -> None:
    sr = 44100
    a = _crash_like(sr)
    # Degenerate: next onset before the window start -> 0.0, no crash.
    assert ao._bloom_amplitude(a, 0.5, next_time_sec=0.0, sample_rate=sr) == 0.0


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
