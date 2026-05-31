"""Tests for the scoring orchestrator (`app.scoring.score_map`).

The audio reference (ADTOF / separation) is injected as a fake, so these
exercise the real assembly: raw score, global align, corrected score, and
the ParaDB/MIDI entry points end-to-end without torch or audio."""
from __future__ import annotations

import io
import json
import zipfile

import mido
import pytest

from app.scoring.score_map import score_midi, score_onsets, score_paradb


def test_score_onsets_perfect_is_100() -> None:
    chart = {"k": [0.0, 1.0, 2.0], "s": [0.5, 1.5]}
    result = score_onsets(chart, chart, separation_skipped=True)
    assert result.score == 100
    assert result.score_corrected == 100
    assert result.audio_reference == "drum_track"
    assert result.tempo_ratio == pytest.approx(1.0, abs=1e-3)


def test_score_onsets_offset_chart_corrected_beats_raw() -> None:
    audio = {"k": [1.0, 2.0, 3.0, 4.0]}
    chart = {"k": [t - 0.045 for t in audio["k"]]}  # 45 ms early
    result = score_onsets(chart, audio, separation_skipped=False)
    assert result.audio_reference == "separated"
    assert result.score_corrected > result.score
    assert result.offset_sec == pytest.approx(0.045, abs=5e-3)


def _rlrr_zip() -> bytes:
    doc = {
        "version": 0.7,
        "recordingMetadata": {"complexity": 4},
        "audioFileData": {"songTracks": [], "drumTracks": ["drums.ogg"]},
        "instruments": [{"name": "BP_Kick_C_1", "class": "BP_Kick_C"}],
        "events": [{"name": "BP_Kick_C_1", "vel": 100, "loc": 0, "time": 0.5}],
        "bpmEvents": [],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("chart_expert.rlrr", json.dumps(doc))
        z.writestr("drums.ogg", b"DRUMDATA")
    return buf.getvalue()


def test_score_paradb_uses_drum_track_and_scores(tmp_path) -> None:
    # Fake ADTOF: the drum audio's only onset matches the chart's kick.
    result = score_paradb(
        _rlrr_zip(), work_dir=tmp_path, detect=lambda path: {"k": [0.5]}
    )
    assert result.separation_skipped is True
    assert result.audio_reference == "drum_track"
    assert result.score_corrected == 100


def test_score_midi_separates_then_scores(tmp_path) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=480))
    mid.tracks.append(track)
    buf = io.BytesIO()
    mid.save(file=buf)

    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"x")
    produced = tmp_path / "drum_stem.wav"
    produced.write_bytes(b"y")

    class FakeStemsAll:
        drum_stem = produced

    class FakeSeparator:
        def run_stems_all(self, audio_path, work_dir):
            return FakeStemsAll()

    result = score_midi(
        buf.getvalue(),
        audio_path=audio,
        work_dir=tmp_path,
        separator=FakeSeparator(),
        detect=lambda path: {"k": [0.5]},  # matches the kick at beat 1 @120bpm
    )
    assert result.separation_skipped is False
    assert result.audio_reference == "separated"
    assert result.score_corrected == 100
