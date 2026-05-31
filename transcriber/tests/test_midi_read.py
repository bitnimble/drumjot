"""Unit tests for `app.scoring.midi_read`: a MIDI file -> per-lane onset
seconds (tempo-aware, channel-9-with-fallback, GM fold)."""
from __future__ import annotations

import io

import mido
import pytest

from app.scoring.midi_read import onsets_from_midi_bytes

_TPB = 480


def _build_midi(tracks: list[list[mido.Message | mido.MetaMessage]]) -> bytes:
    mid = mido.MidiFile(ticks_per_beat=_TPB)
    for events in tracks:
        track = mido.MidiTrack()
        track.extend(events)
        mid.tracks.append(track)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _note(note: int, *, channel: int = 9, time: int = 0) -> mido.Message:
    return mido.Message("note_on", note=note, velocity=100, channel=channel, time=time)


def test_multitrack_tempo_change_applies_across_tracks() -> None:
    # Conductor track changes tempo from 120 to 60 BPM at beat 1; the drum
    # track's notes must honour it (the `mid.tracks`-per-track bug would
    # mis-time them at a constant 120 BPM). Kicks at ticks 0 / 480 / 960:
    # [0, 0.5 (1 beat @120), 1.5 (+1 beat @60)] seconds.
    conductor = [
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0),
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60), time=_TPB),
    ]
    drums = [_note(36, time=0), _note(36, time=_TPB), _note(36, time=_TPB)]
    chart = onsets_from_midi_bytes(_build_midi([conductor, drums]))
    assert chart.onsets_by_lane["k"] == pytest.approx([0.0, 0.5, 1.5])


def test_gm_notes_fold_to_lanes() -> None:
    drums = [
        _note(36, time=0),  # kick
        _note(38, time=10),  # snare
        _note(42, time=10),  # closed hat
        _note(49, time=10),  # crash -> cymbals
    ]
    chart = onsets_from_midi_bytes(_build_midi([drums]))
    assert set(chart.onsets_by_lane) == {"k", "s", "h", "cy"}


def test_prefers_channel_9_when_present() -> None:
    # A melodic note on ch0 at the same pitch as a drum should be ignored
    # when channel-9 drum notes exist.
    drums = [_note(36, channel=9, time=0), _note(38, channel=0, time=10)]
    chart = onsets_from_midi_bytes(_build_midi([drums]))
    assert "k" in chart.onsets_by_lane
    assert "s" not in chart.onsets_by_lane  # ch0 note ignored
    assert chart.used_all_channels is False


def test_falls_back_to_all_channels_without_channel_9() -> None:
    drums = [_note(36, channel=0, time=0), _note(38, channel=0, time=10)]
    chart = onsets_from_midi_bytes(_build_midi([drums]))
    assert set(chart.onsets_by_lane) == {"k", "s"}
    assert chart.used_all_channels is True


def test_unmapped_notes_counted() -> None:
    drums = [_note(36, time=0), _note(39, time=10)]  # 39 = hand clap, no lane
    chart = onsets_from_midi_bytes(_build_midi([drums]))
    assert chart.onsets_by_lane.get("k") == pytest.approx([0.0])
    assert chart.unmapped_notes == 1


def test_no_drum_mapped_notes_raises() -> None:
    drums = [_note(39, time=0), _note(54, time=10)]  # clap + tambourine: no lanes
    with pytest.raises(ValueError, match="drum"):
        onsets_from_midi_bytes(_build_midi([drums]))
