import mido
import pytest

import drumjot_training.lanes as lanes
import drumjot_training.midi_labels as midi_labels


def _midi_120bpm() -> mido.MidiFile:
    """kick on beat 1 (0.5 s), snare on beat 2 (1.0 s) at 120 BPM."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mid.add_track()
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))  # 120 BPM
    tr.append(mido.Message("note_on", channel=9, note=36, velocity=100, time=480))
    tr.append(mido.Message("note_on", channel=9, note=38, velocity=100, time=480))
    return mid


def test_extracts_per_lane_onset_times():
    out = midi_labels.onsets_by_lane(_midi_120bpm())
    assert out["k"] == pytest.approx([0.5], abs=1e-3)
    assert out["s"] == pytest.approx([1.0], abs=1e-3)


def test_all_lanes_present_even_when_empty():
    out = midi_labels.onsets_by_lane(_midi_120bpm())
    assert set(out) == set(lanes.WEIGHT_LANES)  # output lanes + the `x` negative lane
    assert out["t"] == []
    assert out["hc"] == []
    assert out["mc"] == []
    assert out["x"] == []


def test_note_on_with_zero_velocity_is_ignored():
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mid.add_track()
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    tr.append(mido.Message("note_on", channel=9, note=36, velocity=0, time=480))
    assert midi_labels.onsets_by_lane(mid)["k"] == []


def test_non_kit_percussion_goes_to_negative_lane():
    # non-kit GM perc (note 60 = hi bongo) is kept as a hard negative, not a lane
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mid.add_track()
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    tr.append(mido.Message("note_on", channel=9, note=60, velocity=100, time=480))
    out = midi_labels.onsets_by_lane(mid)
    assert out["x"] == pytest.approx([0.5], abs=1e-3)
    assert all(v == [] for ln, v in out.items() if ln != "x")  # no OUTPUT lane fired


def test_truly_out_of_everything_note_is_dropped():
    # a note that is neither a kit lane nor mapped aux percussion drops entirely
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mid.add_track()
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    tr.append(mido.Message("note_on", channel=9, note=34, velocity=100, time=480))
    out = midi_labels.onsets_by_lane(mid)
    assert all(v == [] for v in out.values())
