import mido

from drumjot_training import a2md


def _midi(notes, tmp_path):
    """notes: list of (channel, gm_note). Save a 1-track MIDI, return its path."""
    mf = mido.MidiFile()
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    for ch, note in notes:
        tr.append(mido.Message("note_on", note=note, velocity=100, channel=ch, time=20))
        tr.append(mido.Message("note_off", note=note, velocity=0, channel=ch, time=10))
    p = tmp_path / "x.mid"
    mf.save(str(p))
    return p


def test_drum_onsets_only_from_percussion_channel(tmp_path):
    # snare(38) on the drum channel(9) + a melodic 38 on channel 0; only the drum one counts
    p = _midi([(9, 38), (0, 38), (9, 42)], tmp_path)
    out = a2md.drum_onsets_by_lane(p)
    assert len(out["s"]) == 1    # the channel-0 melodic D2 is NOT a snare
    assert len(out["hc"]) == 1   # channel-9 closed hat (42)


def test_restricted_onsets_keeps_only_the_stems_lanes(tmp_path):
    p = _midi([(9, 42), (9, 46), (9, 51), (9, 36)], tmp_path)  # closed-hat, open-hat, ride, kick
    h = a2md.restricted_onsets(p, "h")
    assert h["hc"] and h["ho"] and not h["rd"] and not h["k"]
    c = a2md.restricted_onsets(p, "c")
    assert c["rd"] and not c["hc"]


def test_distinguishes_hat_and_cymbal_articulations(tmp_path):
    # 42 closed, 44 pedal, 46 open hat; 49 crash, 51 ride -> all land in distinct lanes
    p = _midi([(9, 42), (9, 44), (9, 46), (9, 49), (9, 51)], tmp_path)
    out = a2md.drum_onsets_by_lane(p)
    assert out["hc"] and out["hp"] and out["ho"] and out["cr"] and out["rd"]
