import mido
import numpy as np

import drumjot_training.clean as clean
from drumjot_training.egmd import EgmdClip


def _write_midi(path, notes):
    """notes: list of (gm_note, tick_delta)."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mid.add_track()
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    for note, dt in notes:
        tr.append(mido.Message("note_on", channel=9, note=note, velocity=100, time=dt))
    mid.save(path)


def _clip(midi_path):
    return EgmdClip(
        audio_path=midi_path.with_suffix(".wav"),
        midi_path=midi_path,
        split="train",
        duration=1.0,
        bpm=120.0,
    )


def test_exact_duplicate_is_dropped(tmp_path):
    a = tmp_path / "a.mid"
    _write_midi(a, [(36, 480), (38, 480)])
    b = tmp_path / "b.mid"
    _write_midi(b, [(36, 480), (38, 480)])  # identical content
    kept, dropped = clean.dedup_clips([_clip(a), _clip(b)])
    assert len(kept) == 1
    assert len(dropped) == 1


def test_distinct_patterns_both_kept(tmp_path):
    a = tmp_path / "a.mid"
    _write_midi(a, [(36, 480)])
    b = tmp_path / "b.mid"
    _write_midi(b, [(38, 480), (42, 240)])
    kept, dropped = clean.dedup_clips([_clip(a), _clip(b)])
    assert len(kept) == 2
    assert dropped == []


def test_dropped_records_carry_a_reason(tmp_path):
    a = tmp_path / "a.mid"
    _write_midi(a, [(36, 480)])
    b = tmp_path / "b.mid"
    _write_midi(b, [(36, 480)])
    _kept, dropped = clean.dedup_clips([_clip(a), _clip(b)])
    assert dropped[0][1] in {"exact", "near"}


def test_support_score_all_onsets_backed_by_a_transient():
    env = np.zeros(200)
    env[50] = 1.0  # 0.50 s @ 100 fps
    env[100] = 1.0  # 1.00 s
    r = clean.support_score({"k": [0.50], "s": [1.00]}, env, 100.0, support_floor=0.5)
    assert r["n_total"] == 2
    assert r["fraction"] == 1.0


def test_support_score_flags_unsupported_onset():
    env = np.zeros(200)
    env[50] = 1.0  # only a transient at 0.50 s
    r = clean.support_score({"k": [0.50], "s": [1.50]}, env, 100.0, support_floor=0.5)
    assert r["n_supported"] == 1
    assert r["fraction"] == 0.5


def test_support_score_empty_is_one():
    r = clean.support_score({"k": []}, np.zeros(50), 100.0, support_floor=0.5)
    assert r["fraction"] == 1.0


def test_filter_by_support_drops_low_scoring_clips():
    scores = {"a": 0.9, "b": 0.4, "c": 0.8}
    kept, dropped = clean.filter_by_support(["a", "b", "c"], lambda c: scores[c], min_support=0.5)
    assert kept == ["a", "c"]
    assert [c for c, _frac in dropped] == ["b"]
