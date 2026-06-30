"""Deterministic-logic tests for the beat-tracker A/B harness.

Covers the pure pieces that don't need E-GMD data, madmom, or torch:
clip selection (stratified + deterministic), synthetic onset degradation
(ratios, determinism, stem bleed), and MIDI-derived ground truth.
"""
from __future__ import annotations

import collections
from pathlib import Path

import mido

from benchmarks.beat_ab import Clip, _score, _write_summary, select_clips
from benchmarks.beat_gt import (
    GtGrid,
    LaneOnset,
    gt_grid,
    lane_onsets,
    parse_time_sig,
    sanity_coverage,
)
from benchmarks.onset_synth import (
    DROP_FRAC,
    UNIFORM_FP_FRAC,
    stable_seed,
    synthesize_align_onsets,
)

# ---------- selection ----------

def _clip(track_id: str, bpm: float, ts: tuple[int, int], duration: float = 60.0) -> Clip:
    return Clip(track_id, Path(f"{track_id}.wav"), Path(f"{track_id}.midi"), bpm, ts, duration)


def _fake_clips() -> list[Clip]:
    clips: list[Clip] = []
    tempos = [70, 85, 100, 115, 130, 145, 160, 180]
    for i, bpm in enumerate(tempos * 8):  # 64 4/4 clips spread across bands
        clips.append(_clip(f"ff_{i:03d}", bpm, (4, 4)))
    odd = [(3, 4), (6, 8), (5, 4), (7, 8)]
    for i in range(40):  # 40 non-4/4 clips
        clips.append(_clip(f"nff_{i:03d}", tempos[i % len(tempos)], odd[i % len(odd)]))
    return clips


def test_selection_honours_count_and_nonfourfour_quota():
    sel = select_clips(_fake_clips(), n_total=48, nonfourfour_quota=12, min_bars=8,
                       min_per_meter=2)
    assert len(sel) == 48
    assert sum(not c.is_four_four for c in sel) == 12


def test_selection_guarantees_each_meter():
    sel = select_clips(_fake_clips(), n_total=48, nonfourfour_quota=16, min_per_meter=3)
    meters = collections.Counter(c.time_sig for c in sel if not c.is_four_four)
    for meter in [(3, 4), (6, 8), (5, 4), (7, 8)]:
        assert meters[meter] >= 3, (meter, meters)


def test_selection_includes_rare_meter_below_quota():
    # Only 2 clips of a rare meter exist; both must still be selected.
    clips = _fake_clips() + [_clip("rare0", 120, (7, 8)), _clip("rare1", 95, (7, 8))]
    # Drop the synthetic 7/8s from _fake_clips so (7,8) availability is exactly 2.
    clips = [c for c in clips if c.time_sig != (7, 8) or c.track_id.startswith("rare")]
    sel = select_clips(clips, n_total=48, nonfourfour_quota=16, min_per_meter=5)
    picked = {c.track_id for c in sel if c.time_sig == (7, 8)}
    assert picked == {"rare0", "rare1"}


def test_selection_is_deterministic():
    a = select_clips(_fake_clips(), n_total=48, nonfourfour_quota=12)
    b = select_clips(_fake_clips(), n_total=48, nonfourfour_quota=12)
    assert [c.track_id for c in a] == [c.track_id for c in b]


def test_selection_spreads_across_tempo_bands():
    sel = select_clips(_fake_clips(), n_total=48, nonfourfour_quota=12)
    bands = {c.band for c in sel}
    assert len(bands) >= 4  # all four configured bands represented


def test_selection_filters_short_clips():
    clips = _fake_clips() + [_clip("tiny", 120, (4, 4), duration=2.0)]
    sel = select_clips(clips, n_total=200, nonfourfour_quota=40, min_bars=8)
    assert "tiny" not in {c.track_id for c in sel}


# ---------- synthetic onsets ----------

def _onsets(n: int) -> list[LaneOnset]:
    lanes = ["kick", "snare", "hihat", "tom", "ride", "crash"]
    return [LaneOnset(time=i * 0.12, velocity=90, lane=lanes[i % len(lanes)]) for i in range(n)]


def test_synthetic_is_deterministic():
    o = _onsets(300)
    a = synthesize_align_onsets(o, stable_seed("track_x"), duration=40.0)
    b = synthesize_align_onsets(o, stable_seed("track_x"), duration=40.0)
    assert a == b


def test_synthetic_seed_varies_output():
    o = _onsets(300)
    a = synthesize_align_onsets(o, stable_seed("track_x"), duration=40.0)
    b = synthesize_align_onsets(o, stable_seed("track_y"), duration=40.0)
    assert a != b


def test_synthetic_recall_in_ballpark():
    o = _onsets(400)
    out = synthesize_align_onsets(o, stable_seed("r"), duration=60.0)
    true_times = {round(x.time, 6) for x in o}
    kept = sum(1 for t, _ in out if round(t, 6) in true_times)
    recall = kept / len(o)
    assert abs(recall - (1 - DROP_FRAC)) < 0.06


def test_synthetic_adds_false_positives():
    o = _onsets(400)
    out = synthesize_align_onsets(o, stable_seed("r"), duration=60.0)
    true_times = {round(x.time, 6) for x in o}
    kept = sum(1 for t, _ in out if round(t, 6) in true_times)
    # FP (uniform) + bleed ghosts mean strictly more outputs than kept trues.
    assert len(out) > kept
    extras = len(out) - kept
    # Loosely: at least the uniform-FP budget's worth of extras.
    assert extras >= UNIFORM_FP_FRAC * kept * 0.5


def test_synthetic_empty_input():
    assert synthesize_align_onsets([], stable_seed("e"), duration=10.0) == []


# ---------- ground truth ----------

def _write_midi(
    tmp: Path, tempo_bpm: float, notes: list[tuple[int, int]], ticks_per_beat: int = 480
) -> Path:
    """notes = [(pitch, beat_index)] placed on the quarter-note grid."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=int(60_000_000 / tempo_bpm), time=0))
    prev_tick = 0
    for pitch, beat in sorted(notes, key=lambda n: n[1]):
        tick = beat * ticks_per_beat
        tr.append(mido.Message("note_on", channel=9, note=pitch, velocity=100, time=tick - prev_tick))
        tr.append(mido.Message("note_off", channel=9, note=pitch, velocity=0, time=0))
        prev_tick = tick
    path = tmp / "x.midi"
    mid.save(str(path))
    return path


def test_gt_grid_four_four(tmp_path: Path):
    midi = _write_midi(tmp_path, 120.0, [(36, i) for i in range(8)])
    grid = gt_grid(midi, (4, 4), bpm=120.0)
    assert len(grid.beats) == 8
    # 120 BPM -> 0.5 s/beat.
    assert grid.beats == [round(b, 6) for b in grid.beats]
    assert abs(grid.beats[1] - 0.5) < 1e-6
    assert abs(grid.beats[-1] - 3.5) < 1e-6
    # Downbeats every 4 beats: t=0.0, 2.0.
    assert len(grid.downbeats) == 2
    assert abs(grid.downbeats[1] - 2.0) < 1e-6


def test_gt_grid_three_four(tmp_path: Path):
    midi = _write_midi(tmp_path, 120.0, [(36, i) for i in range(9)])
    grid = gt_grid(midi, (3, 4), bpm=120.0)
    # 9 beats -> downbeats at beats 0,3,6 = t 0,1.5,3.0.
    assert len(grid.downbeats) == 3
    assert abs(grid.downbeats[1] - 1.5) < 1e-6


def test_lane_onsets_maps_pitches(tmp_path: Path):
    midi = _write_midi(tmp_path, 120.0, [(36, 0), (38, 1), (42, 2), (45, 3), (51, 4), (49, 5)])
    onsets = lane_onsets(midi)
    assert [o.lane for o in onsets] == ["kick", "snare", "hihat", "tom", "ride", "crash"]


def test_sanity_coverage_high_when_on_grid(tmp_path: Path):
    midi = _write_midi(tmp_path, 120.0, [(36, i) for i in range(8)])
    grid = gt_grid(midi, (4, 4), bpm=120.0)
    onsets = lane_onsets(midi)
    assert sanity_coverage(grid, onsets) == 1.0


def test_sanity_coverage_low_when_off_grid():
    grid = GtGrid(beats=[0.0, 0.5, 1.0, 1.5], downbeats=[0.0], bpm=120.0, time_sig=(4, 4))
    onsets = [LaneOnset(time=b + 0.3, velocity=90, lane="kick") for b in grid.beats]
    assert sanity_coverage(grid, onsets) == 0.0


def test_parse_time_sig():
    assert parse_time_sig("4-4") == (4, 4)
    assert parse_time_sig("6-8") == (6, 8)


# ---------- scoring + report ----------

def _grid_8() -> GtGrid:
    beats = [round(i * 0.5, 6) for i in range(8)]
    downbeats = [beats[0], beats[4]]
    return GtGrid(beats=beats, downbeats=downbeats, bpm=120.0, time_sig=(4, 4))


def test_score_perfect_match():
    g = _grid_8()
    s = _score(g, g.beats, g.downbeats)
    assert s["downbeat_f"] == 1.0
    assert s["beat_f"] == 1.0
    assert s["amlt"] == 1.0
    assert s["tempo_err"] == 0.0
    assert s["bar_len_ok"] is True
    assert s["tempo_oct_within4"] is True


def test_score_empty_estimate_is_zero_not_crash():
    g = _grid_8()
    s = _score(g, [], [])
    assert s["downbeat_f"] == 0.0
    assert s["beat_f"] == 0.0
    assert s["bar_len_ok"] is False


def test_score_wrong_beats_per_bar_flags_bar_len():
    # 5/4 GT read as 4/4: detected bar spans 4 of the 5 beats -> ratio 0.8.
    g = GtGrid(beats=[i * 0.5 for i in range(20)], downbeats=[0.0, 2.5, 5.0, 7.5],
               bpm=120.0, time_sig=(5, 4))
    est_db = [0.0, 2.0, 4.0, 6.0, 8.0]  # every 4 beats
    s = _score(g, g.beats, est_db)
    assert s["bar_len_ok"] is False
    assert abs(s["bar_len_ratio"] - 0.8) < 0.01


def _fake_result(downbeat_f: float, bar_ok: bool) -> dict:
    return {
        "downbeat_f": downbeat_f, "bar_len_ok": bar_ok, "bar_len_ratio": 1.0 if bar_ok else 0.8,
        "tempo_oct_within4": True, "tempo_err": 1.0, "tempo_oct": 1.0,
        "amlt": downbeat_f, "beat_f": downbeat_f, "cmlt": downbeat_f,
        "est_bpm": 120.0, "gt_bpm": 120.0,
    }


def test_write_summary_smoke(tmp_path: Path):
    rows: list[dict] = []
    for i in range(8):
        rows.append({
            "track_id": f"t{i}", "band": "90-120", "time_sig": "4/4" if i % 2 else "3/4",
            "is_4_4": bool(i % 2), "sanity_cov": 1.0,
            "production": _fake_result(0.9, True),
        })
    _write_summary(tmp_path, "synthetic", rows)
    text = (tmp_path / "summary.md").read_text()
    assert "Beat-tracker A/B" in text
    assert "downbeat_f" in text and "bar_len_ok" in text
    assert "production" in text
    assert "Per time signature" in text
