"""Characterization (golden) tests pinning the CURRENT beats.py behaviour.

These exist to make the beats.py module split a provably behaviour-preserving
refactor: every golden value below was captured by running the pre-split code
on the synthetic inputs, so a post-split value that diverges is a regression.

All inputs are hand-built numpy / python beat, downbeat, onset and accent
arrays -- no audio, no model, no GPU -- so the whole file is fast and CPU-only.
It covers the subtle paths the review flagged: 3/4-vs-6/8 meter choice,
double-time downbeat decimation, autocorr meter recovery, downbeat smoothing
edge cases, tempo segmentation (constant / ramp / step / localized), drift, and
the coarse+fine chart<->audio alignment offset.

Everything imports from the public `app.pipeline.beats` facade, so the same
suite runs unchanged against the module-split layout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.pipeline.beats import (
    BarInfo,
    BeatStructure,
    BeatTick,
    _bar_length_from_autocorr,
    _beats_downbeats_to_raw,
    _best_downbeat_phase,
    _choose_time_signature,
    _coarse_offset_from_envelope,
    _dominant_gap_fraction,
    _finalize_bar,
    _finalize_bar_tempos,
    _pad_trailing_bars,
    _raw_to_structure,
    _recover_bar_length_if_incoherent,
    _summarize,
    align_beats_to_onsets,
    candidates_with_beat_positions,
    detect_feel_for_bars,
    summarize_bar_for_prompt,
)

# ---------- helpers (mirror the synthetic constructions used to capture golden) ----------

_R = 6  # rounding digits used when the golden values were captured


def _r(x: float, n: int = _R) -> float:
    return round(float(x), n)


def _ramp_beat_times(b0: float, b1: float, n_beats: int) -> list[float]:
    L = float(n_beats)
    times: list[float] = []
    for d in range(n_beats + 1):
        if abs(b1 - b0) < 1e-9:
            times.append(d * 60.0 / b0)
        else:
            bpm_d = (b0 * b0 + (b1 * b1 - b0 * b0) * (d / L)) ** 0.5
            times.append(120.0 * L * (bpm_d - b0) / (b1 * b1 - b0 * b0))
    return times


def _quantize(times: list[float], fps: float = 43.07) -> list[float]:
    return [round(t * fps) / fps for t in times]


def _concat_ramp_blocks(blocks: list[tuple[float, float, int]]) -> list[float]:
    times = [0.0]
    cursor = 0.0
    for b0, b1, nb in blocks:
        seg = _ramp_beat_times(b0, b1, nb)
        for t in seg[1:]:
            times.append(cursor + t)
        cursor = times[-1]
    return times


def _structure_from_beat_times(times: list[float], count: int = 4) -> BeatStructure:
    beats = [
        BeatTick(time=t, beat_in_bar=(i % count) + 1, bar_index=i // count)
        for i, t in enumerate(times)
    ]
    n_bars = (len(times) + count - 1) // count
    bars = [
        _finalize_bar(b, beats[b * count:(b + 1) * count])
        for b in range(n_bars)
        if beats[b * count:(b + 1) * count]
    ]
    return _summarize(beats, bars)


def _structure_from_bars(bar_lengths: list[int], spb: float = 0.5) -> BeatStructure:
    beats: list[float] = []
    downbeats: list[float] = []
    t = 0.0
    for n in bar_lengths:
        downbeats.append(round(t, 3))
        for _ in range(n):
            beats.append(round(t, 3))
            t += spb
    return _raw_to_structure(_beats_downbeats_to_raw(beats, downbeats))


def _structure_from_bars_dur(bars: list[tuple[int, float]]) -> BeatStructure:
    beats: list[float] = []
    downbeats: list[float] = []
    t = 0.0
    for n, dur in bars:
        downbeats.append(round(t, 4))
        step = dur / n
        for _ in range(n):
            beats.append(round(t, 4))
            t += step
    return _raw_to_structure(_beats_downbeats_to_raw(beats, downbeats))


def _frame_quantized_constant(
    tempo_bpm: float, fps: float, n_bars: int, count: int = 4
) -> BeatStructure:
    beat_gap = 60.0 / tempo_bpm
    beats: list[BeatTick] = []
    for i in range(n_bars * count):
        q_t = round((i * beat_gap) * fps) / fps
        beats.append(BeatTick(time=q_t, beat_in_bar=(i % count) + 1, bar_index=i // count))
    bars = [_finalize_bar(b, beats[b * count:(b + 1) * count]) for b in range(n_bars)]
    return _summarize(beats, bars)


def _accent(n: int, period: int, phase: int = 0, strong: float = 1.0,
            weak: float = 0.2) -> np.ndarray:
    s = np.full(n, weak, dtype=np.float64)
    s[np.arange(n) % period == phase] = strong
    return s


def _pulse_env(frame_times, centers, width=0.01, height=1.0):
    env = np.zeros_like(frame_times)
    for c in centers:
        env += height * np.exp(-((frame_times - c) ** 2) / (2 * width**2))
    return env


def _make_bar(index: int, start: float, beat_gap: float, count: int = 4) -> BarInfo:
    beats = [
        BeatTick(time=start + i * beat_gap, beat_in_bar=i + 1, bar_index=index)
        for i in range(count)
    ]
    return BarInfo(index=index, start_time=start, end_time=start + count * beat_gap,
                   beats=beats, time_signature=(count, 4), tempo_bpm=60.0 / beat_gap)


def _ts_bars(counts: list[int], beat_gap: float = 0.5) -> list[BarInfo]:
    bars: list[BarInfo] = []
    t = 0.0
    for i, c in enumerate(counts):
        bars.append(_make_bar(i, t, beat_gap, count=c))
        t += c * beat_gap
    return bars


def _wandering(amp: float, period: int, n: int) -> list[float]:
    return [0.5 * i + amp * float(np.sin(2.0 * np.pi * i / period)) for i in range(n)]


def _seg_summary(s: BeatStructure) -> list[dict]:
    return [
        {"start_beat": seg.start_beat, "end_beat": seg.end_beat,
         "start_bpm": _r(seg.start_bpm), "end_bpm": _r(seg.end_bpm),
         "is_ramp": seg.is_ramp()}
        for seg in s.tempo_segments
    ]


def _beat_times(s: BeatStructure) -> list[float]:
    return [_r(b.time) for b in s.beats]


def _build_44(bpm: float, n_bars: int) -> tuple[BeatStructure, float]:
    gap = 60.0 / bpm
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    t = 0.0
    for b_idx in range(n_bars):
        bb = [BeatTick(time=t + j * gap, beat_in_bar=j + 1, bar_index=b_idx) for j in range(4)]
        beats.extend(bb)
        bars.append(_finalize_bar(b_idx, bb))
        t += 4 * gap
    return _summarize(beats, bars), gap


# ---------- golden reference values (captured from pre-split beats.py) ----------

GOLDEN: dict = {
    "choose_ts": {
        "4,120.0": [4, 4], "3,80.0": [3, 4], "5,140.0": [5, 4], "7,100.0": [7, 4],
        "6,70.0": [6, 4], "6,95.0": [6, 4], "6,100.0": [6, 8], "6,160.0": [6, 8],
        "12,130.0": [12, 8], "12,80.0": [12, 4], "2,90.0": [2, 4], "9,200.0": [9, 4],
    },
    "bar_len_autocorr": {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9},
    "bar_len_fundamental_6_to_3": 3,
    "bar_len_no_demote_4_to_2": 4,
    "bar_len_pure_six": 6,
    "bar_len_flat_none": None,
    "bar_len_too_few": None,
    "best_phase_5_2": 2,
    "dom_gap_coherent": 1.0,
    "dom_gap_scattered": 0.3,
    "recovery_coherent_out": [float(x) for x in range(0, 40, 2)],
    "smooth": {
        "merged_8": {"sigs": [[4, 4]] * 6, "has_ts_changes": False, "nbeats": 24},
        "merged_6_in_3": {"sigs": [[3, 4]] * 6, "has_ts_changes": False, "nbeats": 18},
        "frag_2_2": {"sigs": [[4, 4]] * 5, "has_ts_changes": False, "nbeats": 20},
        "frag_1_3": {"sigs": [[4, 4]] * 5, "has_ts_changes": False, "nbeats": 20},
        "sustained_34": {
            "sigs": [[4, 4], [4, 4], [3, 4], [3, 4], [3, 4], [4, 4], [4, 4]],
            "has_ts_changes": True, "nbeats": 25,
        },
        "genuine_68": {
            "sigs": [[4, 4], [4, 4], [6, 8], [4, 4], [4, 4]],
            "has_ts_changes": False, "nbeats": 22,
        },
        "no_majority": {
            "sigs": [[4, 4], [3, 4], [5, 4], [4, 4], [3, 4]],
            "has_ts_changes": True, "nbeats": 19,
        },
    },
    "smooth_dur": {
        "decimate_3": {"sigs": [[3, 4]] * 5, "has_ts_changes": False},
        "split_merged_dur": {"sigs": [[3, 4]] * 6, "has_ts_changes": False},
        "decimate_4": {"sigs": [[4, 4]] * 5, "has_ts_changes": False},
    },
    "pickup": {"nbeats": 14, "bar0_first_time": 0.0,
               "downbeat_times": [0.0, 1.0, 3.0, 5.0]},
    "far_downbeat_raw_pos": [1, 2, 3, 4],
    "summary_modal": {"its": [4, 4], "has": False},
    "summary_sustained": {"its": [4, 4], "has": True},
    "summary_robust_tempo": 120.0,
    "const_song": {
        "segments": [{"start_beat": 0, "end_beat": 63, "start_bpm": 119.996159,
                      "end_bpm": 119.996159, "is_ramp": False}],
        "has_tempo_changes": False, "initial_tempo": 119.996,
        "beat_times": [
            -0.000672, 0.499344, 0.99936, 1.499376, 1.999392, 2.499408, 2.999424,
            3.49944, 3.999456, 4.499472, 4.999488, 5.499504, 5.99952, 6.499536,
            6.999552, 7.499568, 7.999584, 8.4996, 8.999616, 9.499633, 9.999649,
            10.499665, 10.999681, 11.499697, 11.999713, 12.499729, 12.999745,
            13.499761, 13.999777, 14.499793, 14.999809, 15.499825, 15.999841,
            16.499857, 16.999873, 17.499889, 17.999905, 18.499921, 18.999937,
            19.499953, 19.999969, 20.499985, 21.000001, 21.500017, 22.000033,
            22.500049, 23.000065, 23.500081, 24.000097, 24.500113, 25.000129,
            25.500145, 26.000161, 26.500177, 27.000193, 27.500209, 28.000225,
            28.500241, 29.000257, 29.500273, 30.000289, 30.500305, 31.000321,
            31.500337,
        ],
    },
    "const_100_segments": [{"start_beat": 0, "end_beat": 47, "start_bpm": 99.98656,
                            "end_bpm": 99.98656, "is_ramp": False}],
    "single_ramp": {
        "segments": [{"start_beat": 0, "end_beat": 64, "start_bpm": 109.948711,
                      "end_bpm": 140.022896, "is_ramp": True}],
        "has_tempo_changes": True,
    },
    "ramp_100_150_segments": [{"start_beat": 0, "end_beat": 48, "start_bpm": 99.981381,
                               "end_bpm": 150.002967, "is_ramp": True}],
    "localized_ramp": {
        "segments": [
            {"start_beat": 0, "end_beat": 33, "start_bpm": 117.474724,
             "end_bpm": 124.557109, "is_ramp": True},
            {"start_beat": 34, "end_beat": 60, "start_bpm": 139.64615,
             "end_bpm": 140.334638, "is_ramp": True},
        ],
        "has_tempo_changes": True,
    },
    "two_accel_segments": [
        {"start_beat": 0, "end_beat": 27, "start_bpm": 124.270539,
         "end_bpm": 136.172663, "is_ramp": True},
        {"start_beat": 28, "end_beat": 56, "start_bpm": 128.352095,
         "end_bpm": 138.446276, "is_ramp": True},
    ],
    "hard_step": {
        "segments": [
            {"start_beat": 0, "end_beat": 23, "start_bpm": 120.034715,
             "end_bpm": 120.034715, "is_ramp": False},
            {"start_beat": 24, "end_beat": 48, "start_bpm": 140.035219,
             "end_bpm": 140.035219, "is_ramp": False},
        ],
        "has_tempo_changes": True,
    },
    "step_refine_segments": [
        {"start_beat": 0, "end_beat": 15, "start_bpm": 120.0, "end_bpm": 120.0,
         "is_ramp": False},
        {"start_beat": 16, "end_beat": 32, "start_bpm": 150.0, "end_bpm": 150.0,
         "is_ramp": False},
    ],
    "step_refine_quantized_segments": [
        {"start_beat": 0, "end_beat": 19, "start_bpm": 126.034864,
         "end_bpm": 126.034864, "is_ramp": False},
        {"start_beat": 20, "end_beat": 40, "start_bpm": 95.960126,
         "end_bpm": 95.960126, "is_ramp": False},
    ],
    "genuine_accel": {
        "has_tempo_changes": True,
        "bars_tempo": [97.065, 103.276, 109.134, 114.693, 119.995, 125.071,
                       129.95, 134.651, 139.194, 143.593, 147.861, 152.009],
        "segments": [{"start_beat": 0, "end_beat": 47, "start_bpm": 94.825025,
                      "end_bpm": 153.462926, "is_ramp": True}],
    },
    "drift_const": [0.0] * 16,
    "drift_wander": [
        -0.014852, 0.001426, 0.025944, 0.031056, 0.031056, 0.011651, -0.015793,
        -0.043236, -0.062642, -0.062642, -0.057529, -0.033011, -0.000455,
        0.032101, 0.056619, 0.061732, 0.061732, 0.042326, 0.014883, -0.012561,
        -0.031966, -0.031966, -0.026854, -0.014595,
    ],
    "drift_long_bar": {
        "has_tempo_changes": False,
        "drift": [0.015035, 0.01097, 0.00284, -0.00529, -0.013419, -0.013419,
                  0.022192, 0.022192, 0.014062, 0.005932, -0.002197, -0.006262],
        "recon_start": [-0.004065, 2.0, 4.0, 6.0, 8.0, 10.00813, 12.05187, 14.06,
                        16.06, 18.06, 20.06, 22.064065],
    },
    "pad": {
        "nbars": 5,
        "beat_times": [
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
            7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
        ],
    },
    "pad_position_525": [2, 3.5],
    "pad_position_50": None,
    "coarse_known": 0.1,
    "coarse_smaller": 0.05,
    "coarse_flat": 0.0,
    "align_late": {
        "beat_times": [-0.0, 0.375, 0.75, 1.125], "bar0_start": -0.0,
        "bar0_tempo": 160.0, "align_offset": -0.03, "align_fine": -0.03,
    },
    "align_strongest": 1.02,
    "align_no_nearby": 2.0,
    "align_humanized": {
        "beat_times": [-0.0005, 0.3745, 0.7495, 1.1245, 1.4995, 1.8745, 2.2495,
                       2.6245],
        "bar0_tempo": 160.0, "bar1_tempo": 160.0,
    },
    "feel_straight16": ["straight16"] * 4,
    "feel_triplet": ["triplet"] * 4,
    "feel_sparse": ["sparse"] * 4,
    "candidates": {
        "k": [{"time": 0.0, "bar": 0, "beat": 1.0, "strength": 1.0},
              {"time": 0.5, "bar": 0, "beat": 2.0, "strength": 0.5},
              {"time": 1.0, "bar": 0, "beat": 3.0, "strength": 0.7}],
        "s": [{"time": 0.5, "bar": 0, "beat": 2.0, "strength": 0.9}],
    },
    "summarize_bar": {"bar": 0, "time_signature": "4/4", "tempo_bpm": 120.0,
                      "feel": "straight16", "start_time": 0.0},
}


# ---------- time-signature choice (3/4 vs 6/8, 12/8) ----------


@pytest.mark.parametrize("count,tempo", [
    (4, 120.0), (3, 80.0), (5, 140.0), (7, 100.0), (6, 70.0), (6, 95.0),
    (6, 100.0), (6, 160.0), (12, 130.0), (12, 80.0), (2, 90.0), (9, 200.0),
])
def test_choose_time_signature(count: int, tempo: float) -> None:
    key = f"{count},{tempo}"
    assert list(_choose_time_signature(count, tempo)) == GOLDEN["choose_ts"][key]


# ---------- autocorr meter recovery ----------


@pytest.mark.parametrize("bl", [2, 3, 4, 5, 6, 7, 8, 9])
def test_bar_length_autocorr(bl: int) -> None:
    assert _bar_length_from_autocorr(_accent(300, bl)) == GOLDEN["bar_len_autocorr"][str(bl)]


def test_bar_length_fundamental_6_to_3() -> None:
    n = 300
    s = np.full(n, 0.1)
    s[np.arange(n) % 6 == 0] = 1.0
    s[np.arange(n) % 6 == 3] = 0.95
    assert _bar_length_from_autocorr(s) == GOLDEN["bar_len_fundamental_6_to_3"]


def test_bar_length_no_demote_4_to_2() -> None:
    n = 300
    s = np.full(n, 0.15)
    s[np.arange(n) % 4 == 0] = 1.0
    s[np.arange(n) % 4 == 2] = 0.7
    assert _bar_length_from_autocorr(s) == GOLDEN["bar_len_no_demote_4_to_2"]


def test_bar_length_pure_six() -> None:
    assert _bar_length_from_autocorr(_accent(300, 6)) == GOLDEN["bar_len_pure_six"]


def test_bar_length_flat_none() -> None:
    assert _bar_length_from_autocorr(np.full(200, 0.5)) is GOLDEN["bar_len_flat_none"]


def test_bar_length_too_few() -> None:
    assert _bar_length_from_autocorr(_accent(15, 5)) is GOLDEN["bar_len_too_few"]


def test_best_downbeat_phase() -> None:
    assert _best_downbeat_phase(_accent(300, 5, phase=2), 5) == GOLDEN["best_phase_5_2"]


def test_dominant_gap_fraction_coherent() -> None:
    bts = np.arange(0, 40.0, 0.5)
    assert _r(_dominant_gap_fraction(bts, bts[::4])) == GOLDEN["dom_gap_coherent"]


def test_dominant_gap_fraction_scattered() -> None:
    bts = np.arange(0, 40.0, 0.5)
    idx = [0, 3, 4, 8, 9, 10, 14, 20, 22, 27, 35]
    assert _r(_dominant_gap_fraction(bts, bts[idx])) == GOLDEN["dom_gap_scattered"]


def test_recovery_leaves_coherent_untouched() -> None:
    bts = np.arange(0, 40.0, 0.5)
    out = _recover_bar_length_if_incoherent(bts, bts[::4], Path("/no/such"))
    assert [_r(x) for x in np.asarray(out)] == GOLDEN["recovery_coherent_out"]


# ---------- downbeat smoothing (merge / split / decimate / preserve) ----------


@pytest.mark.parametrize("name,bar_lengths", [
    ("merged_8", [4, 4, 8, 4, 4]),
    ("merged_6_in_3", [3, 3, 6, 3, 3]),
    ("frag_2_2", [4, 4, 2, 2, 4, 4]),
    ("frag_1_3", [4, 4, 1, 3, 4, 4]),
    ("sustained_34", [4, 4, 3, 3, 3, 4, 4]),
    ("genuine_68", [4, 4, 6, 4, 4]),
    ("no_majority", [4, 3, 5, 4, 3]),
])
def test_downbeat_smoothing(name: str, bar_lengths: list[int]) -> None:
    s = _structure_from_bars(bar_lengths)
    exp = GOLDEN["smooth"][name]
    assert [list(b.time_signature) for b in s.bars] == exp["sigs"]
    assert s.has_time_sig_changes is exp["has_ts_changes"]
    assert len(s.beats) == exp["nbeats"]


@pytest.mark.parametrize("name,bars", [
    ("decimate_3", [(3, 1.5), (3, 1.5), (6, 1.5), (3, 1.5), (3, 1.5)]),
    ("split_merged_dur", [(3, 1.5), (3, 1.5), (6, 3.0), (3, 1.5), (3, 1.5)]),
    ("decimate_4", [(4, 2.0), (4, 2.0), (8, 2.0), (4, 2.0), (4, 2.0)]),
])
def test_downbeat_smoothing_by_duration(name: str, bars: list[tuple[int, float]]) -> None:
    s = _structure_from_bars_dur(bars)
    exp = GOLDEN["smooth_dur"][name]
    assert [list(b.time_signature) for b in s.bars] == exp["sigs"]
    assert s.has_time_sig_changes is exp["has_ts_changes"]


def test_beats_downbeats_raw_leading_pickup() -> None:
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    downbeats = [1.0, 3.0, 5.0]
    s = _raw_to_structure(_beats_downbeats_to_raw(beats, downbeats))
    exp = GOLDEN["pickup"]
    assert len(s.beats) == exp["nbeats"]
    assert _r(s.bars[0].beats[0].time) == exp["bar0_first_time"]
    assert [_r(b.time) for b in s.beats if b.beat_in_bar == 1] == exp["downbeat_times"]


def test_beats_downbeats_raw_drops_far_downbeat() -> None:
    raw = _beats_downbeats_to_raw([0.0, 0.5, 1.0, 1.5], [0.0, 1.06])
    assert [int(r[1]) for r in raw] == GOLDEN["far_downbeat_raw_pos"]


# ---------- robust global summary ----------


def test_summary_modal_meter() -> None:
    s = _summarize([], _ts_bars([4, 2, 4, 4, 4, 4]))
    assert list(s.initial_time_signature) == GOLDEN["summary_modal"]["its"]
    assert s.has_time_sig_changes is GOLDEN["summary_modal"]["has"]


def test_summary_sustained_meter() -> None:
    s = _summarize([], _ts_bars([4, 4, 4, 3, 3, 3, 4, 4]))
    assert list(s.initial_time_signature) == GOLDEN["summary_sustained"]["its"]
    assert s.has_time_sig_changes is GOLDEN["summary_sustained"]["has"]


def test_summary_robust_initial_tempo() -> None:
    bars = _ts_bars([4, 4, 4, 4, 4])
    bars[1].tempo_bpm = 600.0
    s = _summarize([], bars)
    assert _r(s.initial_tempo, 3) == GOLDEN["summary_robust_tempo"]


# ---------- tempo segmentation / regularization ----------


def test_constant_song_segments_and_beats() -> None:
    s = _frame_quantized_constant(120.0, 43.07, 16)
    _finalize_bar_tempos(s)
    exp = GOLDEN["const_song"]
    assert _seg_summary(s) == exp["segments"]
    assert s.has_tempo_changes is exp["has_tempo_changes"]
    assert _r(s.initial_tempo, 3) == exp["initial_tempo"]
    assert _beat_times(s) == exp["beat_times"]


def test_constant_100_segments() -> None:
    s = _frame_quantized_constant(100.0, 43.07, 12)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["const_100_segments"]


def test_single_ramp_segments() -> None:
    times = _quantize(_ramp_beat_times(110.0, 140.0, 64))
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["single_ramp"]["segments"]
    assert s.has_tempo_changes is GOLDEN["single_ramp"]["has_tempo_changes"]


def test_ramp_100_150_segments() -> None:
    times = _quantize(_ramp_beat_times(100.0, 150.0, 48))
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["ramp_100_150_segments"]


def test_localized_ramp_segments() -> None:
    times = _quantize(_concat_ramp_blocks(
        [(120.0, 120.0, 24), (120.0, 140.0, 12), (140.0, 140.0, 24)]))
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["localized_ramp"]["segments"]
    assert s.has_tempo_changes is GOLDEN["localized_ramp"]["has_tempo_changes"]


def test_two_accelerandos_segments() -> None:
    times = _quantize(_concat_ramp_blocks(
        [(112.0, 132.0, 8), (132.0, 132.0, 40), (132.0, 152.0, 8)]))
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["two_accel_segments"]


def test_hard_step_segments() -> None:
    times = _quantize(_concat_ramp_blocks([(120.0, 120.0, 24), (140.0, 140.0, 24)]))
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["hard_step"]["segments"]
    assert s.has_tempo_changes is GOLDEN["hard_step"]["has_tempo_changes"]


def test_step_boundary_refined() -> None:
    times = _concat_ramp_blocks([(120.0, 120.0, 16), (150.0, 150.0, 16)])
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["step_refine_segments"]


def test_step_boundary_refined_quantized() -> None:
    times = _quantize(_concat_ramp_blocks([(126.0, 126.0, 20), (96.0, 96.0, 20)]))
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    assert _seg_summary(s) == GOLDEN["step_refine_quantized_segments"]


def test_genuine_accelerando_preserved() -> None:
    beats: list[BeatTick] = []
    bars: list[BarInfo] = []
    t = 0.0
    for b_idx in range(12):
        gap = 60.0 / (100.0 + b_idx * 5.0)
        bb = [BeatTick(time=t + j * gap, beat_in_bar=j + 1, bar_index=b_idx)
              for j in range(4)]
        beats.extend(bb)
        bars.append(_finalize_bar(b_idx, bb))
        t += 4 * gap
    s = _summarize(beats, bars)
    _finalize_bar_tempos(s)
    exp = GOLDEN["genuine_accel"]
    assert s.has_tempo_changes is exp["has_tempo_changes"]
    assert [_r(b.tempo_bpm, 3) for b in s.bars] == exp["bars_tempo"]
    assert _seg_summary(s) == exp["segments"]


# ---------- drift ----------


def test_drift_constant_is_zero() -> None:
    s = _frame_quantized_constant(120.0, 100.0, 16)
    _finalize_bar_tempos(s)
    assert [_r(b.drift_sec) for b in s.bars] == GOLDEN["drift_const"]


def test_drift_wander_captured() -> None:
    s = _structure_from_beat_times(_wandering(0.06, 48, 96))
    _finalize_bar_tempos(s)
    assert [_r(b.drift_sec) for b in s.bars] == GOLDEN["drift_wander"]


def test_drift_sub_threshold_long_bar() -> None:
    times: list[float] = []
    t = 0.0
    for i in range(48):
        times.append(t)
        t += 0.5
        if i == 23:
            t += 0.06
    s = _structure_from_beat_times(times)
    _finalize_bar_tempos(s)
    exp = GOLDEN["drift_long_bar"]
    assert s.has_tempo_changes is exp["has_tempo_changes"]
    assert [_r(b.drift_sec) for b in s.bars] == exp["drift"]
    assert [_r(b.start_time + b.drift_sec) for b in s.bars] == exp["recon_start"]


# ---------- trailing-bar padding + position() ----------


def test_pad_trailing_bars() -> None:
    bar = _make_bar(0, 0.0, 0.5)
    s = BeatStructure(beats=list(bar.beats), bars=[bar], initial_tempo=120.0,
                      initial_time_signature=(4, 4))
    _pad_trailing_bars(s, 10.0)
    exp = GOLDEN["pad"]
    assert len(s.bars) == exp["nbars"]
    assert _beat_times(s) == exp["beat_times"]
    pos = s.position(5.25)
    assert [pos[0], _r(pos[1])] == GOLDEN["pad_position_525"]
    assert s.position(50.0) is GOLDEN["pad_position_50"]


# ---------- coarse envelope alignment ----------


def test_coarse_offset_known_phase() -> None:
    ft = np.arange(0.0, 10.0, 0.002)
    bts = np.array([0.5 * k for k in range(1, 18)], dtype=float)
    env = _pulse_env(ft, bts + 0.1)
    off = _coarse_offset_from_envelope(bts, env, ft, max_shift=1.0, step=0.002,
                                       center_penalty=0.15, prominence=1.10)
    assert _r(off) == GOLDEN["coarse_known"]


def test_coarse_offset_smaller_shift() -> None:
    ft = np.arange(0.0, 3.0, 0.002)
    env = _pulse_env(ft, [1.05, 1.45])
    off = _coarse_offset_from_envelope(np.array([1.0]), env, ft, max_shift=0.5,
                                       step=0.002, center_penalty=0.15, prominence=1.10)
    assert _r(off) == GOLDEN["coarse_smaller"]


def test_coarse_offset_flat_envelope() -> None:
    ft = np.arange(0.0, 5.0, 0.002)
    env = np.ones_like(ft)
    off = _coarse_offset_from_envelope(np.array([0.5 * k for k in range(1, 10)]), env,
                                       ft, max_shift=1.0, step=0.002,
                                       center_penalty=0.15, prominence=1.10)
    assert _r(off) == GOLDEN["coarse_flat"]


# ---------- fine onset alignment ----------


def test_align_late_beats() -> None:
    beat_gap = 60.0 / 160.0
    beats = [BeatTick(time=i * beat_gap + 0.030, beat_in_bar=i + 1, bar_index=0)
             for i in range(4)]
    bar = BarInfo(index=0, start_time=beats[0].time, end_time=beats[-1].time + beat_gap,
                  beats=list(beats), time_signature=(4, 4), tempo_bpm=60.0 / beat_gap)
    s = BeatStructure(beats=list(beats), bars=[bar])
    align_beats_to_onsets(s, [(i * beat_gap, 10.0) for i in range(4)], max_distance=0.05)
    exp = GOLDEN["align_late"]
    assert _beat_times(s) == exp["beat_times"]
    assert _r(s.bars[0].start_time) == exp["bar0_start"]
    assert _r(s.bars[0].tempo_bpm, 3) == exp["bar0_tempo"]
    assert _r(s.align_offset_sec) == exp["align_offset"]
    assert _r(s.align_fine_offset_sec) == exp["align_fine"]


def test_align_strongest_not_closest() -> None:
    beat = BeatTick(time=1.000, beat_in_bar=1, bar_index=0)
    bar = BarInfo(index=0, start_time=1.000, end_time=1.500, beats=[beat],
                  time_signature=(4, 4), tempo_bpm=120.0)
    s = BeatStructure(beats=[beat], bars=[bar])
    align_beats_to_onsets(s, [(0.995, 0.1), (1.020, 5.0)], max_distance=0.05)
    assert _r(s.beats[0].time) == GOLDEN["align_strongest"]


def test_align_no_nearby_onset() -> None:
    beat = BeatTick(time=2.000, beat_in_bar=1, bar_index=0)
    bar = BarInfo(index=0, start_time=2.000, end_time=2.500, beats=[beat],
                  time_signature=(4, 4), tempo_bpm=120.0)
    s = BeatStructure(beats=[beat], bars=[bar])
    align_beats_to_onsets(s, [(1.800, 5.0), (2.200, 5.0)], max_distance=0.05)
    assert _r(s.beats[0].time) == GOLDEN["align_no_nearby"]


def test_align_humanized_preserves_per_bar_tempo() -> None:
    beat_gap = 60.0 / 160.0
    beats = [BeatTick(time=i * beat_gap, beat_in_bar=(i % 4) + 1, bar_index=i // 4)
             for i in range(8)]
    bars = []
    for b_idx in range(2):
        bb = beats[b_idx * 4:(b_idx + 1) * 4]
        bars.append(BarInfo(index=b_idx, start_time=bb[0].time,
                            end_time=bb[-1].time + beat_gap, beats=list(bb),
                            time_signature=(4, 4), tempo_bpm=160.0))
    s = BeatStructure(beats=list(beats), bars=bars)
    jitter = [0.012, -0.009, 0.014, -0.011, 0.010, -0.013, 0.008, -0.012]
    align_beats_to_onsets(s, [(i * beat_gap + jitter[i], 5.0) for i in range(8)],
                          max_distance=0.05)
    exp = GOLDEN["align_humanized"]
    assert _beat_times(s) == exp["beat_times"]
    assert _r(s.bars[0].tempo_bpm, 4) == exp["bar0_tempo"]
    assert _r(s.bars[1].tempo_bpm, 4) == exp["bar1_tempo"]


# ---------- feel detection ----------


def test_feel_straight16() -> None:
    s, gap = _build_44(120.0, 4)
    onsets = [beat.time + k * gap / 4.0
              for bar in s.bars for beat in bar.beats for k in range(4)]
    detect_feel_for_bars(s, onsets)
    assert [b.feel for b in s.bars] == GOLDEN["feel_straight16"]


def test_feel_triplet() -> None:
    s, gap = _build_44(120.0, 4)
    onsets = [beat.time + k * gap / 3.0
              for bar in s.bars for beat in bar.beats for k in range(3)]
    detect_feel_for_bars(s, onsets)
    assert [b.feel for b in s.bars] == GOLDEN["feel_triplet"]


def test_feel_sparse() -> None:
    s, _ = _build_44(120.0, 4)
    detect_feel_for_bars(s, [s.bars[0].beats[0].time])
    assert [b.feel for b in s.bars] == GOLDEN["feel_sparse"]


# ---------- LLM-friendly summary ----------


def test_candidates_with_beat_positions() -> None:
    s, gap = _build_44(120.0, 3)
    _pad_trailing_bars(s, 20.0)
    cands = candidates_with_beat_positions(
        {"k": [(0.0, 1.0), (gap, 0.5), (2 * gap, 0.7)], "s": [(gap, 0.9)]}, s)
    got = {
        p: [{"time": _r(c.time), "bar": c.bar, "beat": _r(c.beat_in_bar),
             "strength": _r(c.strength)} for c in cs]
        for p, cs in cands.items()
    }
    assert got == GOLDEN["candidates"]


def test_summarize_bar_for_prompt() -> None:
    s, _ = _build_44(120.0, 3)
    _pad_trailing_bars(s, 20.0)
    assert summarize_bar_for_prompt(s.bars[0]) == GOLDEN["summarize_bar"]
