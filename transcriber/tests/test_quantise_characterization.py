"""Characterization (golden) tests pinning the CURRENT quantise.py behaviour.

These exist to make the quantise.py module split a provably
behaviour-preserving refactor: every golden value below was captured by
running the pre-split code on the synthetic inputs, so a post-split value
that diverges is a regression.

All inputs are hand-built python OnsetCandidate lists, SimpleNamespace bar
structures and (for the envelope re-snap) a tiny synthetic OnsetEnvelope --
no audio, no model, no GPU -- so the whole file is fast and CPU-only. The
LLM residual pass is exercised with the network call stubbed
(`_call_window` patched to a fixed response) so it's deterministic and
offline.

Coverage: the geometric per-(lane, bar) slot snap (+ cross-bar overflow
reassignment + sub-slot residual), the shared shift applier
(`_apply_llm_shifts`) both accepting a valid shift and rejecting a
colliding group, cross-bar target walking, per-note-current-slot, the
subdivision-grid inference + snapping pass, the envelope re-snap, the
LLM-facing indexing / windowing / prompt formatting / shift extraction,
and the small render helpers (slot labels, residual tags, token budget).

Everything imports from the public `app.pipeline.quantise` facade, so the
same suite runs unchanged against the module-split layout.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.models import OnsetCandidate
from app.pipeline.envelope import OnsetEnvelope
from app.pipeline.quantise import (
    _apply_llm_shifts,
    _build_summary,
    _build_windows,
    _candidate_grids,
    _circular_dist,
    _current_slot,
    _envelope_snap,
    _extract_shifts,
    _format_window,
    _geometric_snap,
    _index_for_llm,
    _infer_grid,
    _max_tokens_for,
    _musical_grid_snap,
    _nearest_grid_slot,
    _residual_tag,
    _resolve_cross_bar_target,
    _slot_label,
    quantise_kept_onsets,
)

_R = 9  # rounding digits used when the golden values were captured
_SPAN = 2.0 / 48  # slot_span for a 4/4 @ 120 BPM bar (2.0 s, 48 slots)


def _r(x: float | None, n: int = _R) -> float | None:
    return None if x is None else round(float(x), n)


def _bar(index, start_time, end_time, ts=(4, 4), bpm=120.0, feel="straight16"):
    return SimpleNamespace(
        index=index, start_time=start_time, end_time=end_time,
        time_signature=ts, tempo_bpm=bpm, feel=feel, beats=[],
    )


def _structure(bars):
    return SimpleNamespace(
        bars=bars, initial_tempo=120.0, initial_time_signature=(4, 4),
    )


def _cand(**kw) -> OnsetCandidate:
    return OnsetCandidate(**kw)


# ---------- _resolve_cross_bar_target ----------

def test_resolve_cross_bar_target_walks_boundaries() -> None:
    st = _structure([
        _bar(0, 0.0, 2.0),
        _bar(1, 2.0, 4.0, ts=(3, 4)),
        _bar(2, 4.0, 5.5, ts=(3, 4)),
    ])
    assert _resolve_cross_bar_target(0, 47, 1, st, 12) == (1, 0)
    assert _resolve_cross_bar_target(0, 47, 2, st, 12) == (1, 1)
    assert _resolve_cross_bar_target(0, 0, -1, st, 12) is None  # off song front
    assert _resolve_cross_bar_target(1, 0, -1, st, 12) == (0, 47)  # back into 4/4 bar
    assert _resolve_cross_bar_target(0, 10, 0, st, 12) == (0, 10)  # no-op
    assert _resolve_cross_bar_target(2, 35, 1, st, 12) is None  # off song end
    assert _resolve_cross_bar_target(0, 46, 3, st, 12) == (1, 1)  # multi-bar walk


# ---------- _current_slot ----------

def test_current_slot() -> None:
    st = _structure([_bar(0, 0.0, 2.0)])
    b = st.bars[0]

    def mk(t, qt):
        c = _cand(time=t, strength=1.0, bar=0, beat_in_bar=1.0)
        c.quantised_time = qt
        return c

    assert _current_slot(mk(0.0, None), b, 12) == 0
    assert _current_slot(mk(0.5, None), b, 12) == 12
    assert _current_slot(mk(0.5, 0.5), b, 12) == 12
    assert _current_slot(mk(1.999, None), b, 12) == 47  # clamped to last slot
    assert _current_slot(mk(0.0, 47 * _SPAN), b, 12) == 47  # qt wins over raw time


# ---------- _candidate_grids ----------

def test_candidate_grids_12() -> None:
    grids = [(n, list(p)) for n, p in _candidate_grids(12)]
    assert grids == [
        ("quarter", [0]),
        ("straight_8", [0, 6]),
        ("straight_16", [0, 3, 6, 9]),
        ("triplet_8", [0, 4, 8]),
        ("triplet_16", [0, 2, 4, 6, 8, 10]),
        ("swing_8", [0, 8]),
    ]


def test_candidate_grids_16_drops_non_integer_grids() -> None:
    # At 16 slots/beat the triplet + swing grids fall off integer slots.
    grids = [(n, list(p)) for n, p in _candidate_grids(16)]
    assert grids == [
        ("quarter", [0]),
        ("straight_8", [0, 8]),
        ("straight_16", [0, 4, 8, 12]),
    ]


# ---------- _circular_dist ----------

def test_circular_dist_wraps_the_beat() -> None:
    positions = (0, 3, 6, 9)
    assert _circular_dist(0, positions, 12) == 0
    assert _circular_dist(1, positions, 12) == 1
    assert _circular_dist(2, positions, 12) == 1
    assert _circular_dist(11, positions, 12) == 1  # wraps to slot 0/12
    assert _circular_dist(6, positions, 12) == 0
    assert _circular_dist(10, positions, 12) == 1


# ---------- _nearest_grid_slot ----------

def test_nearest_grid_slot_beat_cyclic() -> None:
    pos = (0, 6)
    assert _nearest_grid_slot(0, pos, 12) == 0
    assert _nearest_grid_slot(1, pos, 12) == 0
    assert _nearest_grid_slot(2, pos, 12) == 0
    assert _nearest_grid_slot(11, pos, 12) == 12  # snaps forward to next downbeat
    assert _nearest_grid_slot(13, pos, 12) == 12
    assert _nearest_grid_slot(23, pos, 12) == 24
    assert _nearest_grid_slot(5, pos, 12) == 6
    assert _nearest_grid_slot(7, pos, 12) == 6


# ---------- _infer_grid ----------

def test_infer_grid() -> None:
    grids = _candidate_grids(12)
    assert _infer_grid([0, 6, 0, 6, 0, 6], grids, 12) == ("straight_8", (0, 6))
    assert _infer_grid([0, 0, 0, 0, 0], grids, 12) == ("quarter", (0,))
    assert _infer_grid([0, 6], grids, 12) is None  # < _GRID_MIN_ONSETS
    assert _infer_grid([0, 4, 8, 0, 4, 8], grids, 12) == ("triplet_8", (0, 4, 8))
    assert _infer_grid([0, 3, 6, 9, 0, 3, 6, 9], grids, 12) == (
        "straight_16", (0, 3, 6, 9)
    )
    # A messy population still resolves to straight_8 here (pinned as-is).
    assert _infer_grid([0, 1, 5, 7, 11, 2], grids, 12) == ("straight_8", (0, 6))


# ---------- _slot_label ----------

def test_slot_label() -> None:
    assert _slot_label(0, 12) == "(beat 1)"
    assert _slot_label(6, 12) == "(& of 1)"
    assert _slot_label(3, 12) == "(e of 1)"
    assert _slot_label(9, 12) == "(a of 1)"
    assert _slot_label(4, 12) == "(trip-2 of 1)"
    assert _slot_label(8, 12) == "(trip-3 of 1)"
    assert _slot_label(5, 12) == "(1/48 +5 of 1)"
    assert _slot_label(12, 12) == "(beat 2)"
    assert _slot_label(13, 12) == "(1/48 +1 of 2)"
    assert _slot_label(18, 12) == "(& of 2)"


# ---------- _residual_tag ----------

def test_residual_tag() -> None:
    assert _residual_tag(None) == ""
    assert _residual_tag(0.0) == ""
    assert _residual_tag(0.1) == ""  # below the 0.25 threshold
    assert _residual_tag(0.25) == " r+0.25"
    assert _residual_tag(-0.3) == " r-0.30"
    assert _residual_tag(0.45) == " r+0.45"
    assert _residual_tag(-0.5) == " r-0.50"


# ---------- _max_tokens_for ----------

def test_max_tokens_for() -> None:
    assert _max_tokens_for(0) == 8192
    assert _max_tokens_for(10) == 8192
    assert _max_tokens_for(100) == 8192
    assert _max_tokens_for(1000) == 17024
    assert _max_tokens_for(5000) == 81024


# ---------- _geometric_snap (full scenario) ----------

def _geometric_scenario():
    st = _structure([_bar(0, 0.0, 2.0), _bar(1, 2.0, 4.0)])
    k = [
        _cand(time=0.005, strength=5.0, bar=0, beat_in_bar=1.0),
        _cand(time=0.51, strength=5.0, bar=0, beat_in_bar=2.02),
        _cand(time=1.02, strength=5.0, bar=0, beat_in_bar=3.05),
        _cand(time=1.995, strength=5.0, bar=0, beat_in_bar=4.99),  # overflow -> bar 1
    ]
    s = [
        _cand(time=0.26, strength=5.0, bar=0, beat_in_bar=1.55),
        _cand(time=0.27, strength=5.0, bar=0, beat_in_bar=1.56),
    ]
    return st, {"k": k, "s": s}


def test_geometric_snap_full_scenario() -> None:
    st, kept = _geometric_scenario()
    shifts = _geometric_snap(kept, st, slots_per_beat=12)
    assert shifts == {("s", 0): -1}

    got = [
        (p, i, c.bar, round(c.beat_in_bar, 6), _r(c.quantised_time),
         c.quantised_shift_slots, c.off_grid, _r(c.quantised_residual_slots))
        for p, cands in kept.items()
        for i, c in enumerate(cands)
    ]
    assert got == [
        ("k", 0, 0, 1.0, 0.0, 0, False, 0.0),
        ("k", 1, 0, 2.02, 0.5, 0, False, 0.24),
        ("k", 2, 0, 3.05, 1.041666667, 0, False, -0.4),
        ("k", 3, 1, 1.0, 2.0, 0, False, 0.0),  # overflow reassigned to bar 1 downbeat
        ("s", 0, 0, 1.55, 0.25, -1, False, -0.4),
        ("s", 1, 0, 1.56, 0.291666667, 0, False, -0.28),
    ]


# ---------- _index_for_llm ----------

def test_index_for_llm_orders_by_bar_slot_pitch() -> None:
    st, kept = _geometric_scenario()
    _geometric_snap(kept, st, slots_per_beat=12)
    entries = _index_for_llm(kept, st, slots_per_beat=12)
    got = [(e.pitch, e.idx, e.bar, e.slot, _r(e.residual)) for e in entries]
    assert got == [
        ("k", 0, 0, 0, 0.0),
        ("s", 0, 0, 6, -0.4),
        ("s", 1, 0, 7, -0.28),
        ("k", 1, 0, 12, 0.24),
        ("k", 2, 0, 25, -0.4),
        ("k", 3, 1, 0, 0.0),
    ]


# ---------- _build_windows + _format_window ----------

def test_build_and_format_windows() -> None:
    st, kept = _geometric_scenario()
    _geometric_snap(kept, st, slots_per_beat=12)
    entries = _index_for_llm(kept, st, slots_per_beat=12)
    windows = _build_windows(
        entries, st, target_onsets=150, max_bars=8, context_bars=1
    )
    assert len(windows) == 1
    w = windows[0]
    assert sorted(w.core_set) == [0, 1]
    assert w.render_bars == [0, 1]
    assert w.local_to_global == [
        ("k", 0), ("s", 0), ("s", 1), ("k", 1), ("k", 2), ("k", 3),
    ]
    assert _format_window(st, w, slots_per_beat=12) == (
        "Bar 0 [4/4, 120.0 BPM, feel=straight16]:\n"
        "  slot  1 (beat 1): #0(k)\n"
        "  slot  7 (& of 1): #1(s r-0.40)\n"
        "  slot  8 (1/48 +7 of 1): #2(s r-0.28)\n"
        "  slot 13 (beat 2): #3(k)\n"
        "  slot 26 (1/48 +1 of 3): #4(k r-0.40)\n"
        "\n"
        "Bar 1 [4/4, 120.0 BPM, feel=straight16]:\n"
        "  slot  1 (beat 1): #5(k)"
    )


# ---------- _musical_grid_snap ----------

def test_musical_grid_snap_snaps_stray_hit_onto_the_voted_grid() -> None:
    st = _structure([_bar(0, 0.0, 2.0)])
    hh = []
    for slot in [0, 6, 12, 18, 24, 30, 36, 43]:  # straight-8 lane, last hit at 43
        c = _cand(time=slot * _SPAN, strength=5.0, bar=0, beat_in_bar=1.0 + slot / 12)
        c.quantised_time = slot * _SPAN
        c.off_grid = False
        hh.append(c)
    kept = {"h": hh}
    shifts = _musical_grid_snap(kept, st, slots_per_beat=12)
    assert shifts == {("h", 7): -1}
    got = [(_r(c.quantised_time), c.quantised_shift_slots) for c in hh]
    assert got == [
        (0.0, None), (0.25, None), (0.5, None), (0.75, None),
        (1.0, None), (1.25, None), (1.5, None), (1.75, -1),
    ]


# ---------- _apply_llm_shifts ----------

def _two_kicks():
    st = _structure([_bar(0, 0.0, 2.0)])
    k0 = _cand(time=0.50, strength=5.0, bar=0, beat_in_bar=2.0)
    k1 = _cand(time=0.54, strength=5.0, bar=0, beat_in_bar=2.0 + 1 / 12)
    kept = {"k": [k0, k1]}
    _geometric_snap(kept, st, slots_per_beat=12)
    return st, kept


def test_apply_llm_shifts_rejects_colliding_group() -> None:
    st, kept = _two_kicks()
    _apply_llm_shifts(kept, st, {("k", 0): 1}, slots_per_beat=12)
    assert [_r(c.quantised_time) for c in kept["k"]] == [0.5, 0.541666667]


def test_apply_llm_shifts_applies_valid_shift() -> None:
    st, kept = _two_kicks()
    _apply_llm_shifts(kept, st, {("k", 0): -1}, slots_per_beat=12)
    assert [_r(c.quantised_time) for c in kept["k"]] == [0.458333333, 0.541666667]


# ---------- _envelope_snap ----------

def test_envelope_snap_moves_onset_onto_stronger_transient() -> None:
    st = _structure([_bar(0, 0.0, 2.0)])
    frame_times = np.arange(0, 400) * (2.0 / 400)
    env = np.full(400, 0.05)
    peak_idx = int((2 * _SPAN) / (2.0 / 400))  # strong bump at slot 2
    env[peak_idx - 1:peak_idx + 2] = 5.0
    oenv = OnsetEnvelope(frame_times=frame_times, env=env, ref=5.0)
    c = _cand(time=0.0, strength=5.0, bar=0, beat_in_bar=1.0)
    c.quantised_time = 0.0
    c.off_grid = False
    kept = {"h": [c]}
    shifts = _envelope_snap(kept, st, {"h": oenv}, slots_per_beat=12)
    assert shifts == {("h", 0): 2}
    assert _r(c.quantised_time) == 0.083333333  # slot 2 = 2 * _SPAN
    assert c.quantised_shift_slots == 2


# ---------- _extract_shifts ----------

def _tool_use_response(shifts):
    block = SimpleNamespace(type="tool_use", name="shift_onsets", input={"shifts": shifts})
    return SimpleNamespace(content=[block])


def test_extract_shifts_parses_and_filters() -> None:
    resp = _tool_use_response([
        {"id": 0, "shift": 1},
        {"id": 2, "shift": -2},
        {"id": 99, "shift": 1},   # out of range -> dropped
        {"id": 1, "shift": "x"},  # non-int -> dropped
        "junk",                    # non-dict -> dropped
    ])
    assert _extract_shifts(resp, n=5) == {0: 1, 2: -2}


def test_extract_shifts_no_tool_block() -> None:
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    assert _extract_shifts(resp, n=5) == {}


# ---------- quantise_kept_onsets (orchestration, LLM stubbed) ----------

def test_quantise_kept_onsets_llm_stubbed_end_to_end() -> None:
    st, kept = _geometric_scenario()

    def fake_call_window(client, template, structure, window, wi, spb, cancel_event):
        # Shift the first shiftable onset (#0 -> ("k", 0)) later by 1 slot.
        return {window.local_to_global[0]: 1}

    with patch("app.pipeline.quantise_llm.settings") as msettings, \
         patch("app.pipeline.quantise_llm._load_prompt_template", return_value="{BARS}"), \
         patch("app.pipeline.quantise_llm.anthropic.Anthropic", return_value=object()), \
         patch("app.pipeline.quantise_llm._call_window", side_effect=fake_call_window):
        msettings.anthropic_api_key = "test-key"
        summary = quantise_kept_onsets(
            kept, st, use_llm=True, use_grid=True, slots_per_beat=12,
        )

    assert summary["llm_status"] == "ok"
    assert summary["llm_shifted"] == 1
    assert summary["off_grid"] == 0
    assert summary["slots_per_beat"] == 12
    # ("k", 0) was on slot 0; the stubbed LLM shift moves it +1 slot.
    assert kept["k"][0].llm_shift_slots == 1
    assert _r(kept["k"][0].quantised_time) == _r(_SPAN)


def test_quantise_kept_onsets_geometric_only() -> None:
    st, kept = _geometric_scenario()
    summary = quantise_kept_onsets(
        kept, st, use_llm=False, use_grid=False, slots_per_beat=12,
    )
    assert summary["llm_status"] == "skipped"
    assert summary["geometric_shifted"] == 1  # ("s", 0) shifted -1
    assert summary["grid_shifted"] == 0
    assert summary["llm_shifted"] == 0
    assert kept["s"][0].geometric_shift_slots == -1


# ---------- _build_summary ----------

def test_build_summary_shape() -> None:
    st, kept = _geometric_scenario()
    geo = _geometric_snap(kept, st, slots_per_beat=12)
    summary = _build_summary(
        kept_by_pitch=kept,
        geometric_shifts=geo,
        envelope_shifts={},
        grid_shifts={},
        llm_shifts={},
        llm_status="skipped",
        slots_per_beat=12,
    )
    assert summary["geometric_shifted"] == 1
    assert summary["off_grid"] == 0
    assert summary["match_band"] == 2
    assert summary["max_llm_shift"] == 2
    assert summary["slots_per_beat"] == 12
    # Only the shifted onset ("s", 0) appears in per_pitch (non-zero shift).
    assert list(summary["per_pitch"].keys()) == ["s"]
    assert summary["per_pitch"]["s"][0]["idx"] == 0
    assert summary["per_pitch"]["s"][0]["geometric_shift"] == -1
