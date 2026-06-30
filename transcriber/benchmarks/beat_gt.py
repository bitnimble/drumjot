"""Ground-truth beat grid + per-lane drum onsets from E-GMD MIDI.

Built on `mido` (already a dep) so the beat A/B needs no `pretty_midi`.
The grid drives the mir_eval scoring; the per-lane onsets feed the
synthetic align-onset generator (`onset_synth.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import mido

# General-MIDI percussion -> the 6 Drumjot lanes. Finer than the
# benchmark's 3-class fold (`core/classes.GM_PITCH_TO_CLASS`) because the
# stem-bleed confounder transfers onsets between *spectrally similar*
# lanes (toms/ride/crash matter), which the 3-class map drops.
GM_PITCH_TO_LANE: dict[int, str] = {
    35: "kick", 36: "kick",
    37: "snare", 38: "snare", 40: "snare",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    42: "hihat", 44: "hihat", 46: "hihat",
    51: "ride", 53: "ride", 59: "ride",
    49: "crash", 52: "crash", 55: "crash", 57: "crash",
}

# Spectral-similarity groups bleed is allowed to cross. Membranes share
# broadband transients; metals share high-frequency wash.
LANE_GROUPS: dict[str, frozenset[str]] = {
    "membrane": frozenset({"kick", "snare", "tom"}),
    "metal": frozenset({"hihat", "ride", "crash"}),
}


@dataclass(frozen=True, slots=True)
class LaneOnset:
    time: float
    velocity: int  # 1..127
    lane: str


@dataclass(frozen=True, slots=True)
class GtGrid:
    beats: list[float]       # every beat time, seconds
    downbeats: list[float]   # bar-start times, seconds
    bpm: float
    time_sig: tuple[int, int]


def parse_time_sig(raw: str) -> tuple[int, int]:
    """E-GMD CSV `time_signature` is formatted `N-D` (e.g. `4-4`, `6-8`)."""
    num, _, den = raw.strip().partition("-")
    return (int(num), int(den))


def lane_onsets(midi_path) -> list[LaneOnset]:
    """Per-lane drum onsets (channel-9 note-ons), absolute seconds."""
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat
    out: list[LaneOnset] = []
    for track in mid.tracks:
        tempo = 500_000
        elapsed = 0.0
        for msg in track:
            if msg.time:
                elapsed += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel == 9:
                lane = GM_PITCH_TO_LANE.get(msg.note)
                if lane is None:
                    continue
                out.append(LaneOnset(time=elapsed, velocity=msg.velocity, lane=lane))
    out.sort(key=lambda o: (o.time, o.lane))
    return out


def _tempo_map(mid: mido.MidiFile) -> tuple[list[tuple[int, int]], int]:
    """Absolute-tick (tempo-change, last-event) summary across all tracks."""
    changes: list[tuple[int, int]] = []
    last_tick = 0
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                changes.append((tick, msg.tempo))
        last_tick = max(last_tick, tick)
    changes.sort(key=lambda c: c[0])
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 500_000))
    return changes, last_tick


def _tick_to_sec(changes: list[tuple[int, int]], tpb: int, tick: int) -> float:
    sec = 0.0
    prev_tick, prev_tempo = changes[0]
    for ct, tempo in changes[1:]:
        if ct >= tick:
            break
        sec += mido.tick2second(ct - prev_tick, tpb, prev_tempo)
        prev_tick, prev_tempo = ct, tempo
    sec += mido.tick2second(tick - prev_tick, tpb, prev_tempo)
    return sec


def gt_grid(midi_path, time_sig: tuple[int, int], bpm: float | None = None) -> GtGrid:
    """Beat + downbeat grid from the MIDI tempo map.

    `time_sig` is taken from the E-GMD CSV (authoritative; E-GMD clips are
    constant-meter). Beats step at the time-signature's beat unit
    (`4/denominator` quarter notes); downbeats land every `numerator`
    beats. The grid is anchored at tick 0 = 0 s, which is the click the
    performance was recorded against.
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat
    changes, last_tick = _tempo_map(mid)
    num, den = time_sig
    beat_len_ticks = tpb * 4.0 / den

    beats: list[float] = []
    downbeats: list[float] = []
    i = 0
    while True:
        tick = round(i * beat_len_ticks)
        if tick > last_tick:
            break
        sec = _tick_to_sec(changes, tpb, tick)
        beats.append(sec)
        if i % num == 0:
            downbeats.append(sec)
        i += 1

    if bpm is None:
        bpm = 60_000_000.0 / changes[0][1]
    return GtGrid(beats=beats, downbeats=downbeats, bpm=float(bpm), time_sig=time_sig)


def sanity_coverage(
    grid: GtGrid, onsets: list[LaneOnset], window: float = 0.06
) -> float:
    """Fraction of GT beats with a drum onset within `window` seconds.

    A low value means the performance drifts off its own click (rubato /
    loose timing) or the MIDI<->grid phase is broken, so downbeats can't
    be scored fairly. Used as a load-time drop gate.
    """
    if not grid.beats or not onsets:
        return 0.0
    times = sorted(o.time for o in onsets)
    import bisect

    hit = 0
    for b in grid.beats:
        j = bisect.bisect_left(times, b)
        near = False
        for k in (j - 1, j):
            if 0 <= k < len(times) and abs(times[k] - b) <= window:
                near = True
                break
        hit += near
    return hit / len(grid.beats)
