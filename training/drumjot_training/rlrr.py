"""Parse Paradiddle / ParaDB `.rlrr` drum charts into per-lane onset times.

Python port of the class mapping in `src/rlrr/drums.ts`. A chart `event` carries
an absolute-seconds `time` and an instrument-instance `name` (e.g. `BP_Snare_C_1`);
the class (`BP_Snare_C`) maps to one of our 10 lanes.

Two source-of-truth quirks (see src/rlrr/drums.ts):
  - Hi-hat open/closed/pedal is NOT encoded in the class (Paradiddle uses one
    `BP_HiHat_C`). The only disambiguator is the optional `event.midi` extension
    (42=closed, 46=open, 44=pedal); absent -> assume closed.
  - There is no side-stick class in rlrr, so the `ss` lane stays empty.
All toms -> `t`; china/splash + cowbell/tambourine are dropped (the `mc` and
`mp` lanes were removed), as is tuned/aux percussion (timpani, triangle,
bongo, mallets, gong) outside the drum-kit vocab. Ride bell, where present,
folds into `rd`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from drumjot_training.lanes import LANES

_CLASS_TO_LANE: dict[str, str] = {
    "BP_Kick_C": "k",
    "BP_Snare_C": "s",
    "BP_HiHat_C": "hc",  # refined by event.midi (46->ho, 44->hp)
    "BP_Crash13_C": "cr", "BP_Crash15_C": "cr", "BP_Crash17_C": "cr",
    # BP_China15_C deliberately unmapped: the `mc` lane was removed (china drops)
    "BP_FloorTom_C": "t", "BP_Tom1_C": "t", "BP_Tom2_C": "t",
    "BP_Ride17_C": "rd", "BP_Ride20_C": "rd",
    # Tambourine/Cowbell deliberately unmapped: the `mp` lane was removed
    # (see lanes.py); they drop like the tuned/aux percussion classes.
}
_HIHAT_MIDI_TO_LANE = {42: "hc", 46: "ho", 44: "hp"}
_INSTANCE_RE = re.compile(r"^(BP_.+_C)_\d+$")


def instance_name_to_class(name: str) -> str | None:
    """`BP_Snare_C_1` -> `BP_Snare_C` (drops the trailing instance index)."""
    m = _INSTANCE_RE.match(name or "")
    return m.group(1) if m else None


def lane_for_event(name: str, midi=None) -> str | None:
    """Lane for one event, or None if out-of-kit. Hi-hat uses `midi` if given."""
    cls = instance_name_to_class(name)
    if cls is None:
        return None
    if cls == "BP_HiHat_C" and midi is not None:
        return _HIHAT_MIDI_TO_LANE.get(int(midi), "hc")
    return _CLASS_TO_LANE.get(cls)


def load(rlrr: object) -> dict:
    """Accept a parsed dict or a path to an `.rlrr` JSON file.

    Read as BYTES, not text: ~10% of community maps ship a UTF-16 (BOM) `.rlrr`
    (Windows-authored), which `read_text()` (UTF-8) rejects with a
    UnicodeDecodeError. `json.loads` on bytes auto-detects UTF-8/16/32 (with or
    without BOM) per the JSON spec, so those maps parse instead of being culled."""
    if isinstance(rlrr, (str, Path)):
        return json.loads(Path(rlrr).read_bytes())
    return rlrr  # type: ignore[return-value]


def _bimodal_hihat_open_vel(events):
    """If the hi-hat track encodes open/closed purely by velocity, return the
    (quieter) open velocity; else None.

    Some mappers split open vs closed via velocity since rlrr has no openness
    field. We only trust it in the unambiguous case: the hi-hat track (with no
    `midi` extension) uses EXACTLY two distinct velocities. One value means no
    distinction; three or more means real dynamics, not an open/closed code.
    The quieter of the two is taken as open, the louder as closed.
    """
    vels = set()
    for ev in events:
        if instance_name_to_class(ev.get("name", "")) == "BP_HiHat_C":
            if ev.get("midi") is not None:
                return None  # explicit openness present -> don't infer from vel
            v = ev.get("vel")
            if v is not None:
                vels.add(v)
    return min(vels) if len(vels) == 2 else None


def onsets_by_lane(rlrr: object) -> dict[str, list[float]]:
    """Per-lane sorted onset times (seconds). Always returns all 10 lanes.

    Hi-hat openness: `event.midi` (46/44) wins; otherwise, if the whole hi-hat
    track uses exactly two velocities, the quieter ones are taken as open (`ho`),
    the louder as closed (`hc`); else all hi-hats are closed.
    """
    data = load(rlrr)
    events = data.get("events", [])
    open_vel = _bimodal_hihat_open_vel(events)
    out: dict[str, list[float]] = {lane: [] for lane in LANES}
    for ev in events:
        midi = ev.get("midi")
        lane = lane_for_event(ev.get("name", ""), midi)
        if lane == "hc" and midi is None and open_vel is not None and ev.get("vel") == open_vel:
            lane = "ho"  # bimodal-velocity open hi-hat
        if lane is not None:
            out[lane].append(float(ev["time"]))
    for ts in out.values():
        ts.sort()
    return out


def song_tracks(rlrr: object) -> list[str]:
    """Always-on backing/song tracks (`audioFileData.songTracks`)."""
    afd = load(rlrr).get("audioFileData") or {}
    seen: list[str] = []
    for n in afd.get("songTracks") or []:
        if n and n not in seen:
            seen.append(n)
    return seen


def drum_tracks(rlrr: object) -> list[str]:
    """Default-muted drum tracks, excluding any also listed as a song track
    (per paradb.ts: a file in both arrays counts as a song track)."""
    afd = load(rlrr).get("audioFileData") or {}
    song = set(song_tracks(rlrr))
    seen: list[str] = []
    for n in afd.get("drumTracks") or []:
        if n and n not in song and n not in seen:
            seen.append(n)
    return seen


def audio_tracks(rlrr: object) -> list[str]:
    """Unique audio filenames referenced (song first, then drum tracks)."""
    return song_tracks(rlrr) + drum_tracks(rlrr)


# Lanes our model splits but a hand chart often won't: hi-hat articulations,
# and ride / crash. Each group has a parent label used when the chart doesn't
# make the distinction.
_GROUPS: dict[str, tuple[str, ...]] = {
    "h": ("hc", "hp", "ho"),
    "cym": ("rd", "cr"),
}
_GROUPED = {s for subs in _GROUPS.values() for s in subs}

# Stable order for aggregating/printing comparison labels (parents + subs).
REPORT_ORDER = ("k", "s", "ss", "t", "h", "hc", "hp", "ho", "cym", "rd", "cr")


def has_lane_track(rlrr: object, lane: str) -> bool:
    """True if the chart's kit (`instruments[]`) defines a dedicated instrument
    mapping to `lane`. Used to decide whether to score a sparse aux lane: only
    if the map actually charts that instrument."""
    for inst in load(rlrr).get("instruments", []):
        cls = inst.get("class")
        if cls and _CLASS_TO_LANE.get(cls) == lane:
            return True
    return False


def comparison_pairs(
    gt: dict[str, list[float]], est: dict[str, list[float]]
) -> list[tuple[str, list[float], list[float]]]:
    """Optimistic (label, ref, est) pairs for scoring model `est` vs chart `gt`.

    Per group (hi-hat articulations; ride/crash/misc-cymbal): if the chart
    distinguishes >=2 subclasses (e.g. it has both open AND closed hats, or
    both ride AND crash), score each present subclass on its own. Otherwise
    fold the whole group, on BOTH the model and chart sides, to the parent
    label, so the model isn't penalised for a subclass distinction the mapper
    never charted. Ungrouped lanes (k/s/ss/t) pass through unchanged. Pairs
    whose chart side is empty are still returned (caller skips empty refs)."""
    pairs: list[tuple[str, list[float], list[float]]] = []
    for lane in LANES:
        if lane not in _GROUPED:
            pairs.append((lane, list(gt.get(lane, [])), list(est.get(lane, []))))
    for parent, subs in _GROUPS.items():
        present = [s for s in subs if gt.get(s)]
        if len(present) >= 2:  # chart makes the distinction -> score per subclass
            for s in present:
                pairs.append((s, list(gt.get(s, [])), list(est.get(s, []))))
        elif len(present) == 1:  # chart lumps the group -> fold both sides
            ref = sorted(t for s in subs for t in gt.get(s, []))
            es = sorted(t for s in subs for t in est.get(s, []))
            pairs.append((parent, ref, es))
        # 0 present -> no chart onsets in the group; nothing to score
    return pairs


def complexity(rlrr: object) -> int:
    """Difficulty (1-4); used to pick the hardest chart in a multi-chart zip."""
    return int((load(rlrr).get("recordingMetadata") or {}).get("complexity", 0))


_DIFFICULTY_RANK = (("expert", 4), ("hard", 3), ("medium", 2), ("easy", 1))


def difficulty_rank(name: str) -> int:
    """Difficulty implied by a chart's filename (expert>hard>medium>easy, else 0).

    `recordingMetadata.complexity` is coarse (often the SAME 1-4 for a map's Expert
    and Hard charts), so it ties; this breaks the tie toward the genuinely harder
    chart. Mirrors transcriber `paradb_read._difficulty_rank`."""
    n = name.lower()
    for word, rank in _DIFFICULTY_RANK:
        if word in n:
            return rank
    return 0


def pick_hardest(charts) -> Path | None:
    """The hardest of several `.rlrr` paths, chosen DETERMINISTICALLY: by
    `complexity`, then filename `difficulty_rank`, then the path itself. Without the
    tiebreaks a complexity tie resolves by directory-iteration order (e.g.
    `rglob`), which varies run-to-run -- a map with Expert+Hard charts at the same
    complexity would then parse a different GT each run."""
    charts = list(charts)
    if not charts:
        return None
    return max(charts, key=lambda p: (complexity(p), difficulty_rank(Path(p).name), str(p)))
