"""Read a ParaDB `.zip` map pack: the best-difficulty chart + its audio.

A pack ships one or more `.rlrr` charts (one per difficulty) plus the audio
they reference, typically a drumless "song" track and a drums-only track.
We pick the highest-complexity chart (the most complete), parse it into
per-lane onset seconds (`rlrr_read`), and extract the referenced audio
bytes. The drums-only track, when present, is the preferred scoring
reference (it can skip separation). Port of `src/rlrr/paradb.ts` on stdlib
`zipfile`.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.scoring.rlrr_read import RlrrChart, chart_from_rlrr, parse_rlrr_bytes


@dataclass
class ParadbAudio:
    name: str  # basename
    data: bytes
    is_drums: bool  # True for audioFileData.drumTracks entries


@dataclass
class ParadbPack:
    chart: RlrrChart
    rlrr_name: str
    song_audio: list[ParadbAudio]  # songTracks, role full-mix
    drum_audio: list[ParadbAudio]  # drumTracks, role drums-only


def load_paradb_zip(path: Path) -> ParadbPack:
    return load_paradb_bytes(path.read_bytes())


def load_paradb_bytes(data: bytes) -> ParadbPack:
    """Parse a ParaDB pack from its zip bytes. Raises `ValueError` on a
    malformed pack (no zip directory, or no `.rlrr` chart)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid zip archive: {exc}") from exc

    names = zf.namelist()
    rlrr_names = [n for n in names if n.lower().endswith(".rlrr")]
    if not rlrr_names:
        raise ValueError("No .rlrr chart found in the ParaDB pack.")

    candidates = [(n, parse_rlrr_bytes(zf.read(n))) for n in rlrr_names]
    chosen_name, chosen = candidates[0]
    for name, rlrr in candidates[1:]:
        if (_complexity(rlrr), _difficulty_rank(name)) > (
            _complexity(chosen),
            _difficulty_rank(chosen_name),
        ):
            chosen_name, chosen = name, rlrr

    chart = chart_from_rlrr(chosen)

    # Resolve referenced audio: song tracks first, then drum tracks. De-dupe
    # on the resolved zip entry, so a file referenced in both arrays loads
    # once and keeps its (earlier) song-track classification.
    song_audio: list[ParadbAudio] = []
    drum_audio: list[ParadbAudio] = []
    seen: set[str] = set()
    refs = [(ref, False) for ref in chart.song_tracks] + [
        (ref, True) for ref in chart.drum_tracks
    ]
    for ref, is_drums in refs:
        entry = _resolve_entry(names, ref)
        if entry is None or entry in seen:
            continue
        seen.add(entry)
        audio = ParadbAudio(name=_basename(entry), data=zf.read(entry), is_drums=is_drums)
        (drum_audio if is_drums else song_audio).append(audio)

    return ParadbPack(
        chart=chart, rlrr_name=chosen_name, song_audio=song_audio, drum_audio=drum_audio
    )


def _complexity(rlrr: dict) -> int:
    meta = rlrr.get("recordingMetadata") or {}
    return int(meta.get("complexity") or 0)


def _difficulty_rank(name: str) -> int:
    n = name.lower()
    for word, rank in (("expert", 4), ("hard", 3), ("medium", 2), ("easy", 1)):
        if word in n:
            return rank
    return 0


def _basename(entry: str) -> str:
    return entry.replace("\\", "/").rsplit("/", 1)[-1]


def _resolve_entry(entries: list[str], ref: str) -> str | None:
    """Resolve an audioFileData reference to a zip entry. Prefer an exact
    normalized full-path match; fall back to a case-insensitive basename
    match for the common bare-name case."""
    norm = lambda s: s.replace("\\", "/").lower()  # noqa: E731
    full_ref = norm(ref)
    for e in entries:
        if norm(e) == full_ref:
            return e
    wanted = _basename(ref).lower()
    for e in entries:
        if _basename(e).lower() == wanted:
            return e
    return None
