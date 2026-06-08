"""Deduplication helpers for the data-cleaning stage (design spec §3.1).

Two levers: exact-duplicate detection via file content hash, and
near-duplicate detection via an onset *signature*, a hash of the per-lane
onset structure rounded to a time bin, so two takes of the same pattern
(or a clip and its slightly-shifted copy) collapse to the same key. E-GMD
is known to carry duplicates; this is how we catch them before the split.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path


def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-1 of a file's bytes (exact-duplicate key)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def onset_signature(
    onsets_by_lane: Mapping[str, Sequence[float]],
    round_s: float = 0.01,
) -> str:
    """A stable hash of the rounded per-lane onset structure.

    Onset times are quantized to `round_s` bins and sorted within each
    lane, so order doesn't matter and sub-bin jitter collapses; lanes with
    no onsets are omitted. Identical patterns -> identical signature.
    """
    parts: list[str] = []
    for lane in sorted(onsets_by_lane):
        times = onsets_by_lane[lane]
        if not times:
            continue
        bins = sorted(round(float(t) / round_s) for t in times)
        parts.append(lane + ":" + ",".join(str(b) for b in bins))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()
