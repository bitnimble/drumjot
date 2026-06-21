"""Data-cleaning stage (design spec §3), dedup pass.

This is the dependency-light first slice of the cleaning stage: drop exact
and near-duplicate clips before the train/val/test split (E-GMD is known to
carry duplicates). Exact dups share MIDI file bytes; near dups share an
onset *signature* (the rounded per-lane onset structure), catching re-saves
and slight shifts.

The remaining cleaning steps from spec §3; quality scoring via
`transcriber/app/scoring/` and forced-align-or-discard against the
onset-strength envelope, depend on audio stems and the scoring package, so
they live behind that infra and are TODO here (see README / spec §3.2-3.4).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from drumjot_training import forced_align
from drumjot_training.dedup import onset_signature, sha1_of_file
from drumjot_training.egmd import EgmdClip
from drumjot_training.midi_labels import onsets_from_path


def dedup_clips(clips: Iterable[EgmdClip]) -> tuple[list[EgmdClip], list[tuple[EgmdClip, str]]]:
    """Split `clips` into (kept, dropped) by MIDI exact- and near-duplicate.

    `dropped` carries a reason ("exact" | "near") per clip. First occurrence
    of a pattern is kept; later identical ones are dropped.
    """
    seen_hash: set[str] = set()
    seen_sig: set[str] = set()
    kept: list[EgmdClip] = []
    dropped: list[tuple[EgmdClip, str]] = []
    for clip in clips:
        file_hash = sha1_of_file(clip.midi_path)
        if file_hash in seen_hash:
            dropped.append((clip, "exact"))
            continue
        sig = onset_signature(onsets_from_path(clip.midi_path))
        if sig in seen_sig:
            dropped.append((clip, "near"))
            continue
        seen_hash.add(file_hash)
        seen_sig.add(sig)
        kept.append(clip)
    return kept, dropped


def support_score(
    onsets_by_lane: Mapping[str, Sequence[float]],
    env: np.ndarray,
    env_fps: float,
    *,
    window_s: float = 0.05,
    support_floor: float,
) -> dict:
    """Fraction of labeled onsets backed by a real onset-strength peak.

    For each onset, `forced_align.align_lane` checks for an envelope peak
    >= `support_floor` within +/-`window_s`. A clip with mislabels or gross
    timing offsets scores low (its onsets land where there's no transient),
    which is how the cleaning stage catches E-GMD's known flaws (spec §3.3/§3.4).
    Returns {fraction, n_total, n_supported}; empty input scores 1.0.
    """
    total = supported = 0
    for ts in onsets_by_lane.values():
        if not ts:
            continue
        for _t, ok in forced_align.align_lane(ts, env, env_fps, window_s, support_floor):
            total += 1
            supported += int(ok)
    return {
        "fraction": supported / total if total else 1.0,
        "n_total": total,
        "n_supported": supported,
    }


def filter_lanes_by_support(
    onsets_by_lane: Mapping[str, Sequence[float]],
    env: np.ndarray,
    env_fps: float,
    *,
    support_floor: float,
    min_support: float,
    window_s: float = 0.05,
    snap: bool = True,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Per-(clip, lane) label-quality gate + onset snap.

    (Lane-level companion to `filter_by_support`, which gates whole clips.)

    For each lane, `forced_align.align_lane` snaps every onset to its window's
    envelope peak and flags support (a peak >= `support_floor` within
    +/-`window_s`). The lane's support fraction gates the WHOLE lane: below
    `min_support` it's zeroed out (its labels are mislabeled / mis-aligned for THIS
    clip and would poison the corpus -- e.g. an A2MD ride MIDI that doesn't match
    the recording). A KEPT lane's onsets are returned at their **snapped** times
    when `snap` (supported onsets moved onto the real transient; unsupported ones
    -- a minority in a kept lane -- left at their original time, NOT dropped, so we
    don't manufacture false-negative penalties for the oracle). A lane with no
    onsets passes through untouched and is omitted from the report. Dataset-agnostic;
    a near-no-op on clean labels (synthetic STAR scores ~1.0).
    """
    filtered: dict[str, list[float]] = {}
    support: dict[str, float] = {}
    for lane, ts in onsets_by_lane.items():
        ts = list(ts)
        if not ts:
            filtered[lane] = []
            continue
        aligned = forced_align.align_lane(ts, env, env_fps, window_s, support_floor)
        support[lane] = sum(1 for _t, ok in aligned if ok) / len(aligned)
        if support[lane] < min_support:
            filtered[lane] = []
        else:
            filtered[lane] = sorted(t for t, _ok in aligned) if snap else ts
    return filtered, support


def audio_support_score(
    audio_path,
    onsets_by_lane: Mapping[str, Sequence[float]],
    *,
    window_s: float = 0.05,
    support_floor: float,
    hop_length: int = 64,
) -> dict:
    """`support_score` against the onset-strength envelope of `audio_path`."""
    env, fps = forced_align.onset_envelope(audio_path, hop_length=hop_length)
    return support_score(
        onsets_by_lane, env, fps, window_s=window_s, support_floor=support_floor
    )


def filter_by_support(
    clips: Iterable,
    score_fn: Callable[[object], float],
    min_support: float,
) -> tuple[list, list[tuple[object, float]]]:
    """Keep clips whose support fraction (`score_fn(clip)`) >= `min_support`.

    `score_fn` is injected (e.g. `lambda c: audio_support_score(...)["fraction"]`)
    so the gating is testable without audio. Returns (kept, dropped) where
    dropped carries each clip's failing fraction.
    """
    kept: list = []
    dropped: list[tuple[object, float]] = []
    for clip in clips:
        frac = score_fn(clip)
        if frac >= min_support:
            kept.append(clip)
        else:
            dropped.append((clip, frac))
    return kept, dropped
