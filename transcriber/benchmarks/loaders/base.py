"""Shared types and dispatch for dataset loaders."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..core.events import OnsetEvent


@dataclass(frozen=True, slots=True)
class LoadedTrack:
    """One scoreable track: an audio file plus its ground-truth onsets."""

    track_id: str         # stable identifier, used as the JSONL key
    audio_path: Path
    reference: list[OnsetEvent]


class DatasetLoader(Protocol):
    name: str

    def iter_tracks(self, root: Path) -> Iterator[LoadedTrack]:
        """Walk `root` (a `datasets/<name>/` folder) and yield every scoreable track."""
        ...


def get_loader(name: str) -> DatasetLoader:
    """Resolve a loader by its dataset slug.

    Lazy imports so the harness doesn't pay for `mido` / `xml` imports
    on loaders the user isn't running.
    """
    if name == "e-gmd":
        from . import egmd
        return egmd.LOADER
    if name == "mdb-drums":
        from . import mdb_drums
        return mdb_drums.LOADER
    if name == "idmt-smt-drums":
        from . import idmt_smt
        return idmt_smt.LOADER
    raise ValueError(
        f"unknown dataset {name!r}; expected one of: e-gmd, mdb-drums, idmt-smt-drums"
    )
