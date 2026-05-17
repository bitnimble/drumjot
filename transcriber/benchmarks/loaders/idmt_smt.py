"""IDMT-SMT-Drums loader.

Pairs each `annotation_xml/<id>.xml` with `audio/<id>.wav`. The XML
events are read with the stdlib parser (defusedxml is preferable in
adversarial settings, but the dataset is locally-trusted audio
annotations).
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET  # noqa: S405 - locally-trusted data
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..core.classes import IDMT_LABEL_TO_CLASS
from ..core.events import OnsetEvent
from .base import LoadedTrack

log = logging.getLogger(__name__)


@dataclass
class IdmtSmtLoader:
    name: str = "idmt-smt-drums"

    def iter_tracks(self, root: Path) -> Iterator[LoadedTrack]:
        ann_dir = root / "annotation_xml"
        audio_dir = root / "audio"
        if not ann_dir.is_dir():
            raise FileNotFoundError(
                f"IDMT-SMT-Drums annotations missing at {ann_dir}. "
                f"See {root}/README.md for the expected layout."
            )
        if not audio_dir.is_dir():
            raise FileNotFoundError(
                f"IDMT-SMT-Drums audio missing at {audio_dir}. "
                f"See {root}/README.md for the expected layout."
            )

        for ann_path in sorted(ann_dir.glob("*.xml")):
            track_id = ann_path.stem
            audio_path = audio_dir / f"{track_id}.wav"
            if not audio_path.exists():
                log.warning("IDMT: audio missing for %s, skipping", track_id)
                continue

            try:
                reference = _parse_annotation(ann_path)
            except Exception as exc:
                log.warning("IDMT: failed to parse %s: %s", ann_path, exc)
                continue

            yield LoadedTrack(
                track_id=track_id,
                audio_path=audio_path,
                reference=reference,
            )


def _parse_annotation(path: Path) -> list[OnsetEvent]:
    """Read `<event><onsetSec>..</onsetSec><instrument>..</instrument></event>` entries."""
    tree = ET.parse(str(path))  # noqa: S314 - locally-trusted data
    root = tree.getroot()
    events: list[OnsetEvent] = []
    # Find <event> elements anywhere in the tree (the dataset's exact
    # element hierarchy varies between subsets — WaveDrum / RealDrum /
    # TechnoDrum each wrap them slightly differently).
    for ev in root.iter("event"):
        onset_el = ev.find("onsetSec")
        instr_el = ev.find("instrument")
        if onset_el is None or instr_el is None:
            continue
        if onset_el.text is None or instr_el.text is None:
            continue
        try:
            onset = float(onset_el.text)
        except ValueError:
            continue
        drum_class = IDMT_LABEL_TO_CLASS.get(instr_el.text.strip().upper())
        if drum_class is None:
            continue
        events.append(OnsetEvent(time=onset, drum_class=drum_class))
    events.sort(key=lambda e: (e.time, e.drum_class.value))
    return events


LOADER = IdmtSmtLoader()
