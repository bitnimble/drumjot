"""Produce the audio-side reference onsets the scorer aligns against.

Decides where the drum audio comes from, then runs ADTOF once for all five
lanes (`detect_all_lanes_adtof`):

  * A ParaDB pack's drums-only track is fed to ADTOF directly, skipping
    Demucs separation, the cheap path that matters at 15k-track scale.
  * A full mix (uploaded audio, or a pack with only a song track) is
    separated first; the drum stem is the ADTOF input.

The reference is the raw ADTOF detections (no LLM prune): the prune is too
expensive across the corpus, and because the same detector biases every
chart identically, the corrected score stays a valid relative ranking for
filtering (documented recall bias).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipeline.adtof_onsets import detect_all_lanes_adtof

# (drum_stem_path) -> {lane: ascending onset seconds}
DetectFn = Callable[[Path], dict[str, list[float]]]


@dataclass
class AudioReference:
    onsets_by_lane: dict[str, list[float]]
    separation_skipped: bool  # True when a pre-isolated drum stem was used


def _resolve_drum_stem(
    *,
    drum_audio_path: Path | None,
    mix_audio_path: Path | None,
    work_dir: Path,
    separator: Any | None,
) -> tuple[Path, bool]:
    """Return `(drum_stem_path, separation_skipped)`. A pre-isolated drum
    track is used as-is; otherwise the mix is separated. Raises `ValueError`
    when neither is available."""
    if drum_audio_path is not None:
        return drum_audio_path, True
    if mix_audio_path is None:
        raise ValueError("no audio: neither a drum stem nor a mix to separate")
    if separator is None:
        from app.pipeline.separate import Separator

        separator = Separator()
    result = separator.run_stems_all(mix_audio_path, work_dir)
    return result.drum_stem, False


def detect_reference_onsets(
    *,
    work_dir: Path,
    drum_audio_path: Path | None = None,
    mix_audio_path: Path | None = None,
    separator: Any | None = None,
    detect: DetectFn | None = None,
) -> AudioReference:
    """Resolve a drum stem (drum track if given, else separate the mix) and
    run ADTOF over all five lanes to produce the reference onsets."""
    detect = detect or detect_all_lanes_adtof
    stem, separation_skipped = _resolve_drum_stem(
        drum_audio_path=drum_audio_path,
        mix_audio_path=mix_audio_path,
        work_dir=work_dir,
        separator=separator,
    )
    return AudioReference(
        onsets_by_lane=detect(stem), separation_skipped=separation_skipped
    )
