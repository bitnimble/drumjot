"""Top-level scoring orchestrator (batch entry point + web-endpoint core).

Assembles the pieces: parse the chart (ParaDB pack or MIDI), detect the
audio reference (ADTOF on the drum stem), then score raw, globally align,
and score corrected into one `AlignmentResult`. The pure assembly
(`score_onsets`) takes onset lists directly, so it tests without audio; the
`score_paradb` / `score_midi` entry points add file/audio handling.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.scoring.alignment import DEFAULT_BAND_S, DEFAULT_SIGMA_S, score
from app.scoring.audio_onsets import DetectFn, detect_reference_onsets
from app.scoring.correction import global_align
from app.scoring.midi_read import onsets_from_midi_bytes
from app.scoring.models import AlignmentResult, LaneScoreOut
from app.scoring.paradb_read import ParadbAudio, load_paradb_bytes


def score_onsets(
    chart_by_lane: Mapping[str, Sequence[float]],
    audio_by_lane: Mapping[str, Sequence[float]],
    *,
    unmapped_notes: int = 0,
    separation_skipped: bool = False,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
) -> AlignmentResult:
    """Score a chart against an audio reference. Pure: onset lists in,
    `AlignmentResult` out. Scores raw, fits a global warp, then re-scores at
    the corrected positions for the headline number."""
    raw = score(chart_by_lane, audio_by_lane, band=band, sigma=sigma)
    correction = global_align(chart_by_lane, audio_by_lane, band=band, sigma=sigma)
    corrected = score(correction.corrected_by_lane, audio_by_lane, band=band, sigma=sigma)

    per_lane = {
        lane: LaneScoreOut(
            soft_f1=ls.soft_f1,
            soft_precision=ls.soft_precision,
            soft_recall=ls.soft_recall,
            n_chart=ls.n_chart,
            n_audio=ls.n_audio,
        )
        for lane, ls in corrected.per_lane.items()
    }
    return AlignmentResult(
        score=round(100 * raw.f1_weighted),
        score_corrected=round(100 * corrected.f1_weighted),
        f1_macro=corrected.f1_macro,
        f1_weighted=corrected.f1_weighted,
        f1_weighted_raw=raw.f1_weighted,
        per_lane=per_lane,
        offset_sec=correction.offset_sec,
        tempo_ratio=correction.tempo_ratio,
        matched_pairs=correction.matched_pairs,
        corrected_onsets_by_lane=correction.corrected_by_lane,
        unmapped_notes=unmapped_notes,
        audio_reference="drum_track" if separation_skipped else "separated",
        separation_skipped=separation_skipped,
    )


def _write_audio(audio: ParadbAudio, work_dir: Path) -> Path:
    path = work_dir / audio.name
    path.write_bytes(audio.data)
    return path


def score_paradb(
    zip_bytes: bytes,
    *,
    work_dir: Path,
    separator: Any | None = None,
    detect: DetectFn | None = None,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
) -> AlignmentResult:
    """Score a ParaDB `.zip` pack: pick the best chart, use its drums-only
    track as the reference (or separate the song track), and score."""
    pack = load_paradb_bytes(zip_bytes)
    drum_path = _write_audio(pack.drum_audio[0], work_dir) if pack.drum_audio else None
    mix_path = _write_audio(pack.song_audio[0], work_dir) if pack.song_audio else None
    reference = detect_reference_onsets(
        work_dir=work_dir,
        drum_audio_path=drum_path,
        mix_audio_path=mix_path,
        separator=separator,
        detect=detect,
    )
    return score_onsets(
        pack.chart.onsets_by_lane,
        reference.onsets_by_lane,
        unmapped_notes=pack.chart.unmapped_events,
        separation_skipped=reference.separation_skipped,
        band=band,
        sigma=sigma,
    )


def score_midi(
    midi_bytes: bytes,
    *,
    audio_path: Path,
    work_dir: Path,
    separator: Any | None = None,
    detect: DetectFn | None = None,
    band: float = DEFAULT_BAND_S,
    sigma: float = DEFAULT_SIGMA_S,
) -> AlignmentResult:
    """Score a MIDI chart against an uploaded audio file (always separated:
    a raw MIDI upload has no accompanying drum stem)."""
    chart = onsets_from_midi_bytes(midi_bytes)
    reference = detect_reference_onsets(
        work_dir=work_dir, mix_audio_path=audio_path, separator=separator, detect=detect
    )
    return score_onsets(
        chart.onsets_by_lane,
        reference.onsets_by_lane,
        unmapped_notes=chart.unmapped_notes,
        separation_skipped=reference.separation_skipped,
        band=band,
        sigma=sigma,
    )
