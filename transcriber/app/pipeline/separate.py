"""Two-stage drum separation: full mix -> drum stem -> per-instrument stems.

Stage 1 uses Demucs v4 (htdemucs_ft) to extract a drum stem from the full mix.
Stage 2 uses the community MDX23C 6-stem drum-piece separator (aufr33 /
jarredou model) to break the drum stem into kick / snare / toms / hi-hat /
ride / crash.

Both stages run via the `audio-separator` library, which is a thin wrapper
that downloads weights from Hugging Face and dispatches inference onto CUDA
(or CPU if no GPU is available).

Failure modes intentionally surface up to the caller - if the drum-piece
separator can't find a kick, we just won't emit candidates for the kick lane
and the LLM has to infer the kick pattern from context (in practice it can't,
so this is mostly a "log and let the user retry" path).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)


# Map of substrings found in the separated stem filenames -> Drumjot DSL
# pitch letter. Aligned with `src/midi/gm.ts` defaults so a downstream
# `fromMidi` would land on the same pitches.
STEM_NAME_TO_PITCH: dict[str, str] = {
    "kick": "k",
    "snare": "s",
    "hihat": "h",
    "hi-hat": "h",
    "hat": "h",
    "ride": "d",  # 'd' for ride - avoids the `:r` rim-shot modifier clash
    "crash": "c",
    "tom": "t",
}


@dataclass
class SeparatedStems:
    """Outputs of the two-stage separation."""

    drum_stem: Path
    per_instrument: dict[str, Path]  # pitch letter -> wav path


class Separator:
    """Two-stage drum separator. Models are loaded eagerly by `load()` at
    application startup so the first `/transcribe` call doesn't pay
    model-load latency.

    Model weights are downloaded into `settings.models_dir` (mounted as a
    Docker volume so they persist across container restarts).
    """

    def __init__(self) -> None:
        self._stage1 = None
        self._stage2 = None

    def load(self) -> None:
        """Idempotently load both separator models.

        Called once at container startup from the FastAPI lifespan hook,
        and again defensively from `separate()` so callers that bypass
        the lifespan (e.g. unit tests) still work.
        """
        if self._stage1 is not None and self._stage2 is not None:
            return
        # Local import: pulls in heavy ML deps; only needed in worker processes.
        from audio_separator.separator import Separator as AS

        common = dict(
            output_dir=None,  # set per-call
            model_file_dir=str(settings.models_dir),
            # fp16 autocast: known regression in Demucs's CUDA kernels on
            # newer driver/torch combos where inference silently produces
            # all-zeros output (correct shape, zero values). Disable to
            # force fp32 — slower but produces correct results. Toggle
            # back to True once Demucs / torch resolve the upstream issue.
            use_autocast=False,
        )

        t0 = time.perf_counter()
        log.info("Loading stage 1 separator (%s) ...", settings.demucs_model)
        self._stage1 = AS(**common)
        self._stage1.load_model(model_filename=settings.demucs_model)
        log.info(
            "Stage 1 ready in %.2fs (%s)",
            time.perf_counter() - t0,
            settings.demucs_model,
        )

        t1 = time.perf_counter()
        log.info("Loading stage 2 separator (%s) ...", settings.drum_pieces_model)
        self._stage2 = AS(**common)
        self._stage2.load_model(model_filename=settings.drum_pieces_model)
        log.info(
            "Stage 2 ready in %.2fs (%s)",
            time.perf_counter() - t1,
            settings.drum_pieces_model,
        )
        log.info(
            "Separator ready (total %.2fs).",
            time.perf_counter() - t0,
        )

    def separate(self, audio_path: Path, work_dir: Path) -> SeparatedStems:
        # Idempotent: if `load()` was already called from lifespan, this is
        # a no-op fast-path. Otherwise we lazily load here so unit tests
        # and one-off scripts still work.
        self.load()
        assert self._stage1 is not None and self._stage2 is not None

        stage1_dir = work_dir / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        self._stage1.output_dir = str(stage1_dir)

        log.info("Stage 1: extracting drum stem from %s", audio_path.name)
        stems_paths = [Path(p) for p in self._stage1.separate(str(audio_path))]
        drum_candidates = [
            p for p in stems_paths if "drum" in p.stem.lower()
        ]
        if not drum_candidates:
            raise RuntimeError(
                f"Stage 1 separator produced no drum stem. "
                f"Got: {[p.name for p in stems_paths]}"
            )
        drum_stem = drum_candidates[0]

        stage2_dir = work_dir / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        self._stage2.output_dir = str(stage2_dir)

        log.info("Stage 2: splitting drum stem into pieces")
        piece_paths = [Path(p) for p in self._stage2.separate(str(drum_stem))]

        per_instrument: dict[str, Path] = {}
        for path in piece_paths:
            pitch = _pitch_for_stem_name(path.stem)
            if pitch is None:
                log.info("Skipping unrecognised stem %s", path.name)
                continue
            # First-seen wins so we deterministically pick e.g. crash over
            # china if both are present (rare with the default model).
            per_instrument.setdefault(pitch, path)

        log.info("Recovered %d pitches: %s", len(per_instrument), sorted(per_instrument))
        return SeparatedStems(drum_stem=drum_stem, per_instrument=per_instrument)


def _pitch_for_stem_name(stem_name: str) -> str | None:
    name = stem_name.lower()
    for needle, pitch in STEM_NAME_TO_PITCH.items():
        if needle in name:
            return pitch
    return None
