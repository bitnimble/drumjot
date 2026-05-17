"""Two-stage drum separation: full mix -> drum stem -> per-instrument stems.

Stage `stems_all` uses Demucs v4 (htdemucs_ft) to extract a drum stem from
the full mix. Stage `stems_per` uses the community MDX23C 6-stem
drum-piece separator (aufr33 / jarredou model) to break the drum stem
into kick / snare / toms / hi-hat / ride / crash.

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
from pathlib import Path

from app.config import settings
from app.debug import current_debug_sink

log = logging.getLogger(__name__)


# Map of stem-token substrings found in the separated stem filenames ->
# Drumjot DSL pitch letter. Aligned with `src/midi/gm.ts` defaults so a
# downstream `fromMidi` would land on the same pitches.
#
# Tokens are wrapped in literal `(...)` because the MDX23C-DrumSep model
# (and Demucs's stems_all output) emit filenames of the shape
# `<title>_(Drums)_htdemucs_ft_(<stem>)_<model>.wav`. Anchoring the
# match on the parenthesised segment avoids false-positive substring
# hits against arbitrary characters elsewhere in the filename.
STEM_NAME_TO_PITCH: dict[str, str] = {
    "(kick)": "k",
    "(snare)": "s",
    "(hihat)": "h",
    "(hi-hat)": "h",
    "(hh)": "h",
    "(hat)": "h",
    "(ride)": "d",  # 'd' for ride - avoids the `:r` rim-shot modifier clash
    "(crash)": "c",
    # Toms tend to ship as the plural token `(toms)` in MDX23C-DrumSep
    # output; keep the singular form too for forward compatibility.
    "(tom)": "t",
    "(toms)": "t",
}


class Separator:
    """Two-stage drum separator. Models are loaded eagerly by `load()` at
    application startup so the first `/transcribe` call doesn't pay
    model-load latency.

    Model weights are downloaded into `settings.models_dir` (mounted as a
    Docker volume so they persist across container restarts).

    The two stages are exposed as independent methods (`run_stems_all`,
    `run_stems_per`) so the pipeline runner can resume from either one
    without having to re-run the other.
    """

    def __init__(self) -> None:
        self._stems_all = None
        self._stems_per = None

    def load(self) -> None:
        """Idempotently load both separator models.

        Called once at container startup from the FastAPI lifespan hook,
        and again defensively from the per-stage methods so callers that
        bypass the lifespan (e.g. unit tests) still work.
        """
        if self._stems_all is not None and self._stems_per is not None:
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
        log.info("Loading stems_all separator (%s) ...", settings.demucs_model)
        self._stems_all = AS(**common)
        self._stems_all.load_model(model_filename=settings.demucs_model)
        log.info(
            "stems_all ready in %.2fs (%s)",
            time.perf_counter() - t0,
            settings.demucs_model,
        )

        t1 = time.perf_counter()
        log.info("Loading stems_per separator (%s) ...", settings.drum_pieces_model)
        self._stems_per = AS(**common)
        self._stems_per.load_model(model_filename=settings.drum_pieces_model)
        log.info(
            "stems_per ready in %.2fs (%s)",
            time.perf_counter() - t1,
            settings.drum_pieces_model,
        )
        log.info(
            "Separator ready (total %.2fs).",
            time.perf_counter() - t0,
        )

    def run_stems_all(self, audio_path: Path, work_dir: Path) -> Path:
        """Extract a drum stem from the full mix. Returns its absolute path.

        Also persists the drum stem (and any sibling stems Demucs emits —
        bass / other / vocals) into the current debug sink under
        `stems_all/`, so the operator can listen back to intermediates
        while later stages are still running.
        """
        self.load()
        assert self._stems_all is not None

        out_dir = work_dir / "stems_all"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._stems_all.output_dir = str(out_dir)

        log.info("stems_all: extracting drum stem from %s", audio_path.name)
        stems_paths = [Path(p) for p in self._stems_all.separate(str(audio_path))]
        drum_candidates = [p for p in stems_paths if "drum" in p.stem.lower()]
        if not drum_candidates:
            raise RuntimeError(
                f"stems_all produced no drum stem. "
                f"Got: {[p.name for p in stems_paths]}"
            )
        drum_stem = drum_candidates[0]

        sink = current_debug_sink()
        if sink is not None:
            sink.copy_audio("stems_all/drum_stem", drum_stem)
            for path in stems_paths:
                if path == drum_stem:
                    continue
                # bass / other / vocals (htdemucs_ft outputs four stems).
                # Useful for confirming the drum stem is actually clean.
                sink.copy_audio(f"stems_all/{path.stem}", path)
        return drum_stem

    def run_stems_per(self, drum_stem: Path, work_dir: Path) -> dict[str, Path]:
        """Split the drum stem into per-instrument stems keyed by pitch."""
        self.load()
        assert self._stems_per is not None

        out_dir = work_dir / "stems_per"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._stems_per.output_dir = str(out_dir)

        log.info("stems_per: splitting drum stem into pieces")
        piece_paths = [Path(p) for p in self._stems_per.separate(str(drum_stem))]

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

        sink = current_debug_sink()
        if sink is not None:
            for pitch, path in per_instrument.items():
                sink.copy_audio(f"stems_per/{pitch}", path)
        return per_instrument


def _pitch_for_stem_name(stem_name: str) -> str | None:
    name = stem_name.lower()
    for needle, pitch in STEM_NAME_TO_PITCH.items():
        if needle in name:
            return pitch
    return None
