"""Opt-in debug artifact persistence for /transcribe.

When enabled, every request copies its intermediate files into a stable
per-request subdir on disk so an operator can listen back to the stems,
inspect the LLM input, replay the refinement loop, etc.

Usage:

    sink = DebugSink.for_request(
        base_dir=settings.debug_dir,
        original_filename=file.filename,
    )
    if sink:
        sink.copy_audio("input", in_path)
        sink.copy_audio("stage1/drum_stem", stems.drum_stem)
        for pitch, p in stems.per_instrument.items():
            sink.copy_audio(f"stage2/{pitch}", p)
        sink.write_text("initial.jot", jot_dsl)
        sink.write_json("beats.json", _beats_dump(structure))
        sink.finalize(...)

`DebugSink` is intentionally cheap to construct and forgiving: missing
files, non-serializable objects, or write failures are logged but never
raise (we never want a debug-only persistence bug to fail the actual
transcription).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
import shutil
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Request-scoped current DebugSink. Set by /transcribe at the top of each
# request so deep callees (the LLM wrapper, refinement helpers) can dump
# their prompts without having the sink threaded through their signatures.
# FastAPI's request handlers run in async context where ContextVars are
# request-local, so concurrent /transcribe calls don't see each other's
# sinks. Defaults to None when debug persistence is disabled — call sites
# must handle that.
_CURRENT_DEBUG_SINK: ContextVar["DebugSink | None"] = ContextVar(
    "drumjot_debug_sink", default=None
)


def current_debug_sink() -> "DebugSink | None":
    """Return the request-scoped DebugSink, or None if debug is disabled."""
    return _CURRENT_DEBUG_SINK.get()


def set_current_debug_sink(sink: "DebugSink | None") -> Token:
    """Install `sink` as the request-scoped sink. Returns a Token that
    callers MUST pass to `reset_current_debug_sink` (typically in a
    `finally`) so the ContextVar is restored on exit."""
    return _CURRENT_DEBUG_SINK.set(sink)


def reset_current_debug_sink(token: Token) -> None:
    _CURRENT_DEBUG_SINK.reset(token)


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(name: str) -> str:
    base = _FILENAME_SAFE.sub("_", name).strip("_")
    return base[:64] or "audio"


class DebugSink:
    """Writes intermediate artifacts for a single /transcribe request."""

    def __init__(self, request_dir: Path) -> None:
        self.dir = request_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._started = time.perf_counter()
        # Monotonic counter for LLM-call dumps so files sort in call order.
        self._llm_call_seq = 0
        log.info("Debug artifacts will be written to %s", self.dir)

    @classmethod
    def for_request(
        cls,
        base_dir: Path | None,
        original_filename: str | None,
    ) -> DebugSink | None:
        if base_dir is None:
            return None
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Could not create debug dir %s: %s", base_dir, exc)
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        slug = _slugify(Path(original_filename or "audio").stem)
        request_dir = base_dir / f"{stamp}_{short_id}_{slug}"
        try:
            return cls(request_dir)
        except OSError as exc:
            log.warning("Could not create debug request dir %s: %s", request_dir, exc)
            return None

    # ------------------------------------------------------------------ writes

    def copy_audio(self, name: str, src: Path) -> None:
        """Copy `src` to `<dir>/<name>.<src.suffix>` (or just `<name>` if it
        already carries a suffix). Failures are logged, not raised.
        """
        if src is None or not src.exists():
            return
        dest = self.dir / name
        if not dest.suffix:
            dest = dest.with_suffix(src.suffix)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        except OSError as exc:
            log.warning("Debug copy %s -> %s failed: %s", src, dest, exc)

    def write_text(self, name: str, text: str) -> None:
        dest = self.dir / name
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning("Debug write %s failed: %s", dest, exc)

    def write_bytes(self, name: str, data: bytes) -> None:
        dest = self.dir / name
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError as exc:
            log.warning("Debug write_bytes %s failed: %s", dest, exc)

    def write_json(self, name: str, payload: Any) -> None:
        dest = self.dir / name
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(payload, indent=2, default=_json_default),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            log.warning("Debug write_json %s failed: %s", dest, exc)

    def write_llm_prompt(
        self,
        purpose: str,
        model: str,
        prompt: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist the full hydrated prompt for one LLM call.

        Files land at `<dir>/llm/NN_<purpose>.txt` where NN is a
        zero-padded per-request sequence number. Each file opens with a
        small header (model, char count, optional extra kwargs) followed
        by the raw prompt text — we deliberately avoid wrapping the
        prompt in a markdown fence because the prompt itself can contain
        triple backticks.
        """
        self._llm_call_seq += 1
        seq = f"{self._llm_call_seq:02d}"
        safe_purpose = _slugify(purpose) or "llm"
        header_lines: list[str] = [
            f"# LLM call {seq}: {purpose}",
            f"- model: {model}",
            f"- prompt_chars: {len(prompt)}",
        ]
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                header_lines.append(f"- {key}: {value}")
        body = "\n".join(header_lines) + "\n\n----- PROMPT -----\n" + prompt
        if not body.endswith("\n"):
            body += "\n"
        self.write_text(f"llm/{seq}_{safe_purpose}.txt", body)

    def finalize(self, summary: dict[str, Any]) -> None:
        """Write the request summary (timings, options, scores) last."""
        summary = {
            **summary,
            "elapsed_seconds": round(time.perf_counter() - self._started, 3),
        }
        self.write_json("request.json", summary)


# ---------------------------------------------------------------- serialization


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback that knows about dataclasses, Pydantic v2 models,
    Path, and numpy scalars."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def beats_dump(structure: Any) -> dict[str, Any]:
    """Serialize a `BeatStructure` into a plain dict for `beats.json`.

    Kept here (not in `pipeline/beats.py`) so the beats module stays free
    of debug concerns. Tolerates partial dataclasses.
    """
    try:
        return {
            "initial_tempo": getattr(structure, "initial_tempo", None),
            "initial_time_signature": list(
                getattr(structure, "initial_time_signature", (4, 4))
            ),
            "has_tempo_changes": getattr(structure, "has_tempo_changes", False),
            "has_time_sig_changes": getattr(
                structure, "has_time_sig_changes", False
            ),
            "beats": [
                {
                    "time": round(b.time, 4),
                    "beat_in_bar": b.beat_in_bar,
                    "bar_index": b.bar_index,
                }
                for b in getattr(structure, "beats", [])
            ],
            "bars": [
                {
                    "index": bar.index,
                    "start_time": round(bar.start_time, 4),
                    "end_time": round(bar.end_time, 4),
                    "time_signature": list(bar.time_signature),
                    "tempo_bpm": round(bar.tempo_bpm, 2),
                    "feel": bar.feel,
                    "beats": [round(b.time, 4) for b in bar.beats],
                }
                for bar in getattr(structure, "bars", [])
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not serialize BeatStructure: %s", exc)
        return {"error": str(exc)}


def onsets_dump(
    onsets_by_pitch: dict[str, list[Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Serialize per-pitch OnsetCandidate lists for `onsets.json`."""
    out: dict[str, list[dict[str, Any]]] = {}
    for pitch, cands in onsets_by_pitch.items():
        rows: list[dict[str, Any]] = []
        for c in cands:
            if hasattr(c, "model_dump"):
                rows.append(c.model_dump())
            elif dataclasses.is_dataclass(c) and not isinstance(c, type):
                rows.append(dataclasses.asdict(c))
            else:
                rows.append({"time": getattr(c, "time", None)})
        out[pitch] = rows
    return out
