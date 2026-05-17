"""Thin HTTP client over POST /transcribe.

The benchmark harness is deliberately decoupled from the service's
heavyweight ML deps (audio-separator, librosa, madmom, torch) — we just
POST the audio file to a running service and read back the DSL. That
also matches how the web UI exercises the service, so the benchmark
numbers reflect the same path users hit in production.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    """Knobs that map 1:1 to the web UI's controls in the toolbar."""

    refine: bool = True
    lint: bool = True
    best_of_k: int = 1


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    jot_dsl: str
    initial_score: float | None
    final_score: float | None
    elapsed_seconds: float


def transcribe_file(
    audio_path: Path,
    options: TranscribeOptions,
    service_url: str = "http://localhost:8001",
    timeout_seconds: float = 600.0,
) -> TranscribeResult:
    """POST one audio file to /transcribe and parse the response."""
    url = service_url.rstrip("/") + "/transcribe"
    # Service caps uploads at 200 MB (main.py:132); surface a clear
    # error here rather than letting httpx swallow the 413.
    size = audio_path.stat().st_size
    if size > 200_000_000:
        raise ValueError(
            f"{audio_path.name} is {size / 1e6:.1f} MB, over the 200 MB /transcribe limit."
        )

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {
            "refine": "true" if options.refine else "false",
            "lint": "true" if options.lint else "false",
            "best_of_k": str(options.best_of_k),
            # include_candidates is heavy and we don't need it for scoring.
            "include_candidates": "false",
            "debug": "false",
        }
        # Long timeout: refinement + best-of-K can each take a minute on CPU.
        resp = httpx.post(url, files=files, data=data, timeout=timeout_seconds)
    resp.raise_for_status()
    payload = resp.json()

    refinement = payload.get("refinement") or {}
    return TranscribeResult(
        jot_dsl=payload["jot_dsl"],
        initial_score=refinement.get("initial_score"),
        final_score=refinement.get("final_score"),
        elapsed_seconds=float(refinement.get("elapsed_seconds", 0.0)),
    )


def wait_for_service(service_url: str, timeout_seconds: float = 5.0) -> None:
    """Cheap readiness probe — raises if /health doesn't return 200."""
    url = service_url.rstrip("/") + "/health"
    resp = httpx.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "ok":
        raise RuntimeError(f"{url} returned status={body.get('status')!r}")
