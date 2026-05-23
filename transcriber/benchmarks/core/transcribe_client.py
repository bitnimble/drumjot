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
    # "dsl" (default) scores the returned Jot; "filter" scores the
    # returned prediction.mid directly (no Jot in the loop).
    transcribe_mode: str = "dsl"
    # Onset backend to exercise: "librosa" (default) or "adtof". Lets the
    # harness run the A/B by flipping this between two seeded runs.
    onset_backend: str = "librosa"


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    jot_dsl: str
    initial_score: float | None
    final_score: float | None
    elapsed_seconds: float
    # Set only in `filter` mode: URL path (no host) of the predicted
    # MIDI. Compose against `service_url` to download it.
    prediction_midi_url: str | None = None


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
            "transcribe_mode": options.transcribe_mode,
            "onset_backend": options.onset_backend,
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
        jot_dsl=payload.get("jot_dsl", ""),
        initial_score=refinement.get("initial_score"),
        final_score=refinement.get("final_score"),
        elapsed_seconds=float(refinement.get("elapsed_seconds", 0.0)),
        prediction_midi_url=payload.get("prediction_midi_url"),
    )


def fetch_prediction_midi(
    service_url: str,
    midi_url_path: str,
    timeout_seconds: float = 60.0,
) -> bytes:
    """Download the `filter`-pathway prediction MIDI.

    `midi_url_path` is the host-less path from the response
    (`/outputs/<id>/prediction.mid`), composed against `service_url`.
    """
    url = service_url.rstrip("/") + midi_url_path
    resp = httpx.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.content


def wait_for_service(service_url: str, timeout_seconds: float = 5.0) -> None:
    """Cheap readiness probe — raises if /health doesn't return 200."""
    url = service_url.rstrip("/") + "/health"
    resp = httpx.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "ok":
        raise RuntimeError(f"{url} returned status={body.get('status')!r}")
