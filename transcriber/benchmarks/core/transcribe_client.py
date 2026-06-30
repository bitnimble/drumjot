"""Thin HTTP client over POST /transcribe.

The benchmark harness is deliberately decoupled from the service's
heavyweight ML deps (audio-separator, librosa, beat-this, torch), we just
POST the audio file to a running service and read back the predicted
MIDI. That also matches how the web UI exercises the service, so the
benchmark numbers reflect the same path users hit in production.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    """Knobs that map 1:1 to the web UI's controls in the toolbar."""

    # Which audio is fed into the beat tracker.
    beat_input: str = "full_mix"


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    elapsed_seconds: float
    # URL path (no host) of the predicted MIDI. Compose against
    # `service_url` to download it.
    prediction_midi_url: str | None


def transcribe_file(
    audio_path: Path,
    options: TranscribeOptions,
    service_url: str = "http://localhost:8001",
    timeout_seconds: float = 600.0,
) -> TranscribeResult:
    """POST one audio file to /transcribe and parse the NDJSON stream.

    The service streams stage progress as NDJSON; we only care about the
    final `result` envelope which carries the response body.
    """
    url = service_url.rstrip("/") + "/transcribe"
    # Service caps uploads at 200 MB; surface a clear error here rather
    # than letting httpx swallow the 413.
    size = audio_path.stat().st_size
    if size > 200_000_000:
        raise ValueError(
            f"{audio_path.name} is {size / 1e6:.1f} MB, over the 200 MB /transcribe limit."
        )

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {
            "beat_input": options.beat_input,
            # include_candidates is heavy and we don't need it for scoring.
            "include_candidates": "false",
            "debug": "false",
        }
        # Long timeout: separation + filter LLM can each take ~minute on CPU.
        with httpx.stream(
            "POST", url, files=files, data=data, timeout=timeout_seconds
        ) as resp:
            resp.raise_for_status()
            payload: dict | None = None
            elapsed_seconds = 0.0
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "result":
                    payload = event.get("data") or {}
                elif kind == "stage" and event.get("phase") == "end":
                    elapsed_seconds += float(event.get("elapsed_seconds", 0.0))
                elif kind == "error":
                    raise RuntimeError(
                        f"transcribe failed (status={event.get('status_code')}): "
                        f"{event.get('message')}"
                    )

    if payload is None:
        raise RuntimeError("transcribe stream ended without a result event")

    return TranscribeResult(
        elapsed_seconds=elapsed_seconds,
        prediction_midi_url=payload.get("prediction_midi_url"),
    )


def fetch_prediction_midi(
    service_url: str,
    midi_url_path: str,
    timeout_seconds: float = 60.0,
) -> bytes:
    """Download the predicted MIDI.

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
