"""Smoke test for the transcribe runner's debug-bundle replay path.

Gated on a local debug bundle (machine-specific, like the frontend's
E2E_DEBUG_BUNDLE); skipped when absent so CI / fresh checkouts stay green.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from app.comms.core import CancelToken
from app.comms.protocol import PathRef, RequestArgs, RequestMessage
from app.comms.transcribe_runner import TranscribeRunner

BUNDLE = Path(
    os.environ.get("DRUMJOT_TEST_BUNDLE", "/codebox-workspace/drumjot/outputs/itte.zip")
)


@pytest.mark.skipif(not BUNDLE.exists(), reason=f"no debug bundle at {BUNDLE}")
def test_replays_debug_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRUMJOT_OUTPUTS_DIR", str(tmp_path))
    progress: list[tuple[str, float]] = []

    async def emit(stage: str, frac: float, message: str | None = None) -> None:
        progress.append((stage, frac))

    req = RequestMessage(
        type="request",
        id="t1",
        op="transcribe",
        args=RequestArgs(audio=PathRef(kind="path", path=str(BUNDLE))),
    )
    artifacts = asyncio.run(TranscribeRunner().run(req, emit, CancelToken()))

    roles = [a.role for a in artifacts]
    assert "midi" in roles
    assert "stem" in roles
    assert progress[-1][1] == 1.0

    midi = next(a for a in artifacts if a.role == "midi")
    assert isinstance(midi.ref, PathRef)
    out = Path(midi.ref.path)
    assert out.exists() and out.stat().st_size > 0
