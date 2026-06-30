"""Smoke test for the transcribe runner's debug-bundle replay path.

Gated on a local debug bundle (machine-specific, like the frontend's
E2E_DEBUG_BUNDLE); skipped when absent so CI / fresh checkouts stay green.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.comms import transcribe_runner as tr
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


@pytest.mark.skipif(not BUNDLE.exists(), reason=f"no debug bundle at {BUNDLE}")
def test_sidecar_subprocess_transcribes_bundle(tmp_path: Path) -> None:
    """Full process e2e: spawn `python -m app.sidecar`, send a transcribe
    request for the bundle over stdin, read a `result` with a MIDI artifact."""
    root = Path(__file__).resolve().parents[1]
    req = RequestMessage(
        type="request",
        id="s1",
        op="transcribe",
        args=RequestArgs(audio=PathRef(kind="path", path=str(BUNDLE))),
    ).model_dump_json()
    proc = subprocess.run(
        [sys.executable, "-m", "app.sidecar"],
        input=req + "\n",
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "DRUMJOT_OUTPUTS_DIR": str(tmp_path)},
        timeout=120,
        check=False,
    )
    frames = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert frames, f"no frames; stderr={proc.stderr}"
    assert frames[-1]["type"] == "result", frames[-1]
    assert any(a["role"] == "midi" for a in frames[-1]["artifacts"])


def test_live_transcribe_collects_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live path bridges the pipeline's sync progress to `emit`, writes the
    predicted MIDI, and copies stems out -- validated with the heavy pipeline
    (torch/GPU) monkeypatched out; only a real GPU run is left unvalidated."""
    monkeypatch.setenv("DRUMJOT_OUTPUTS_DIR", str(tmp_path))
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFFfake")
    stem_src = tmp_path / "stem_k.flac"
    stem_src.write_bytes(b"FLACfake")

    def fake_pipeline(audio_path, params, progress, cancel_event):  # noqa: ANN001
        progress({"stage": "onsets", "phase": "start"})
        return tr.LiveResult(midi=b"MThd-fake", stems=[("stem", "k", stem_src)], duration=2.0)

    monkeypatch.setattr(tr, "_run_live_pipeline", fake_pipeline)
    stages: list[str] = []

    async def emit(stage: str, frac: float, message: str | None = None) -> None:
        stages.append(stage)

    req = RequestMessage(
        type="request",
        id="L1",
        op="transcribe",
        args=RequestArgs(audio=PathRef(kind="path", path=str(audio)), params={"quantise": False}),
    )
    artifacts = asyncio.run(TranscribeRunner().run(req, emit, CancelToken()))

    roles = [a.role for a in artifacts]
    assert "midi" in roles and "stem" in roles
    assert "onsets" in stages  # per-stage progress bridged from the worker thread
    midi = next(a for a in artifacts if a.role == "midi")
    assert isinstance(midi.ref, PathRef)
    assert Path(midi.ref.path).read_bytes() == b"MThd-fake"
