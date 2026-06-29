"""Tests for the stdio control-protocol adapter + models (app.comms)."""
from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.comms.core import Cancelled, CancelToken, Registry
from app.comms.protocol import (
    CLIENT_MESSAGE_ADAPTER,
    PROTOCOL_VERSION,
    SERVER_MESSAGE_ADAPTER,
    RequestMessage,
)
from app.comms.runners import build_registry
from app.comms.stdio_adapter import StdioAdapter


def _request(job_id: str, path: str) -> str:
    return RequestMessage(
        type="request",
        id=job_id,
        op="transcribe",
        args={"audio": {"kind": "path", "path": path}},
    ).model_dump_json()


def _run_adapter(registry: Registry, frames: list[str]) -> list[dict]:
    stdin = io.StringIO("".join(f + "\n" for f in frames))
    stdout = io.StringIO()
    asyncio.run(StdioAdapter(registry, stdin=stdin, stdout=stdout).run())
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_client_message_round_trip() -> None:
    req = RequestMessage(
        type="request",
        id="j1",
        op="transcribe",
        args={"audio": {"kind": "path", "path": "/a.mp3"}, "params": {"quantise": True}},
    )
    parsed = CLIENT_MESSAGE_ADAPTER.validate_json(req.model_dump_json())
    assert parsed == req
    assert req.v == PROTOCOL_VERSION


def test_echo_runner_emits_progress_then_result() -> None:
    out = _run_adapter(build_registry(), [_request("j1", "/song.mp3")])
    types = [m["type"] for m in out]
    assert types.count("progress") == 3
    assert types[-1] == "result"
    assert out[-1]["artifacts"][0]["ref"]["path"] == "/song.mp3"
    for m in out:  # every emitted frame is a valid server message
        SERVER_MESSAGE_ADAPTER.validate_python(m)


def test_unknown_op_errors() -> None:
    out = _run_adapter({}, [_request("j2", "/x.mp3")])  # empty registry
    assert out[-1]["type"] == "error"
    assert out[-1]["code"] == "unknown_op"


def test_malformed_line_is_skipped() -> None:
    out = _run_adapter(build_registry(), ["not json{", _request("j3", "/y.mp3")])
    assert out[-1]["type"] == "result"


def test_cancel_token() -> None:
    tok = CancelToken()
    tok.check()  # no raise before cancel
    tok.cancel()
    assert tok.cancelled
    with pytest.raises(Cancelled):
        tok.check()


def test_sidecar_subprocess_round_trip() -> None:
    """Spawn the sidecar exactly as the Rust broker does (`python -m
    app.sidecar`), pipe a request over stdin, read frames off stdout."""
    transcriber_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "app.sidecar"],
        input=_request("sub1", "/song.mp3") + "\n",
        capture_output=True,
        text=True,
        cwd=transcriber_root,
        timeout=60,
        check=False,
    )
    frames = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert frames, f"no frames on stdout; stderr={proc.stderr}"
    assert frames[-1]["type"] == "result"
    assert frames[-1]["artifacts"][0]["ref"]["path"] == "/song.mp3"
