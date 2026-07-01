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
from app.comms.runners import EchoRunner
from app.comms.stdio_adapter import StdioAdapter

# All real ops now drive real (torch + model-download) runners, so the in-process
# plumbing tests wire an explicit EchoRunner under the `separate` op key.
ECHO_REGISTRY: Registry = {"separate": EchoRunner()}


def _request(job_id: str, path: str, op: str = "separate", params: dict | None = None) -> str:
    return RequestMessage(
        type="request",
        id=job_id,
        op=op,
        args={"audio": {"kind": "path", "path": path}, "params": params or {}},
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
    out = _run_adapter(ECHO_REGISTRY, [_request("j1", "/song.mp3")])
    types = [m["type"] for m in out]
    assert types.count("progress") == 3
    assert types[-1] == "result"
    assert out[-1]["artifacts"][0]["ref"]["path"] == "/song.mp3"
    for m in out:  # every emitted frame is a valid server message
        SERVER_MESSAGE_ADAPTER.validate_python(m)


def test_runner_result_carries_data() -> None:
    """A runner returning RunnerResult surfaces its `data` on the result frame
    (the alignLyrics mechanism)."""
    from app.comms.core import RunnerResult

    class DataRunner:
        async def run(self, request, emit, cancel) -> RunnerResult:  # type: ignore[no-untyped-def]
            await emit("working", 0.5, None)
            return RunnerResult(artifacts=[], data={"lines": [{"startSec": 0.0, "text": "hi"}]})

    out = _run_adapter({"alignLyrics": DataRunner()}, [_request("d1", "/x.wav", op="alignLyrics")])
    assert out[-1]["type"] == "result"
    assert out[-1]["data"] == {"lines": [{"startSec": 0.0, "text": "hi"}]}
    SERVER_MESSAGE_ADAPTER.validate_python(out[-1])


def test_unknown_op_errors() -> None:
    out = _run_adapter({}, [_request("j2", "/x.mp3")])  # empty registry
    assert out[-1]["type"] == "error"
    assert out[-1]["code"] == "unknown_op"


def test_malformed_line_is_skipped() -> None:
    out = _run_adapter(ECHO_REGISTRY, ["not json{", _request("j3", "/y.mp3")])
    assert out[-1]["type"] == "result"


def test_invalid_request_with_id_gets_error_frame() -> None:
    # A structurally-invalid frame that still carries an id must get a terminal
    # error, not be silently dropped (which would hang the broker waiting for a
    # result that never comes). Missing `args` fails RequestMessage validation.
    bad = json.dumps({"v": 1, "type": "request", "id": "bad1", "op": "separate"})
    out = _run_adapter(ECHO_REGISTRY, [bad])
    assert len(out) == 1
    assert out[0]["type"] == "error"
    assert out[0]["id"] == "bad1"
    assert out[0]["code"] == "bad_request"


def test_nan_progress_frac_is_clamped_not_fatal() -> None:
    # A runner reporting a NaN/out-of-range fraction must not turn the job into a
    # spurious error (or, on the live path, silently drop the frame); the fraction
    # is clamped and the job still completes.
    class NanRunner:
        async def run(self, request, emit, cancel):  # type: ignore[no-untyped-def]
            await emit("working", float("nan"), None)
            await emit("working", 5.0, None)
            return []

    out = _run_adapter({"separate": NanRunner()}, [_request("n1", "/z.wav")])
    assert out[-1]["type"] == "result"
    progress = [m for m in out if m["type"] == "progress"]
    assert progress and all(0.0 <= m["frac"] <= 1.0 for m in progress)


def test_cancel_token() -> None:
    tok = CancelToken()
    tok.check()  # no raise before cancel
    tok.cancel()
    assert tok.cancelled
    with pytest.raises(Cancelled):
        tok.check()


def test_sidecar_subprocess_round_trip() -> None:
    """Spawn the sidecar exactly as the Rust broker does (`python -m
    app.sidecar`), pipe a request over stdin, read frames off stdout. Uses an
    invalid `separate` stage so the runner errors out *before* importing torch or
    downloading any model, this exercises the real registry's dispatch + error
    path over the subprocess without the GPU stack."""
    transcriber_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "app.sidecar"],
        input=_request("sub1", "/song.mp3", params={"stage": "__invalid__"}) + "\n",
        capture_output=True,
        text=True,
        cwd=transcriber_root,
        timeout=60,
        check=False,
    )
    frames = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert frames, f"no frames on stdout; stderr={proc.stderr}"
    assert frames[-1]["type"] == "error"
    assert frames[-1]["id"] == "sub1"
