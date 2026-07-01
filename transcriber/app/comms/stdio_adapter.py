"""Stdio transport adapter.

Reads newline-delimited client frames from stdin, drives the core runner
registry, writes backend frames to stdout. **stdout is the protocol channel
only** -- all logging must go to stderr (the entrypoint configures that), or the
broker will choke on non-JSON lines. A request runs as its own task so a
`cancel` frame that arrives mid-job can reach the running token.
"""
from __future__ import annotations

import asyncio
import json
from typing import TextIO

from pydantic import BaseModel, ValidationError

from .core import Cancelled, CancelToken, Registry, RunnerResult
from .protocol import (
    CLIENT_MESSAGE_ADAPTER,
    CancelMessage,
    ErrorMessage,
    ProgressMessage,
    RequestMessage,
    ResultMessage,
)


class StdioAdapter:
    def __init__(
        self,
        registry: Registry,
        *,
        stdin: TextIO,
        stdout: TextIO,
    ) -> None:
        self._registry = registry
        self._stdin = stdin
        self._stdout = stdout
        self._tokens: dict[str, CancelToken] = {}
        self._write_lock = asyncio.Lock()

    async def _send(self, msg: BaseModel) -> None:
        line = msg.model_dump_json(exclude_none=True)
        loop = asyncio.get_running_loop()
        async with self._write_lock:
            # Off-thread write+flush: a blocking flush on the loop thread (when the
            # broker back-pressures its stdout pipe) would stall the whole loop, so
            # a `cancel` frame the broker just sent couldn't be read until the write
            # unblocked. Keeping the write off the loop preserves cancel/EOF
            # responsiveness under a slow consumer.
            await loop.run_in_executor(None, self._write_line, line)

    def _write_line(self, line: str) -> None:
        self._stdout.write(line + "\n")
        self._stdout.flush()

    async def _handle_request(self, req: RequestMessage, token: CancelToken) -> None:
        async def emit(stage: str, frac: float, message: str | None = None) -> None:
            # Clamp defensively: a NaN or out-of-range fraction makes the
            # ProgressMessage(ge=0, le=1) constructor raise, which on the live
            # transcribe path is swallowed (progress silently lost) and on the
            # direct-emit runners turns the whole job into a spurious `internal`
            # error. Progress is advisory; never let it be fatal.
            safe_frac = 0.0 if frac != frac else min(1.0, max(0.0, frac))
            await self._send(
                ProgressMessage(id=req.id, stage=stage, frac=safe_frac, message=message)
            )

        try:
            runner = self._registry.get(req.op)
            if runner is None:
                await self._send(
                    ErrorMessage(
                        id=req.id,
                        code="unknown_op",
                        message=f"no runner for op {req.op!r}",
                        recoverable=False,
                    )
                )
                return
            result = await runner.run(req, emit, token)
            if isinstance(result, RunnerResult):
                await self._send(
                    ResultMessage(id=req.id, artifacts=list(result.artifacts), data=result.data)
                )
            else:
                await self._send(ResultMessage(id=req.id, artifacts=list(result)))
        except Cancelled:
            await self._send(
                ErrorMessage(id=req.id, code="cancelled", message="job cancelled", recoverable=True)
            )
        except Exception as exc:  # noqa: BLE001 - a terminal frame must always go out
            await self._send(
                ErrorMessage(id=req.id, code="internal", message=str(exc), recoverable=False)
            )
        finally:
            self._tokens.pop(req.id, None)

    def _handle_cancel(self, msg: CancelMessage) -> None:
        token = self._tokens.get(msg.id)
        if token is not None:
            token.cancel()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task[None]] = set()
        while True:
            # Blocking readline off-thread so cancel frames can interleave with
            # an in-flight job.
            line = await loop.run_in_executor(None, self._stdin.readline)
            if line == "":  # EOF
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue  # not JSON at all; nothing to correlate a reply to
            try:
                msg = CLIENT_MESSAGE_ADAPTER.validate_python(raw)
            except ValidationError:
                # A structurally-invalid frame (unknown fields, a future protocol
                # version, a bad op shape). If it still carries a string id, send a
                # terminal error so the broker/frontend promise resolves instead of
                # hanging forever waiting for a result that will never arrive.
                req_id = raw.get("id") if isinstance(raw, dict) else None
                if isinstance(req_id, str):
                    await self._send(
                        ErrorMessage(
                            id=req_id,
                            code="bad_request",
                            message="malformed or unsupported request frame",
                            recoverable=False,
                        )
                    )
                continue
            if isinstance(msg, RequestMessage):
                # Register the cancel token synchronously, before scheduling the
                # task, so a cancel frame read on the very next iteration can't
                # race ahead of the token's registration and be dropped.
                token = CancelToken()
                self._tokens[msg.id] = token
                task = asyncio.create_task(self._handle_request(msg, token))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            elif isinstance(msg, CancelMessage):
                self._handle_cancel(msg)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
