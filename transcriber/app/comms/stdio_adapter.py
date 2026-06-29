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
        async with self._write_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    async def _handle_request(self, req: RequestMessage) -> None:
        token = CancelToken()
        self._tokens[req.id] = token

        async def emit(stage: str, frac: float, message: str | None = None) -> None:
            await self._send(
                ProgressMessage(id=req.id, stage=stage, frac=frac, message=message)
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
                msg = CLIENT_MESSAGE_ADAPTER.validate_python(json.loads(line))
            except (json.JSONDecodeError, ValidationError):
                continue  # tolerate a malformed client line rather than die
            if isinstance(msg, RequestMessage):
                task = asyncio.create_task(self._handle_request(msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            elif isinstance(msg, CancelMessage):
                self._handle_cancel(msg)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
