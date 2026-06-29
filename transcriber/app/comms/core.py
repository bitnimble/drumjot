"""Transport-agnostic backend core.

A `Runner` turns one request into a list of artifacts, reporting progress
through a callback. It knows nothing about stdio / HTTP / Tauri, the transport
adapter owns that. This is the seam that lets the same backend logic serve the
local stdio sidecar and a remote HTTP/WS deployment.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from .protocol import Artifact, RequestMessage

# (stage, frac in 0..1, optional message) -> awaitable
EmitProgress = Callable[[str, float, "str | None"], Awaitable[None]]


class Cancelled(Exception):
    """Raised by a runner that observes cancellation via its CancelToken."""


class CancelToken:
    """Cooperative cancellation: a long-running runner polls `check()` (or
    `cancelled`) at safe points and bails out by raising `Cancelled`."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        if self._cancelled:
            raise Cancelled


class Runner(Protocol):
    async def run(
        self,
        request: RequestMessage,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> list[Artifact]: ...


# op name -> the runner that handles it
Registry = dict[str, Runner]
