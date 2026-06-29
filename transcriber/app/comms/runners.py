"""Runner registry.

`transcribe` is wired to {@link TranscribeRunner}, `separate` to
{@link SeparateRunner} (real stem separation). `alignLyrics` still uses
`EchoRunner` until its real runner lands.
"""
from __future__ import annotations

import asyncio

from .core import CancelToken, EmitProgress, Registry
from .protocol import Artifact, PathRef, RequestMessage
from .separate_runner import SeparateRunner
from .transcribe_runner import TranscribeRunner


class EchoRunner:
    """Plumbing stub: emits a few progress frames then returns a path artifact
    pointing back at the input, proving the transport + protocol round trip
    without the ML stack."""

    async def run(
        self,
        request: RequestMessage,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> list[Artifact]:
        stages = ("received", "processing", "finishing")
        for i, stage in enumerate(stages):
            cancel.check()
            await emit(stage, (i + 1) / len(stages), None)
            await asyncio.sleep(0)
        source = request.args.audio
        path = source.path if isinstance(source, PathRef) else "<remote-upload>"
        return [Artifact(role="midi", ref=PathRef(kind="path", path=path))]


def build_registry() -> Registry:
    echo = EchoRunner()
    return {
        "transcribe": TranscribeRunner(),
        "separate": SeparateRunner(),
        "alignLyrics": echo,
    }
