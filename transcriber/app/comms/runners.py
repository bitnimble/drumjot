"""Runner registry.

`transcribe` -> {@link TranscribeRunner}, `separate` -> {@link SeparateRunner},
`alignLyrics` -> {@link AlignLyricsRunner}, `beats` -> {@link BeatsRunner}.
`EchoRunner` remains as a plumbing stub for tests.
"""
from __future__ import annotations

import asyncio

from .align_lyrics_runner import AlignLyricsRunner
from .beats_runner import BeatsRunner
from .core import CancelToken, EmitProgress, Registry, RunnerResult
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
    ) -> RunnerResult:
        stages = ("received", "processing", "finishing")
        for i, stage in enumerate(stages):
            cancel.check()
            await emit(stage, (i + 1) / len(stages), None)
            await asyncio.sleep(0)
        source = request.args.audio
        path = source.path if isinstance(source, PathRef) else "<remote-upload>"
        return RunnerResult(artifacts=[Artifact(role="midi", ref=PathRef(kind="path", path=path))])


def build_registry() -> Registry:
    return {
        "transcribe": TranscribeRunner(),
        "separate": SeparateRunner(),
        "alignLyrics": AlignLyricsRunner(),
        "beats": BeatsRunner(),
    }
