"""Runner registry.

PARKED: the real per-op runners that drive `app.pipeline.runner.run_pipeline`
(transcribe) and the separation / lyrics-alignment paths need the multi-GB torch
stack and real audio to validate, which this autonomous run can't do. Until then
`build_registry` wires `EchoRunner` under every op so the full request ->
progress -> result path is exercisable end to end (and integration-tested from
the Rust broker). Swap in the real runners once the capability install can
materialise torch on a GPU box.
"""
from __future__ import annotations

import asyncio

from .core import CancelToken, EmitProgress, Registry
from .protocol import Artifact, PathRef, RequestMessage


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
    runner = EchoRunner()
    return {"transcribe": runner, "separate": runner, "alignLyrics": runner}
