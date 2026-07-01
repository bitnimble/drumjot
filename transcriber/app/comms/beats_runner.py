"""The `beats` runner: beat + downbeat detection only.

A focused, self-contained op that runs `analyze_beats` (Beat This! -- ONNX by
default) on the input audio and returns the beat/downbeat times as structured
`data`. No separation, onsets, or LLM, so it exercises the ONNX Beat This! model
end-to-end through the sidecar without the full `transcribe` pipeline. The heavy
inference runs off the event loop on a worker thread.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from .core import Cancelled, CancelToken, EmitProgress, RunnerResult
from .protocol import PathRef, RequestMessage


class BeatsRunner:
    async def run(
        self,
        request: RequestMessage,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> RunnerResult:
        source = request.args.audio
        if not isinstance(source, PathRef):
            raise ValueError("beats needs a local file path (remote upload unsupported here)")
        path = Path(source.path)
        await emit("beats", 0.0, path.name)
        data = await asyncio.to_thread(_detect_beats, path)
        if cancel.cancelled:
            raise Cancelled
        await emit("beats", 1.0, None)
        return RunnerResult(artifacts=[], data=data)


def _detect_beats(path: Path) -> dict:
    """Run beat detection and shape the result. Lazy-imports the pipeline so the
    base sidecar stays light until a beats job actually arrives."""
    from app.pipeline.beats import analyze_beats, beat_engine_name

    structure = analyze_beats(path)
    beats = [round(b.time, 6) for b in structure.beats]
    downbeats = [round(b.time, 6) for b in structure.beats if b.beat_in_bar == 1]
    return {
        "beats": beats,
        "downbeats": downbeats,
        "count": len(beats),
        "initialTempo": round(structure.initial_tempo, 3),
        # 'onnx' proves the Beat This! ONNX model ran (vs the torch/librosa paths).
        "engine": beat_engine_name(),
    }
