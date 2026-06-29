"""The `alignLyrics` runner: CTC forced alignment without the HTTP layer.

`mix` runs the vocals separator first (Separator.run_vocals), `vocals` aligns the
supplied stem directly. Needs the `lyrics` capability (+ `lyrics-ja` for
Japanese); the aligner provisions its model on first realign. Returns the
word-timed lines as structured `data` (no file artifacts). Heavy work runs off
the event loop.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .core import CancelToken, EmitProgress, RunnerResult
from .protocol import PathRef, RequestMessage


class AlignLyricsRunner:
    def __init__(self) -> None:
        # Lazily-built Separator for the mix flow, cached so repeat aligns don't
        # reload the vocals model. `Any` to avoid importing the torch stack here.
        self._separator: Any | None = None

    async def run(
        self,
        request: RequestMessage,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> RunnerResult:
        source = request.args.audio
        if not isinstance(source, PathRef):
            raise ValueError("alignLyrics needs a local file path (remote upload unsupported here)")
        params = request.args.params
        kind = str(params.get("kind", "mix"))
        if kind not in ("mix", "vocals"):
            raise ValueError(f"unknown alignLyrics kind: {kind!r}")
        raw_lines = params.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError("alignLyrics requires `lines` (a list of {startSec, text})")
        lang = params.get("language")
        language = lang if isinstance(lang, str) and lang else None

        await emit("aligning", 0.1, kind)
        lines_json = await asyncio.to_thread(
            self._run_align, Path(source.path), kind, raw_lines, language
        )
        cancel.check()
        await emit("done", 1.0, None)
        return RunnerResult(data={"lines": lines_json})

    def _run_align(
        self,
        audio_path: Path,
        kind: str,
        raw_lines: list[Any],
        language: str | None,
    ) -> list[dict[str, Any]]:
        from app.pipeline.lyrics_align import InputLine, get_aligner, lines_to_json

        input_lines = [
            InputLine(start_sec=float(e["startSec"]), text=str(e["text"]))
            for e in raw_lines
            if isinstance(e, dict) and "startSec" in e and "text" in e
        ]
        vocals_path = audio_path
        work: Path | None = None
        if kind == "mix":
            from app.pipeline.separate import Separator

            if self._separator is None:
                self._separator = Separator()
            work = Path(tempfile.mkdtemp(prefix="drumjot_lyrics_"))
            vocals = self._separator.run_vocals(audio_path, work)
            if vocals is None:
                shutil.rmtree(work, ignore_errors=True)
                raise RuntimeError("vocals separator produced no vocals stem")
            vocals_path = vocals
        try:
            lines = get_aligner().realign_text(vocals_path, input_lines, language)
            return lines_to_json(lines)
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
