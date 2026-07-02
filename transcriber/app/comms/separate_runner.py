"""The `separate` runner: stem separation without the rest of the transcribe
pipeline.

`stems_all` splits a mix into the drum stem + drumless backing (BS-Roformer);
`stems_per` splits a drum stem into its per-instrument stems (MDX23C). Needs the
`separation` capability (torch + audio-separator); `Separator.load()` provisions
the models on first use. Outputs land in the asset-scoped outputs dir so the
webview can load them as audio tracks. The heavy work runs off the event loop.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from .core import CancelToken, EmitProgress, RunnerResult
from .protocol import Artifact, PathRef, RequestMessage
from .transcribe_runner import _input_id, _outputs_dir

# Valid `separate` stages: the two Separator passes (mix→drums+backing, drum
# stem→per-instrument).
_STAGES = ("stems_all", "stems_per")


class SeparateRunner:
    async def run(
        self,
        request: RequestMessage,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> RunnerResult:
        source = request.args.audio
        if not isinstance(source, PathRef):
            raise ValueError("separate needs a local file path (remote upload unsupported here)")
        stage = str(request.args.params.get("stage", "stems_all"))
        if stage not in _STAGES:
            raise ValueError(f"unknown separate stage: {stage!r}")

        path = Path(source.path)
        out = _outputs_dir() / _input_id(path) / stage
        out.mkdir(parents=True, exist_ok=True)

        await emit("separating", 0.1, stage)
        # No cooperative cancel mid-separation (the model call isn't interruptible);
        # the broker kills the process on cancel, and we discard a late result.
        named = await asyncio.to_thread(_run_separation, path, stage, out)
        cancel.check()
        await emit("done", 1.0, None)
        return RunnerResult(
            artifacts=[
                Artifact(role=role, name=name, ref=PathRef(kind="path", path=str(p)))
                for (name, role, p) in named
            ]
        )


def _run_separation(audio_path: Path, stage: str, out_dir: Path) -> list[tuple[str, str, Path]]:
    """Run the separator (lazy-imports the torch stack). Returns
    (name, artifact-role, published-path) per produced stem."""
    from app.pipeline.separate import Separator

    work = Path(tempfile.mkdtemp(prefix="drumjot_sep_"))
    sep = Separator()
    produced: list[tuple[str, str, Path]] = []
    try:
        if stage == "stems_all":
            sep.load(stems_all=True, stems_per=False)
            res = sep.run_stems_all(audio_path, work, build_no_drums=True)
            produced.append(("drums", "stem", _publish(res.drum_stem, out_dir)))
            if res.no_drums is not None:
                produced.append(("no_drums", "audio", _publish(res.no_drums, out_dir)))
        else:
            sep.load(stems_all=False, stems_per=True)
            res = sep.run_stems_per(audio_path, work, build_residual=False)
            for pitch, stem_path in res.per_instrument.items():
                produced.append((pitch, "stem", _publish(stem_path, out_dir)))
    finally:
        # Stems are published to out_dir above; the scratch dir can go.
        shutil.rmtree(work, ignore_errors=True)
    return produced


def _publish(src: Path, out_dir: Path) -> Path:
    dest = out_dir / Path(src).name
    shutil.copyfile(src, dest)
    return dest
