"""The `transcribe` runner.

For now it **replays a debug bundle** -- the `.zip` a prior pipeline run
produced -- by extracting its predicted MIDI + per-stem audio into the outputs
dir and returning them as artifact path-refs. This exercises the full
sidecar -> protocol -> artifact-delivery path with *real* pipeline output,
without needing the GPU/torch stack.

Live transcription from raw audio (driving `app.pipeline.runner.run_pipeline`)
needs the installed torch capability + a GPU; that's the next step. The seam is
here: `run()` dispatches a debug-bundle input to `_replay_bundle`, and a raw
audio input would dispatch to a (not-yet-wired) live path.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from .core import CancelToken, EmitProgress
from .protocol import Artifact, PathRef, RequestMessage

MANIFEST_NAME = "debug.json"
MIDI_NAME = "prediction.mid"
# Mapping key for the drumless backing track (mirrors debug_bundle.NO_DRUMS_KEY).
NO_DRUMS_KEY = "no_drums"


def _outputs_dir() -> Path:
    base = os.environ.get("DRUMJOT_OUTPUTS_DIR")
    return Path(base) if base else Path(tempfile.gettempdir()) / "drumjot-outputs"


def _bundle_id(path: Path) -> str:
    """Content-ish id so re-replaying the same bundle reuses its output dir."""
    st = path.stat()
    digest = hashlib.sha1(f"{path}:{st.st_size}:{int(st.st_mtime)}".encode())
    return digest.hexdigest()[:16]


def _is_debug_bundle(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return MANIFEST_NAME in zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


class TranscribeRunner:
    async def run(
        self,
        request: RequestMessage,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> list[Artifact]:
        source = request.args.audio
        if not isinstance(source, PathRef):
            raise ValueError("transcribe needs a local file path (remote upload unsupported here)")
        path = Path(source.path)
        if path.suffix == ".zip" and _is_debug_bundle(path):
            return await self._replay_bundle(path, emit, cancel)
        raise ValueError(
            "live transcription from audio needs the GPU pipeline capability; "
            "pass a debug bundle .zip for now"
        )

    async def _replay_bundle(
        self,
        bundle: Path,
        emit: EmitProgress,
        cancel: CancelToken,
    ) -> list[Artifact]:
        await emit("opening", 0.1, bundle.name)
        out = _outputs_dir() / _bundle_id(bundle)
        out.mkdir(parents=True, exist_ok=True)
        artifacts: list[Artifact] = []
        with zipfile.ZipFile(bundle) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read(MANIFEST_NAME))
            cancel.check()

            if MIDI_NAME in names:
                await emit("midi", 0.4, None)
                midi_path = out / MIDI_NAME
                midi_path.write_bytes(zf.read(MIDI_NAME))
                artifacts.append(
                    Artifact(role="midi", ref=PathRef(kind="path", path=str(midi_path)))
                )

            # `mapping` aliases several keys to the same file (e.g. `d` -> stem_c);
            # dedup by filename so each stem is written once.
            mapping: dict[str, str] = manifest.get("mapping", {})
            audio_files = list(dict.fromkeys(mapping.values()))
            for i, filename in enumerate(audio_files):
                cancel.check()
                if filename not in names:
                    continue
                await emit("audio", 0.4 + 0.5 * (i + 1) / len(audio_files), filename)
                dest = out / filename
                dest.write_bytes(zf.read(filename))
                role = "audio" if filename == f"{NO_DRUMS_KEY}.mp3" else "stem"
                artifacts.append(Artifact(role=role, ref=PathRef(kind="path", path=str(dest))))

        await emit("done", 1.0, None)
        return artifacts
