"""Build the debug `.zip` deliverable for one /transcribe request.

The bundle is the headline artifact for "I want to inspect what
happened" workflows: a single file the operator can download, archive,
re-share, or load back into the web UI to reconstitute the score + every
audio track + the per-stage timings + the full log stream of the run.

Layout:

    <slug>_debug.zip
    ├── prediction.mid       # the kept-onsets predicted MIDI (the score)
    ├── note_provenance.json # per-note keep/reject info for the UI
    ├── no_drums.mp3         # backing audio (drumless mix)
    ├── stem_k.mp3 ...       # one per per-instrument stem
    ├── residual.mp3         # drum_stem − sum(stems): aux percussion +
    │                        # separator residue. Diagnostic-only.
    └── debug.json
        {
          "filename": ...,        # original upload filename
          "options": { ... },     # request form params
          "mapping": {            # pitch letter -> filename inside zip
            "no_drums": "no_drums.mp3",
            "k": "stem_k.mp3", ...,
            "residual": "residual.mp3"
          },
          "metadata": { ... },    # TranscribeMetadata-shaped
          "stage_timings": [ ... ],
          "logs": [ ... ]
        }

MP3 (vs the FLACs OutputSink also writes for the existing audio-track URL
contract) is a deliberate choice: a 5-stem debug bundle for a 3-minute
song at 128 kbps lands ~15-25 MB; the FLAC equivalent would be 100+ MB
which is too big to share casually. The transcoding goes through ffmpeg
(already installed for librosa/soundfile).

This module owns *only* the zip-assembly contract. Per-stage encoding +
URL generation lives in `outputs.OutputSink`; the run-log / stage timings
come from `run_log.RunLog`. The HTTP layer wires the three together.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

from app.outputs import OutputSink
from app.pipeline.cymbal_split import STEM_PITCH_ALIASES as CYMBAL_STEM_ALIASES
from app.run_log import RunLog

log = logging.getLogger(__name__)


# Filename of the JSON manifest inside the zip. Constant so the frontend
# loader can look for it by name.
MANIFEST_FILENAME = "debug.json"

# The synthetic key under `mapping` for the drumless backing audio.
# Distinct from the single-letter drum pitches so a future pitch named
# `n` (none currently exist, but conceivable) can't collide.
NO_DRUMS_KEY = "no_drums"

# The synthetic key under `mapping` for the per-instrument residual track
# (drum stem minus the sum of the 5 separated stems). Multi-letter for the
# same reason as `NO_DRUMS_KEY` — pitch letters are single chars in the DSL.
RESIDUAL_KEY = "residual"

# Filename of the MIDI score inside the zip. The frontend converts it
# to a Jot via `src/midi/from_midi.ts`.
PREDICTION_MIDI_FILENAME = "prediction.mid"

# Filename of the per-note debug provenance sidecar — lists every
# detected onset with its filter decision so the UI can show per-note
# debug details + render rejected onsets as ghosts. Surfaced in the
# manifest's top-level `note_provenance` field.
NOTE_PROVENANCE_FILENAME = "note_provenance.json"


def build_debug_zip(
    *,
    output_sink: OutputSink,
    original_filename: str | None,
    options: dict[str, object],
    metadata: dict[str, object] | None,
    predicted_midi: bytes | None,
    note_provenance: dict[str, object] | None,
    per_instrument_stem_pitches: list[str],
    run_log: RunLog | None,
) -> Path | None:
    """Materialise the debug zip under the OutputSink folder.

    Each audio track is transcoded MP3 lazily from the OutputSink's
    already-on-disk FLAC. Stages that didn't run produce no FLAC and are
    silently skipped from the bundle — the manifest's `mapping` only
    includes audio files that actually landed in the zip.

    Returns the path to the written zip, or `None` if no bundle could be
    assembled (e.g. no FLACs and no score).
    """
    mapping: dict[str, str] = {}
    audio_entries: list[tuple[str, Path]] = []  # (zip path, mp3 source)

    no_drums_flac = output_sink.existing_path(NO_DRUMS_KEY)
    if no_drums_flac is not None:
        mp3 = output_sink.save_mp3_from_wav(NO_DRUMS_KEY, no_drums_flac)
        if mp3 is not None:
            audio_entries.append((mp3.name, mp3))
            mapping[NO_DRUMS_KEY] = mp3.name

    for pitch in sorted(set(per_instrument_stem_pitches)):
        flac_name = f"stem_{pitch}"
        flac = output_sink.existing_path(flac_name)
        if flac is None:
            continue
        mp3 = output_sink.save_mp3_from_wav(flac_name, flac)
        if mp3 is None:
            continue
        audio_entries.append((mp3.name, mp3))
        mapping[pitch] = mp3.name

    # Stem aliases: post-separation splits (today only `cymbal_split`)
    # produce multiple output pitches sharing one input stem file. The
    # manifest declares the shared stem under each output pitch's key so
    # the frontend can cluster all sharing pitches under that one audio
    # row without recomputing the relationship; see
    # `cymbal_split.STEM_PITCH_ALIASES`. Skip aliases whose target stem
    # didn't make it into `mapping` (the stem was absent / encoding
    # failed): the alias would dangle anyway.
    for alias_pitch, source_pitch in CYMBAL_STEM_ALIASES.items():
        if alias_pitch in mapping or source_pitch not in mapping:
            continue
        mapping[alias_pitch] = mapping[source_pitch]

    # Residual = drum_stem − sum(per-instrument stems). Diagnostic-only; # carries auxiliary percussion (cowbell, tambourine, …) the 5-class
    # MDX23C separator has no lane for, plus the separator's own
    # reconstruction error. Bundled when present; absent on older runs
    # (or runs where the residual write failed) is fine.
    residual_flac = output_sink.existing_path(RESIDUAL_KEY)
    if residual_flac is not None:
        mp3 = output_sink.save_mp3_from_wav(RESIDUAL_KEY, residual_flac)
        if mp3 is not None:
            audio_entries.append((mp3.name, mp3))
            mapping[RESIDUAL_KEY] = mp3.name

    manifest: dict[str, object] = {
        "filename": original_filename,
        "options": options,
        "mapping": mapping,
        "metadata": metadata or {},
    }
    if predicted_midi is not None:
        manifest["prediction_midi"] = PREDICTION_MIDI_FILENAME
    if note_provenance is not None:
        manifest["note_provenance"] = NOTE_PROVENANCE_FILENAME
    if run_log is not None:
        run_log_payload = run_log.to_dict()
        manifest["started_at"] = run_log_payload["started_at"]
        manifest["elapsed_seconds"] = run_log_payload["elapsed_seconds"]
        manifest["stage_timings"] = run_log_payload["stage_timings"]
        manifest["logs"] = run_log_payload["logs"]

    has_payload = (
        bool(audio_entries)
        or bool(predicted_midi)
        or bool(note_provenance)
        or bool(run_log)
    )
    if not has_payload:
        log.warning("debug_bundle: nothing to bundle; skipping zip")
        return None

    buffer = io.BytesIO()
    # ZIP_DEFLATED on the JSON manifest reclaims most of the headroom the
    # log dump eats; the MP3 entries are already compressed so DEFLATE on
    # those is a no-op but doesn't hurt — kept uniform for simplicity.
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if predicted_midi is not None:
            zf.writestr(PREDICTION_MIDI_FILENAME, predicted_midi)
        if note_provenance is not None:
            zf.writestr(
                NOTE_PROVENANCE_FILENAME,
                json.dumps(note_provenance, indent=2, default=str),
            )
        for zip_name, src in audio_entries:
            try:
                zf.write(src, arcname=zip_name)
            except OSError as exc:
                log.warning("debug_bundle: could not add %s: %s", src, exc)
        zf.writestr(
            MANIFEST_FILENAME,
            json.dumps(manifest, indent=2, default=str),
        )

    return output_sink.save_bytes("debug.zip", buffer.getvalue())
