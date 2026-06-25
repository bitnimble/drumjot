"""Emit a Drumjot debug bundle from the model's output on a ParaDB map, so the
model's notes can be eyeballed against the per-instrument audio waveforms in the
frontend debug viewer (src/debug_zip.ts). Minimal metadata only (song name +
duration); none of the transcriber's offsets/beat-grid/scores.

Each MDX23C per-instrument stem (kick/snare/hi-hat/cymbals/toms) becomes one
audio track; the model is run on each isolated stem and its kept-lane onsets are
written to a single MIDI, with each DSL pitch bound to its source stem so the
viewer overlays e.g. ride/crash notes on the cymbals waveform.

Reuses the stems cached by eval_paradb.py (run that first with --stems-cache so
<map>.{k,s,h,c,t}.flac exist). Run in the CUDA sandbox (model + MERT + ffmpeg).

Usage:
  python3 make_debug_bundle.py --map <map.zip> --checkpoint <dir> \
      --stems-cache <dir> --out <bundle.zip>
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_paradb import SEP_SR, STEM_TO_LANES, _sum_tracks  # noqa: E402

from drumjot_training import inference, rlrr  # noqa: E402
from drumjot_training.inference import LANE_TO_PITCH  # noqa: E402

# DSL pitch -> GM MIDI note, aligned with transcriber onsets_midi.PITCH_TO_MIDI
# and src/midi/from_midi.ts so the frontend parses the notes back to these lanes.
PITCH_TO_MIDI = {
    "k": 36, "s": 38, "ss": 37, "t": 50, "h": 42, "H": 46,
    "d": 51, "c": 49, "mc": 55, "mp": 56,
}


def _midi_bytes(onsets_by_pitch: dict[str, list[float]], tempo_bpm: float = 120.0) -> bytes:
    """Single GM-drum-channel MIDI at a flat tempo; absolute onset times are
    preserved (what matters for waveform overlay)."""
    import mido

    tpb = 480
    spb = 60.0 / tempo_bpm
    events = []  # (tick, is_on, note, vel)
    for pitch, times in onsets_by_pitch.items():
        note = PITCH_TO_MIDI.get(pitch)
        if note is None:
            continue
        for t in times:
            tick = max(0, int(round(t / spb * tpb)))
            events.append((tick, 1, note, 96))
            events.append((tick + 1, 0, note, 0))
    events.sort(key=lambda e: (e[0], e[1]))  # note_off before note_on at a tick
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    last = 0
    for tick, is_on, note, vel in events:
        tr.append(mido.Message(
            "note_on" if is_on else "note_off", note=note, velocity=vel, channel=9, time=tick - last,
        ))
        last = tick
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="ParaDB map -> Drumjot debug bundle (model notes vs stem waveforms)")
    ap.add_argument("--map", required=True, help="ParaDB map .zip")
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir")
    ap.add_argument("--stems-cache", required=True, help="dir with <map>.{k,s,h,c,t}.flac from eval_paradb")
    ap.add_argument("--out", required=True, help="output bundle .zip")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--window-seconds", type=float, default=30.0)
    ap.add_argument("--full-drum", action="store_true",
                    help="run once on the whole BS-Roformer drum stem (all lanes, one waveform) "
                    "instead of the MDX23C per-instrument split")
    args = ap.parse_args()

    import torch

    map_zip = Path(args.map)
    name = map_zip.stem
    with zipfile.ZipFile(map_zip) as z:
        chart_member = max(
            (n for n in z.namelist() if n.endswith(".rlrr")),
            key=lambda n: rlrr.complexity(json.loads(z.read(n).decode("utf-8", "replace"))),
        )
        chart = json.loads(z.read(chart_member).decode("utf-8", "replace"))
    title = (chart.get("recordingMetadata") or {}).get("title") or name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = inference.load_model(args.checkpoint, device)
    from drumjot_training import embeddings
    encoder = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])  # share across stems

    cache = Path(args.stems_cache)
    onsets_by_pitch: dict[str, list[float]] = {}
    pitch_to_stem: dict[str, str] = {}  # DSL pitch -> stem key (for the audio mapping)
    if args.full_drum:
        # one model pass over the whole BS-Roformer drum stem; keep ALL lanes
        # (no per-instrument isolation). The single drum waveform carries every
        # note row.
        drum_flac = cache / f"{name}.drum.flac"
        if not drum_flac.exists():
            sys.exit(f"no cached drum stem for {name} in {cache} (run eval_paradb.py --stems-cache first)")
        stem_files = {"drum": drum_flac}
        est = inference.transcribe(
            drum_flac, model, meta, encoder, max_seconds=args.max_seconds, window_seconds=args.window_seconds,
        )
        for lane, ts in est.items():
            dsl = LANE_TO_PITCH[lane]
            if ts:
                onsets_by_pitch.setdefault(dsl, []).extend(ts)
            pitch_to_stem[dsl] = "drum"
    else:
        # stem pitch -> its cached flac; run the model on each isolated stem,
        # keeping only that stem's matching lanes.
        stem_files = {p: cache / f"{name}.{p}.flac" for p in STEM_TO_LANES}
        stem_files = {p: f for p, f in stem_files.items() if f.exists()}
        if not stem_files:
            sys.exit(f"no cached stems for {name} in {cache} (run eval_paradb.py --stems-cache first)")
        for spitch, flac in stem_files.items():
            est = inference.transcribe(
                flac, model, meta, encoder, max_seconds=args.max_seconds, window_seconds=args.window_seconds,
            )
            for lane in STEM_TO_LANES[spitch]:
                dsl = LANE_TO_PITCH[lane]
                ts = est.get(lane, [])
                if ts:
                    onsets_by_pitch.setdefault(dsl, []).extend(ts)
                pitch_to_stem[dsl] = spitch  # bind the lane's pitch to this stem's audio
    for ts in onsets_by_pitch.values():
        ts.sort()

    # build the bundle
    import soundfile as sf

    def _to_mp3(src, dst):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", "128k", str(dst)], check=True)

    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as out:
        tdp = Path(td)
        mapping: dict[str, str] = {}

        # The frontend's from_midi (src/midi/gm.ts) FOLDS several GM notes onto
        # one DSL letter: side-stick->s, pedal/open hi-hat->h, splash/china->c,
        # cowbell->b. So the parsed jot only has pitches {k,s,h,t,c,d,...}; the
        # mapping keys MUST be those folded pitches. A key like 'H'/'mc'/'ss'
        # has no matching jot pitch -> the viewer treats it as an orphan stem and
        # renders that file a SECOND time (the duplicate-waveform bug). Mapping
        # keys here are only folded pitches the model actually produced notes for,
        # so every key resolves to a real jot pitch and each stem appears once.
        GM_FOLD = {"ss": "s", "H": "h", "mc": "c", "mp": "b"}
        folded_to_stem = {GM_FOLD.get(p, p): s for p, s in pitch_to_stem.items()}
        folded_notes = {GM_FOLD.get(p, p) for p in onsets_by_pitch}
        needed_stems = {folded_to_stem[fp] for fp in folded_notes if fp in folded_to_stem}
        stem_mp3 = {}
        for spitch in sorted(needed_stems):
            name_mp3 = f"stem_{spitch}.mp3"
            _to_mp3(stem_files[spitch], tdp / name_mp3)
            out.write(tdp / name_mp3, name_mp3)
            stem_mp3[spitch] = name_mp3
        for fp in sorted(folded_notes):
            if folded_to_stem.get(fp) in stem_mp3:
                mapping[fp] = stem_mp3[folded_to_stem[fp]]

        # non-drum backing: the chart's song tracks (for backing+drums maps this
        # is exactly the rest-of-song mix; for full-mix maps it's the whole song).
        # Shown by the viewer as an unmuted standalone track with no notes.
        with zipfile.ZipFile(map_zip) as mz:
            mz.extractall(tdp / "map")
        backing = _sum_tracks(tdp / "map", rlrr.song_tracks(chart), SEP_SR)
        if backing is not None and len(backing):
            import numpy as np
            peak = float(np.max(np.abs(backing)) or 1.0)
            sf.write(str(tdp / "no_drums.wav"), backing / peak * 0.98, SEP_SR)
            _to_mp3(tdp / "no_drums.wav", tdp / "no_drums.mp3")
            out.write(tdp / "no_drums.mp3", "no_drums.mp3")
            mapping["no_drums"] = "no_drums.mp3"

        out.writestr("prediction.mid", _midi_bytes(onsets_by_pitch))
        dur = max((sf.info(str(f)).duration for f in stem_files.values()), default=0.0)
        manifest = {
            "filename": title,
            "mapping": mapping,
            "prediction_midi": "prediction.mid",
            "metadata": {
                "initial_tempo": 120.0,
                "initial_time_signature": [4, 4],
                "duration_seconds": round(dur, 3),
                "stems_used": sorted(k for k in mapping if k != "no_drums"),
            },
            "options": {"source": "make_debug_bundle.py", "checkpoint": args.checkpoint},
        }
        out.writestr("debug.json", json.dumps(manifest, indent=2))

    total = sum(len(v) for v in onsets_by_pitch.values())
    print(f"{title}: {total} notes across {sorted(onsets_by_pitch)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
