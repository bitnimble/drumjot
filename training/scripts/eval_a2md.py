"""Evaluate on A2MD -- a BIGGER real-domain cr/rd eval than MDB.

MDB has only 17 crash songs (eval floor ±0.13); A2MD has ~290 separated-cymbal-stem
tracks with crashes (5566 onsets) and is NOT in our training pool, so it's a clean,
much larger real-domain cymbal benchmark. Uses the pre-separated
`a2md_sep/audio/perstem/{h,c}/<id>.flac` stems + the full-song aligned MIDI.

A2MD labels are APPROXIMATE (aligned arrangements, not hand-transcribed), so:
  * `--min-crash N` skips crash-sparse tracks (noisy per-song F1),
  * `--clean-support F` keeps a track's c-stem only if >= F of its crash labels land
    on a real onset-strength peak (transient-alignment gate via clean.support_score;
    this also fairly drops separation-missed crashes the model couldn't detect).
For PAIRED lever A/Bs the residual label noise largely cancels (same labels per arm).
`--dump-tracks` writes the kept (track, n_crash, support) list to reuse as the eval set.

  PYTHONPATH=dsp:training MODELS_DIR=<cache> python3 training/scripts/eval_a2md.py \
      --checkpoint <ckpt> [--a2md-root /codebox-workspace/datasets/a2md_sep] \
      [--lanes rd,cr] [--min-crash 5] [--clean-support 0.5] [--dump-tracks clean.json]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/

from drumjot_training import a2md, clean, embeddings, inference, runtime  # noqa: E402
from drumjot_training.parampred import eval_gap, report  # noqa: E402


def crash_support(stem: Path, cr_onsets, window_s: float) -> float:
    """Fraction of crash labels on a real c-stem transient (clean.support_score with a
    relative 60th-pct onset-strength floor, matching the training support gate)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(stem), sr=embeddings.HB_SR, mono=True)
    if y.size == 0:
        return 0.0
    env = librosa.onset.onset_strength(y=y, sr=sr)
    fps = sr / 512.0  # librosa default hop_length
    floor = float(np.percentile(env, 60.0))
    return clean.support_score({"cr": list(cr_onsets)}, env, fps,
                               window_s=window_s, support_floor=floor)["fraction"]


def main():
    ap = argparse.ArgumentParser(description="Eval cr/rd on A2MD (bigger real-domain cymbal set)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--a2md-root", default="/codebox-workspace/datasets/a2md_sep")
    ap.add_argument("--lanes", default="rd,cr")
    ap.add_argument("--min-crash", type=int, default=5, help="skip tracks with fewer cr labels")
    ap.add_argument("--clean-support", type=float, default=0.0,
                    help="keep c-stem only if >= this fraction of crash labels land on a "
                         "transient (0 = gate off)")
    ap.add_argument("--support-window", type=float, default=0.05)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--window-seconds", type=float, default=30.0)
    ap.add_argument("--max-tracks", type=int, default=None, help="cap tracks scored (smoke/debug)")
    ap.add_argument("--dump-tracks", default=None, help="write kept (track,n_crash,support) JSON")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import torch

    runtime.tee_stdio(args.log)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    req = {ln.strip() for ln in args.lanes.split(",")}
    model, meta = inference.load_model(args.checkpoint, device)
    encoder = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])

    # group per-stem clips back into tracks: midi_path -> {pitch: audio_path}
    by_track: dict[Path, dict[str, Path]] = defaultdict(dict)
    for c in a2md.perstem_index(args.a2md_root):
        by_track[c.midi_path][c.pitch] = c.audio_path
    print(f"{len(by_track)} A2MD tracks; lanes={sorted(req)} "
          f"min_crash={args.min_crash} clean_support={args.clean_support}", flush=True)

    records: list = []
    used = skip_minc = skip_clean = 0
    kept: list = []
    for midi, pitches in sorted(by_track.items()):
        gt_full = a2md.drum_onsets_by_lane(midi)
        if args.max_seconds is not None:
            gt_full = {ln: [t for t in ts if t < args.max_seconds] for ln, ts in gt_full.items()}
        n_cr = len(gt_full.get("cr", []))
        if "cr" in req and n_cr < args.min_crash:
            skip_minc += 1
            continue
        got = False
        support = None
        for pitch in ("h", "c"):
            restrict = set(a2md.PERSTEM_TO_LANES.get(pitch, ())) & req
            stem = pitches.get(pitch)
            if not restrict or stem is None or not Path(stem).exists():
                continue
            if pitch == "c" and args.clean_support > 0.0:
                support = crash_support(stem, gt_full.get("cr", []), args.support_window)
                if support < args.clean_support:
                    skip_clean += 1
                    continue
            probs, fps = inference.stitched_probs(
                stem, model, meta, encoder, args.max_seconds, args.window_seconds)
            records += eval_gap.lane_gap_records(
                probs, fps, meta["lanes"], meta["thresholds"], gt_full,
                default_threshold=meta["peak_threshold"], tolerance=meta["onset_tolerance_s"],
                restrict_lanes=restrict)
            got = True
        if got:
            used += 1
            kept.append({"track": midi.stem, "n_crash": n_cr, "support": support})
            if args.max_tracks is not None and used >= args.max_tracks:
                break
        n_seen = used + skip_minc + skip_clean
        if n_seen > 0 and n_seen % 50 == 0:
            print(f"  ..{used} scored / {skip_minc} sparse / {skip_clean} c-stems unclean", flush=True)

    if not records:
        print("no records (check --a2md-root has audio/perstem/{h,c})", flush=True)
        return
    if args.dump_tracks:
        Path(args.dump_tracks).write_text(json.dumps(kept, indent=2))
    gaps = report.aggregate(records)
    order = [ln for ln in ("hc", "ho", "rd", "cr") if ln in gaps]
    print(f"\n{used} tracks scored ({skip_minc} sparse tracks, {skip_clean} c-stems gated unclean)\n"
          + report.format_report(gaps, order), flush=True)


if __name__ == "__main__":
    main()
