"""Evaluate peak-pick operating points on MDB-Drums -- a SECOND real-domain test
set (real MedleyDB multitracks through our separation), to re-confirm the
deterministic closed-hat self-calibration win beyond the ~6 ParaDB songs.

Uses the pre-separated `mdb-sep/perstem/{h,c}/<track>.flac` stems + MDB subclass
annotations. MDB isn't in the MERT cache, so each stem is encoded fresh (23
tracks x 2 stems). Reports per lane: current (global thresholds) vs determ
(per-song knee self-calibration) vs predict (optional artifact) vs oracle.

  PYTHONPATH=dsp:training MODELS_DIR=<cache> python3 training/scripts/eval_mdb.py \
      --checkpoint <ckpt> [--param-predictor <joblib>] \
      [--mdb-root /codebox-workspace/datasets/MDBDrums] \
      [--sep-root /codebox-workspace/datasets/mdb-sep/perstem]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/

from drumjot_training import embeddings, inference, mdb, runtime, star  # noqa: E402
from drumjot_training.parampred import eval_gap, hybrid, regressor, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Eval current/determ/oracle on MDB-Drums (real-domain)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mdb-root", default="/codebox-workspace/datasets/MDBDrums")
    ap.add_argument("--sep-root", default="/codebox-workspace/datasets/mdb-sep/perstem")
    ap.add_argument("--param-predictor", default=None, help="optional ParamRegressor joblib (predict column)")
    ap.add_argument("--lanes", default="hc,ho,rd,cr")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--window-seconds", type=float, default=30.0)
    ap.add_argument("--legacy-overlap", action="store_true",
                    help="Score with the OLD overlapping center-crop stitch (fp32) "
                         "instead of the default training-aligned windowing "
                         "(non-overlapping plan_windows cuts + fp16). For A/B only.")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import librosa
    import torch

    runtime.tee_stdio(args.log)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    req = {ln.strip() for ln in args.lanes.split(",")}
    model, meta = inference.load_model(args.checkpoint, device)
    encoder = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    predictor = regressor.ParamRegressor.load(args.param_predictor) if args.param_predictor else None

    sep = Path(args.sep_root)
    clips = mdb.index(args.mdb_root)
    print(f"{len(clips)} MDB tracks; lanes={sorted(req)}", flush=True)
    records: list = []
    used = 0
    for clip in clips:
        gt_full = mdb.onsets_by_lane(clip.subclass_ann)
        if args.max_seconds is not None:
            gt_full = {ln: [t for t in ts if t < args.max_seconds] for ln, ts in gt_full.items()}
        got = False
        for pitch in ("h", "c"):
            restrict = set(star.PERSTEM_TO_LANES.get(pitch, ())) & req
            stem = sep / pitch / f"{clip.track}.flac"
            if not restrict or not stem.exists():
                continue
            wave = None
            sr = None
            if predictor is not None:
                wave, sr = librosa.load(str(stem), sr=embeddings.HB_SR, mono=True)
            probs, fps = inference.stitched_probs(
                stem, model, meta, encoder, args.max_seconds, args.window_seconds,
                legacy_overlap=args.legacy_overlap)
            records += eval_gap.lane_gap_records(
                probs, fps, meta["lanes"], meta["thresholds"], gt_full,
                default_threshold=meta["peak_threshold"], tolerance=meta["onset_tolerance_s"],
                restrict_lanes=restrict, predictor=predictor, waveform=wave, sr=sr)
            got = True
        used += got
        print(f"  {clip.track[:42]:42s} {'ok' if got else 'no stems'}", flush=True)

    if not records:
        print("no records (check --sep-root has the perstem stems)", flush=True)
        return
    gaps = report.aggregate(records)
    order = [ln for ln in ("hc", "ho", "rd", "cr") if ln in gaps]
    print(f"\n{used} tracks scored\n" + report.format_report(gaps, order), flush=True)
    det = [g.det_captured_frac for g in gaps.values() if g.det_captured_frac is not None]
    print(f"  mean determ captured {sum(det) / len(det) * 100:+.1f}% of gap" if det else "", flush=True)
    caps = [g.captured for g in gaps.values()]  # captured = predicted_f1 - current_f1 (always float)
    if predictor and caps:  # cross-check the ParaDB-derived hybrid routing on this independent set
        print(f"  mean captured {sum(caps) / len(caps):+.3f} predicted", flush=True)
        print("\n" + hybrid.format_hybrid(gaps, hybrid.DEFAULT_ROUTING, lane_order=order), flush=True)
        print(f"  mean captured {hybrid.captured(gaps, hybrid.DEFAULT_ROUTING):+.3f} hybrid", flush=True)


if __name__ == "__main__":
    main()
