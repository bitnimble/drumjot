"""SOTA-comparable ADT eval: 5-class onset F-measure (KD/SD/HH/TT/CY) at +/-50 ms.

Published ADT numbers (ADTOF, the 2025 stem-separation systems, etc.) report
**5-class onset F-measure via mir_eval at +/-50 ms** on MDB-Drums / ENST / RBMA,
folding cymbals to one class. Our usual STAR-val / ParaDB numbers aren't
comparable (custom sets, finer 9-lane split, cleaned GT). This script produces the
apples-to-apples number: run our model -> fold our 9 lanes to the 5-class taxonomy
-> mir_eval vs the benchmark's GT in the same 5 classes.

Prediction = our DEPLOYMENT path: per-instrument stems, model run per stem, keep
only that stem's lanes (PERSTEM_TO_LANES) -- cross-instrument leakage discarded,
exactly like the transcriber. So this is the "given our separation" condition.

  KD = k        SD = s+ss     HH = hc+hp+ho     TT = t      CY = rd+cr

CONDITIONS / CAVEATS (printed in the report too):
- Per-stem isolation discards cross-instrument false positives -> mildly optimistic
  vs SOTA-run-on-the-mix (which pays for leakage). It IS our deployment condition.
- ENST: drummer_3 is held out from TRAINING but was in the val pool (thresholds
  were tuned on it) -> mildly optimistic. MDB/RBMA are PRISTINE (never in our
  train or val) -> the honest SOTA comparison; ENST is the works-now proxy.

  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  scripts/sandbox-run env PYTHONPATH=training:dsp python3 training/scripts/sota_eval.py \
      --checkpoint /codebox-workspace/checkpoints/loss_ab_mixed.pt \
      --dataset enst --out-json /codebox-workspace/sota_eval_enst.json

Needs a GPU + real MERT (encodes the eval audio, not cached) -> run in the sandbox.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))

# fixed 9-lane -> 5-class fold (the published ADT taxonomy)
FOLD5: dict[str, tuple[str, ...]] = {
    "KD": ("k",),
    "SD": ("s", "ss"),
    "HH": ("hc", "hp", "ho"),
    "TT": ("t",),
    "CY": ("rd", "cr"),
}
CLASSES = ("KD", "SD", "HH", "TT", "CY")


def fold5(lane_onsets: dict[str, list[float]]) -> dict[str, list[float]]:
    """Fold 9-lane onsets to the 5 published classes (sorted)."""
    return {c: sorted(t for ln in lanes for t in lane_onsets.get(ln, []))
            for c, lanes in FOLD5.items()}


def _load_checkpoint(path: str, device: str):
    """Return (model, meta, encoder). Handles BOTH the raw loss_ab dict
    {state_dict,thresholds,lanes,hidden,num_layers} AND a checkpoint.py dir."""
    import torch
    from head_capacity_sweep import make_cfg

    from drumjot_training import embeddings
    from drumjot_training.model import MultiLaneHeads

    p = Path(path)
    if p.is_dir():
        from drumjot_training import inference
        model, meta = inference.load_model(path, device=device)
        enc = embeddings.make_encoder(meta["encoder"], meta["encoder_layer"])
        return model, meta, enc
    # raw loss_ab .pt -> rebuild model + synthesise meta from the training config
    ck = torch.load(path, map_location=device)
    cfg = make_cfg(int(ck["hidden"]), int(ck.get("num_layers", 2)), 3e-4)
    in_dim = embeddings.feat_dim(cfg.high_band)
    model = MultiLaneHeads(in_dim=in_dim, hidden=int(ck["hidden"]),
                           num_layers=int(ck.get("num_layers", 2)), lane_names=ck["lanes"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(device)
    meta = {
        "lanes": list(ck["lanes"]), "thresholds": ck["thresholds"],
        "encoder": cfg.encoder, "encoder_layer": cfg.encoder_layer, "encoder_fps": cfg.encoder_fps,
        "high_band": cfg.high_band, "in_dim": in_dim, "peak_threshold": cfg.peak_threshold,
        "peak_min_distance_s": getattr(cfg, "peak_min_distance_s", 0.03),
        "onset_tolerance_s": cfg.onset_tolerance_s,
    }
    enc = embeddings.make_encoder(cfg.encoder, cfg.encoder_layer)
    return model, meta, enc


def _enst_tracks(root: str, split: str, val_drummer: str, exclude: str = "hits"):
    """Yield ENST eval tracks: (track_id, {stem_pitch: audio_path}, gt9_dict).

    Per-stem (enst-sep) -> the model's native input. Groups the per-pitch stems by
    take. GT is the full 9-lane annotation (folded to 5-class downstream). `exclude`
    drops takes whose name contains it -- default "hits" skips ENST's isolated
    technique demos (`NNN_hits_...`), keeping the musical phrase recordings the
    standard ADT protocol evaluates on."""
    from drumjot_training import enst

    clips = enst.perstem_index(root)
    clips = enst.perstem_for_split(clips, split, val_drummer=val_drummer)
    by_take: dict[Path, dict] = defaultdict(dict)
    for c in clips:
        by_take[c.annotation_path].setdefault("stems", {})[c.pitch] = c.audio_path
    for ann, d in by_take.items():
        if exclude and exclude in ann.stem:
            continue
        yield ann.stem, d["stems"], enst.onsets_by_lane(ann)


def _mdb_tracks(mdb_root: str, sep_root: str):
    """Yield MDB-Drums eval tracks: (track, {pitch: stem_path}, gt9). Audio = the
    pre-separated `mdb-sep` per-stem tree (separate_mdb_dataset.py); GT parsed from
    MDB's subclass annotations. MDB is PRISTINE (never in our train/val)."""
    from drumjot_training import mdb

    sep = Path(sep_root)
    for clip in mdb.index(mdb_root):
        stems = {p: sep / "perstem" / p / f"{clip.track}.flac" for p in ("k", "s", "h", "c", "t")}
        stems = {p: a for p, a in stems.items() if a.exists()}
        yield clip.track, stems, mdb.onsets_by_lane(clip.subclass_ann)


def _dataset_tracks(dataset: str, args):
    if dataset == "enst":
        root = args.enst_root or os.environ.get("DRUMJOT_ENST", "/codebox-workspace/datasets/enst-sep")
        yield from _enst_tracks(root, args.split, args.val_drummer, exclude=args.exclude_takes)
    elif dataset == "mdb":
        yield from _mdb_tracks(args.mdb_root, args.mdb_sep_root)
    elif dataset == "rbma":
        raise NotImplementedError("rbma: skipped (no free audio source).")
    else:
        raise ValueError(f"unknown dataset {dataset}")


def _predict_perstem(stems: dict, model, meta, encoder, max_seconds) -> dict[str, list[float]]:
    """Run the model on each per-instrument stem, keep only that stem's lanes
    (per-stem isolation, our deployment) -> assembled 9-lane prediction."""
    from drumjot_training import enst, inference

    pred = {ln: [] for ln in meta["lanes"]}
    for pitch, audio in stems.items():
        owned = enst.PERSTEM_TO_LANES.get(pitch, ())
        if not owned:
            continue
        onsets = inference.transcribe(audio, model, meta, encoder=encoder, max_seconds=max_seconds)
        for ln in owned:
            pred[ln] = onsets.get(ln, [])
    return pred


def main():
    ap = argparse.ArgumentParser(description="SOTA-comparable 5-class ADT eval (mir_eval +/-50ms)")
    ap.add_argument("--checkpoint", required=True, help="loss_ab_*.pt raw dict OR a checkpoint.py dir")
    ap.add_argument("--dataset", default="enst", choices=("enst", "mdb", "rbma"))
    ap.add_argument("--enst-root", default=None, help="enst-sep root (per-stem); else $DRUMJOT_ENST")
    ap.add_argument("--mdb-root", default="/codebox-workspace/datasets/MDBDrums", help="MDBDrums clone")
    ap.add_argument("--mdb-sep-root", default="/codebox-workspace/datasets/mdb-sep",
                    help="MDB per-stem tree from separate_mdb_dataset.py")
    ap.add_argument("--split", default="test", help="enst split (test=held-out drummer)")
    ap.add_argument("--val-drummer", default="drummer_3", help="ENST held-out drummer")
    ap.add_argument("--tolerance", type=float, default=0.05, help="mir_eval onset window (s); ADT std 0.05")
    ap.add_argument("--classes", default="KD,SD,HH,TT,CY",
                    help="classes to score (GT for other classes is dropped from mir_eval too); "
                         "e.g. 'HH,CY' to eval a cym+hat-only checkpoint on just those lanes")
    ap.add_argument("--exclude-takes", default="hits",
                    help="skip dataset takes whose name contains this (default 'hits' -> ENST phrases only)")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap per-track audio (debug)")
    ap.add_argument("--max-tracks", type=int, default=0, help="cap tracks (0=all; dry run)")
    ap.add_argument("--out-json", default="/codebox-workspace/sota_eval.json")
    args = ap.parse_args()

    import torch

    from drumjot_training import metrics, runtime

    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))
    runtime.configure_backends()
    log = lambda s: print(s, flush=True)  # noqa: E731
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== SOTA eval: dataset={args.dataset} tol=+/-{args.tolerance*1000:.0f}ms ckpt={args.checkpoint} ===")
    model, meta, encoder = _load_checkpoint(args.checkpoint, device)
    model.eval()
    log(f"  lanes={meta['lanes']} thresholds={ {k: round(v,2) for k,v in meta['thresholds'].items()} }")
    want = {x.strip() for x in args.classes.split(",")}
    classes = [c for c in CLASSES if c in want]
    log(f"  scoring classes: {classes}  (GT for others is excluded from mir_eval)")

    # pooled (micro) counts + per-track F lists, per class
    pooled = {c: {"tp": 0, "ref": 0, "est": 0} for c in classes}
    per_track_f: dict[str, list[float]] = {c: [] for c in classes}
    n_tracks = 0
    for tid, stems, gt9 in _dataset_tracks(args.dataset, args):
        if args.max_tracks and n_tracks >= args.max_tracks:
            break
        if len(stems) < 5:
            log(f"  warn {tid}: only {len(stems)}/5 per-instrument stems found "
                f"-> missing-lane scores will be low (incomplete separation?)")
        pred9 = _predict_perstem(stems, model, meta, encoder, args.max_seconds)
        gt5, pred5 = fold5(gt9), fold5(pred9)
        cells = []
        for c in classes:
            ref, est = gt5[c], pred5[c]
            res = metrics.onset_f1(ref, est, args.tolerance)
            tp = round(res["r"] * len(ref))  # back out TP for pooling
            pooled[c]["tp"] += tp
            pooled[c]["ref"] += len(ref)
            pooled[c]["est"] += len(est)
            if ref:  # per-track macro only over classes present in the track
                per_track_f[c].append(res["f"])
            cells.append(f"{c} {res['f']:.2f}")
        n_tracks += 1
        log(f"  [{n_tracks:3d}] {tid[:40]:40s} | " + "  ".join(cells))

    def pooled_prf(c):
        d = pooled[c]
        r = d["tp"] / d["ref"] if d["ref"] else 0.0
        p = d["tp"] / d["est"] if d["est"] else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return r, p, f

    out = {"config": vars(args), "n_tracks": n_tracks, "pooled": {}, "per_track_mean": {}}
    log(f"\n==== {len(classes)}-class onset F-measure (+/-{args.tolerance*1000:.0f}ms), {n_tracks} tracks ====")
    log(f"  {'':5s} | {'POOLED (micro)':22s} | per-track mean")
    log(f"  {'cls':5s} | {'R':>5s} {'P':>5s} {'F':>5s}        | {'F':>6s} (n)")
    fs_pooled, fs_macro = [], []
    for c in classes:
        r, p, f = pooled_prf(c)
        tm = (sum(per_track_f[c]) / len(per_track_f[c])) if per_track_f[c] else 0.0
        fs_pooled.append(f)
        fs_macro.append(tm)
        out["pooled"][c] = {"recall": r, "precision": p, "f1": f, **pooled[c]}
        out["per_track_mean"][c] = {"f1": tm, "n": len(per_track_f[c])}
        log(f"  {c:5s} | {r:5.3f} {p:5.3f} {f:5.3f}        | {tm:6.3f} ({len(per_track_f[c])})")
    macro_pooled = sum(fs_pooled) / len(fs_pooled)
    macro_mean = sum(fs_macro) / len(fs_macro)
    out["macro_pooled_f1"], out["macro_per_track_f1"] = macro_pooled, macro_mean
    log(f"  {'AVG':5s} | {'':17s}{macro_pooled:5.3f}    | {macro_mean:6.3f}")
    log("\n  Compare the POOLED 5-class AVG to published numbers (MDB ~0.85-0.89, "
        "ENST ~0.85, RBMA ~0.63).")
    log("  CAVEATS: per-stem isolation = no cross-instrument FP (optimistic vs SOTA-on-mix); "
        "ENST drummer_3 was in the val/threshold-tuning pool (mildly optimistic). MDB/RBMA "
        "are the pristine targets -- implement those adapters for the honest number.")
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
