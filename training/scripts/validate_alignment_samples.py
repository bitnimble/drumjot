"""Sample snap+filter decisions across LANES for morning visual review.

The crash case was eyeballed (cymbal_snap_redraw); this gives the same red/green/
magenta-orange snippets for the OTHER lanes (esp. ride/tom, which discard ~30%),
so the dataset-wide alignment can be sanity-checked per lane before training on it.

Per lane: scan that lane's stems, prefer ones with at least one DISCARD (so the
filter is visible), snip +/-`--window`s around a discarded (else random) onset, and
render original(red)/snapped(green)/ghost(magenta)/wrong-lane(orange).

Reuses the shared decision fns (cymbal_snap_redraw) + the source iterator
(align_dataset_onsets), so it can't drift from the real aligner.

  DRUMJOT_STAR=... DRUMJOT_ENST=... DRUMJOT_EGMD=... OMP_NUM_THREADS=8 \
  python training/scripts/validate_alignment_samples.py \
      --lanes rd,t,hc,ho,k,s --per-lane 6 --out-dir /codebox-workspace/align_validate
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))

SR = 44100
SNAP_HOP = 64


def _lane_window(lane):
    from align_dataset_onsets import DEFAULT_WINDOW, SNAP_WINDOW
    return SNAP_WINDOW.get(lane, DEFAULT_WINDOW)


def main():
    ap = argparse.ArgumentParser(description="Per-lane snap+filter validation snippets")
    ap.add_argument("--lanes", default="rd,t,hc,ho,k,s")
    ap.add_argument("--sources", default="star,enst,egmd")
    ap.add_argument("--per-lane", type=int, default=6)
    ap.add_argument("--max-scan", type=int, default=120, help="stems scanned per lane to find samples")
    ap.add_argument("--window", type=float, default=3.0, help="+/- snippet seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/codebox-workspace/align_validate")
    args = ap.parse_args()

    import librosa
    from align_dataset_onsets import iter_source
    from cymbal_snap_redraw import align_or_discard, discard_reason, real_onset_times, render

    log = lambda s: print(s, flush=True)  # noqa: E731
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rng = random.Random(args.seed)

    # gather candidate stems per lane (audio_path, full) across sources
    stems_for = {ln: [] for ln in lanes}
    for name in sources:
        for audio_path, _pitch, slanes, full in iter_source(name, log):
            for ln in lanes:
                if ln in slanes and full.get(ln):
                    stems_for[ln].append((audio_path, full))

    for ln in lanes:
        cand = stems_for[ln]
        rng.shuffle(cand)
        cand = cand[:args.max_scan]
        win = _lane_window(ln)
        made = 0
        log(f"\n[{ln}] window +/-{win}s, {len(cand)} candidate stems")
        for audio_path, full in cand:
            if made >= args.per_lane:
                break
            try:
                y, sr = librosa.load(audio_path, sr=SR, mono=True)
                env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=SNAP_HOP)
                reals = real_onset_times(env, sr / SNAP_HOP)
                onsets = sorted(float(x) for x in full.get(ln, []))
                dec = align_or_discard(onsets, reals, win)
            except Exception as exc:  # noqa: BLE001
                log(f"  skip {Path(audio_path).name}: {exc!r}")
                continue
            discards = [ot for k, ot, _nt in dec if k == "discard"]
            if not discards and not onsets:
                continue
            center = discards[0] if discards else onsets[len(onsets) // 2]
            start = max(0.0, center - args.window)
            dur = 2 * args.window
            y2 = y[int(start * sr):int((start + dur) * sr)]
            if y2.size == 0:
                continue
            wlo, whi = start, start + len(y2) / sr
            in_win = [(k, ot, nt) for k, ot, nt in dec if wlo <= ot <= whi]
            draw = []
            for k, ot, nt in in_win:
                sub = discard_reason(ot, full, exclude=ln) if k == "discard" else None
                draw.append((k, ot - start, nt - start, sub))
            ns = sum(1 for d in draw if d[0] == "snap")
            nd = len(draw) - ns
            title = f"{ln}  {ns} snap / {nd} discard  {Path(audio_path).name}"
            png = out_dir / f"{ln}_{made:02d}_{Path(audio_path).stem[:36]}.png"
            try:
                render(y2, sr, draw, title, png)
                made += 1
            except Exception as exc:  # noqa: BLE001
                log(f"  render fail {Path(audio_path).name}: {exc!r}")
        log(f"[{ln}] wrote {made} samples")
    log(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
