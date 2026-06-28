"""Snip a sample of SUSPECT crash onsets for human eyes+ears review.

The label audit (RESULTS.md 2026-06-19) flagged ~30% of crash onsets as "suspect"
by a strict relative gate (peak vs the clip's loudest crash), while the canonical
support gate flagged only 1.3%. The disagreement is the whole question: are the
suspects mislabels, or real-but-soft crashes? This pulls a random sample so you
can judge by listening + looking.

For each sampled suspect it writes, into --out-dir:
  - a WAV: +/-`--before/--after` s around the onset (the model's crash STEM)
  - a PNG: spectrogram (top) + onset-strength envelope (bottom), sharing a time
    axis, with a RED dashed line at the flagged onset time -- a real crash shows a
    broadband vertical streak + an envelope spike at the line; a mislabel shows
    nothing there.
  - manifest.json / manifest.csv with source, path, time, and the gate stats.

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=8 python training/scripts/cymbal_snip_suspects.py \
      --n 100 --out-dir /codebox-workspace/crash_suspect_samples

Validate the suspect classifier without audio:  python ... --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # head_capacity_sweep
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/

SR = 44100
HOP = 512
FPS = SR / HOP  # ~86 fps
SOURCES_IN_PATH = ("star", "enst", "egmd")


def _source_of(path) -> str:
    p = str(path).lower()
    for s in SOURCES_IN_PATH:
        if s in p:
            return s
    return "other"


def suspect_stats(env, t, *, core_s=0.05, ctx_s=0.4):
    """Relative-gate stats at time `t` (s): (snr, rel). Suspect when there's no
    local rise (snr < 3) AND no global salience (rel < 0.15) -- the same metric
    that flagged ~30% in the hand-rolled audit. ref = the clip's loudest onset."""
    n = env.size
    f = int(round(t * FPS))
    w, c = max(1, int(core_s * FPS)), max(1, int(ctx_s * FPS))
    lo, hi = max(0, f - w), min(n, f + w + 1)
    if lo >= hi:
        return None
    peak = float(env[lo:hi].max())
    ctx = env[max(0, f - c):min(n, f + c + 1)]
    base = float(np.median(ctx)) if ctx.size else 0.0
    ref = float(env.max()) if n else 0.0
    return peak / (base + 1e-9), peak / (ref + 1e-9)


def is_suspect(snr, rel, *, snr_thr=3.0, rel_thr=0.15) -> bool:
    return not (snr >= snr_thr or rel >= rel_thr)


def collect_suspects(specs, log, max_scan=0):
    """One pass over crash stems -> list of suspect onsets {path,t,source,snr,rel}."""
    import librosa

    crash = [(a, o["cr"]) for (a, o, _full) in specs if o.get("cr")]
    if max_scan:
        crash = crash[:max_scan]
    log(f"scanning {len(crash)} crash stems for suspects")
    out = []
    t0 = time.perf_counter()
    for k, (audio_path, onsets) in enumerate(crash, 1):
        try:
            y, _ = librosa.load(audio_path, sr=SR, mono=True)
            env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        except Exception as e:  # noqa: BLE001
            log(f"  skip {Path(audio_path).name}: {e!r}")
            continue
        for t in onsets:
            st = suspect_stats(env, float(t))
            if st is None:
                continue
            snr, rel = st
            if is_suspect(snr, rel):
                out.append({"audio_path": str(audio_path), "t": float(t),
                            "source": _source_of(audio_path), "snr": snr, "rel": rel})
        if k % 100 == 0:
            log(f"  scanned {k}/{len(crash)} ({len(out)} suspects, {time.perf_counter() - t0:.0f}s)")
    return out


def render(y, sr, onset_in_snip, title, png_path):
    """2-panel PNG: log-freq spectrogram + onset-strength envelope, red line @ onset."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)), ref=np.max)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    times = librosa.times_like(env, sr=sr, hop_length=HOP)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    librosa.display.specshow(S, sr=sr, hop_length=HOP, x_axis="time", y_axis="log", ax=ax1)
    ax1.axvline(onset_in_snip, color="red", lw=1.6, ls="--")
    ax1.set(title=title, ylabel="freq (Hz)")
    ax2.plot(times, env, lw=0.9)
    ax2.axvline(onset_in_snip, color="red", lw=1.6, ls="--")
    ax2.set(xlabel="time (s)", ylabel="onset str")
    ax2.margins(x=0)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in name)


def snip_all(samples, out_dir, before, after, log):
    import librosa
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, s in enumerate(samples):
        t = s["t"]
        start = max(0.0, t - before)
        dur = before + after if t >= before else t + after
        try:
            y, sr = librosa.load(s["audio_path"], sr=SR, mono=True, offset=start, duration=dur)
            onset_in_snip = t - start
            stem = _safe(Path(s["audio_path"]).stem)[:40]
            base = f"{i:03d}_{s['source']}_{stem}_t{t:07.2f}"
            wav, png = out_dir / f"{base}.wav", out_dir / f"{base}.png"
            sf.write(wav, y, sr)
            title = (f"{i:03d} {s['source']}  t={t:.2f}s  snr={s['snr']:.2f} rel={s['rel']:.3f}  "
                     f"{Path(s['audio_path']).name}")
            render(y, sr, onset_in_snip, title, png)
        except Exception as e:  # noqa: BLE001
            log(f"  snip {i} failed: {e!r}")
            continue
        manifest.append({"idx": i, **s, "wav": wav.name, "png": png.name,
                         "snip_start": start, "onset_in_snip": onset_in_snip})
        if (i + 1) % 20 == 0:
            log(f"  snipped {i + 1}/{len(samples)}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        cols = ["idx", "source", "t", "snr", "rel", "onset_in_snip", "wav", "png", "audio_path"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)
    return manifest


def _selftest():
    n = 600
    env = np.full(n, 0.1)
    env[300] = 5.0
    snr, rel = suspect_stats(env, 300 / FPS)  # the loud transient
    assert not is_suspect(snr, rel), (snr, rel)
    snr, rel = suspect_stats(env, 100 / FPS)  # flat region
    assert is_suspect(snr, rel), (snr, rel)
    env[120] = 0.6  # soft bump, 12% of the max -> rel<0.15 but local rise -> NOT suspect
    snr, rel = suspect_stats(env, 120 / FPS)
    assert not is_suspect(snr, rel), (snr, rel)
    print("SELFTEST OK (suspect_stats: loud / flat-suspect / soft-rise)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Snip suspect crash onsets for review")
    ap.add_argument("--n", type=int, default=100, help="how many suspects to sample")
    ap.add_argument("--before", type=float, default=3.0)
    ap.add_argument("--after", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--cache", default="/codebox-workspace/mert_cache")
    ap.add_argument("--out-dir", default="/codebox-workspace/crash_suspect_samples")
    ap.add_argument("--max-scan-clips", type=int, default=0, help="cap crash stems scanned (dry run)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    from head_capacity_sweep import build_specs

    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    log(f"=== snip suspects: n={args.n} +/-{args.before}/{args.after}s -> {out_dir} ===")

    tr_specs, va_specs = build_specs(sources, args.pool_cap, Path(args.cache))
    specs = (tr_specs if "train" in splits else []) + (va_specs if "val" in splits else [])
    suspects = collect_suspects(specs, log, max_scan=args.max_scan_clips)
    by_src = {s: sum(1 for x in suspects if x["source"] == s) for s in SOURCES_IN_PATH}
    log(f"total suspects: {len(suspects)}  by source: {by_src}")
    if not suspects:
        log("no suspects found")
        return
    random.Random(args.seed).shuffle(suspects)
    sample = suspects[:args.n]
    samp_src = {s: sum(1 for x in sample if x["source"] == s) for s in SOURCES_IN_PATH}
    log(f"sampling {len(sample)} (seed {args.seed})  by source: {samp_src}")
    snip_all(sample, out_dir, args.before, args.after, log)
    log(f"\ndone -> {out_dir}  (manifest.json / manifest.csv + {len(sample)} wav+png pairs)")


if __name__ == "__main__":
    main()
