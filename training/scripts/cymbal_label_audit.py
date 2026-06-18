"""Crash label-quality audit: is there an actual transient at each labelled onset?

The miss-typing (RESULTS.md 2026-06-18) found the head leaves 74.6% of crash
onsets `dead` (activation ~0). Before chasing that with a better loss, check the
premise: maybe a chunk of those labels are simply WRONG -- the annotation says
"crash at t" but the audio the model is trained on has no transient there. A model
CAN'T (and shouldn't) fire on absent signal, so mislabels would masquerade as
under-firing and a louder loss would only teach it to hallucinate.

This is head-free and GPU-free: for every crash onset in the pool it loads the
crash STEM (what the model is actually trained on) and measures the onset
strength at the labelled time vs the local/global baseline. An onset with no
local rise AND negligible global salience = no transient = a suspect label.

  - full-spectrum onset strength (librosa) -- "is there ANY transient here?"
  - 6-20 kHz flux -- "is it crash-LIKE?" (the band the model's HB block sees)

Per-SOURCE split is load-bearing for attribution:
  - egmd : synthetic, rendered FROM the MIDI labels -> a label can't be wrong by
           construction; a no-transient here means a render/separation drop.
  - star : programmatic -> labels aligned; no-transient ~ separation drop.
  - enst : REAL recordings, MANUAL annotation -> no-transient may be a true label
           error (or a separation drop; isolating the two needs the original mix).

Validate the transient classifier without audio/GPU:  python ... --selftest

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=4 python training/scripts/cymbal_label_audit.py \
      --splits train,val --out-json /codebox-workspace/cymbal_label_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # head_capacity_sweep
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/

FPS = 75.0  # envelope frame rate (hop = SR/FPS), matches the model's frame grid
SR = 44100
HOP = int(round(SR / FPS))  # 588 @ 75 fps (exact)
SOURCES_IN_PATH = ("star", "enst", "egmd")


def classify_onset(env, hf, t, *, core_s=0.05, ctx_s=0.4, snr_thr=3.0, rel_thr=0.15):
    """Is there a transient at time `t` (s) in onset-strength `env`?

    Combines a LOCAL rise (peak in +/-`core_s` over the median of +/-`ctx_s`) with
    a GLOBAL salience (peak vs the clip's 99th pct), so a soft-but-real onset on a
    sparse stem still passes and only a genuinely flat spot is flagged. Returns a
    dict (or {'edge': True} when the window falls off the clip)."""
    n = env.size
    f = int(round(t * FPS))
    w, c = max(1, int(core_s * FPS)), max(1, int(ctx_s * FPS))
    lo, hi = f - w, f + w + 1
    if hi <= 0 or lo >= n:
        return {"edge": True}
    core = env[max(0, lo):min(n, hi)]
    if core.size == 0:
        return {"edge": True}
    peak = float(core.max())
    ctx = env[max(0, f - c):min(n, f + c + 1)]
    base = float(np.median(ctx)) if ctx.size else 0.0
    ref = float(env.max()) if n else 0.0  # loudest onset in the clip (a real crash)
    snr = peak / (base + 1e-9)
    rel = peak / (ref + 1e-9)
    hpeak = float(hf[max(0, lo):min(hf.size, hi)].max()) if hf.size else 0.0
    href = float(hf.max()) if hf.size else 0.0
    has = (snr >= snr_thr) or (rel >= rel_thr)
    return {"edge": False, "snr": snr, "rel": rel, "has_transient": has,
            "hf_rel": hpeak / (href + 1e-9)}


def _source_of(path) -> str:
    p = str(path).lower()
    for s in SOURCES_IN_PATH:
        if s in p:
            return s
    return "other"


def _envelopes(audio_path: str):
    """(full-spectrum onset strength, 6-20 kHz flux) at FPS for one stem."""
    import librosa

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    if y.size == 0:
        return np.zeros(1), np.zeros(1)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = (freqs >= 6000) & (freqs <= 20000)
    hb = S[band]
    hf = np.concatenate([[0.0], np.sum(np.maximum(0.0, np.diff(hb, axis=1)), axis=0)]) \
        if hb.shape[0] else np.zeros_like(env)
    return env, hf


def audit(specs, log, max_clips=0):
    """Per-source stats over every crash onset in `specs` (crash stems only)."""
    crash = [(a, o["cr"]) for (a, o, _full) in specs if o.get("cr")]
    if max_clips:
        crash = crash[:max_clips]
    log(f"crash stems to audit: {len(crash)}")
    agg = {}  # source -> dict of counters + lists
    t0 = time.perf_counter()
    for k, (audio_path, onsets) in enumerate(crash, 1):
        src = _source_of(audio_path)
        a = agg.setdefault(src, {"n": 0, "edge": 0, "no_transient": 0, "snr": [], "rel": [], "hf_rel": []})
        try:
            env, hf = _envelopes(audio_path)
        except Exception as e:  # noqa: BLE001
            log(f"  skip {Path(audio_path).name}: {e!r}")
            continue
        for t in onsets:
            r = classify_onset(env, hf, float(t))
            if r["edge"]:
                a["edge"] += 1
                continue
            a["n"] += 1
            a["snr"].append(r["snr"])
            a["rel"].append(r["rel"])
            a["hf_rel"].append(r["hf_rel"])
            if not r["has_transient"]:
                a["no_transient"] += 1
        if k % 100 == 0:
            log(f"  audited {k}/{len(crash)} stems ({time.perf_counter() - t0:.0f}s)")
    return agg


def _report(agg, log):
    out = {}
    log("\n==== crash label audit (suspect = no transient at the labelled time) ====")
    log(f"  {'source':7s} {'onsets':>7s} {'no-trans':>9s} {'%suspect':>9s} "
        f"{'medSNR':>7s} {'medRel':>7s} {'medHFrel':>9s} {'edge':>6s}")
    tot = {"n": 0, "no_transient": 0, "edge": 0}
    for src in sorted(agg):
        a = agg[src]
        n = a["n"]
        if n == 0:
            continue
        susp = a["no_transient"] / n
        med = lambda xs: float(np.median(xs)) if xs else 0.0  # noqa: E731
        log(f"  {src:7s} {n:7d} {a['no_transient']:9d} {susp:9.1%} "
            f"{med(a['snr']):7.2f} {med(a['rel']):7.3f} {med(a['hf_rel']):9.3f} {a['edge']:6d}")
        out[src] = {"onsets": n, "no_transient": a["no_transient"], "pct_suspect": susp,
                    "median_snr": med(a["snr"]), "median_rel": med(a["rel"]),
                    "median_hf_rel": med(a["hf_rel"]), "edge": a["edge"]}
        tot["n"] += n
        tot["no_transient"] += a["no_transient"]
        tot["edge"] += a["edge"]
    if tot["n"]:
        log(f"  {'ALL':7s} {tot['n']:7d} {tot['no_transient']:9d} "
            f"{tot['no_transient'] / tot['n']:9.1%} {'':7s} {'':7s} {'':9s} {tot['edge']:6d}")
        out["ALL"] = {"onsets": tot["n"], "no_transient": tot["no_transient"],
                      "pct_suspect": tot["no_transient"] / tot["n"], "edge": tot["edge"]}
    log("\n  Sanity: most crash labels SHOULD have transients -- a high %suspect on a")
    log("  SYNTHETIC source (egmd/star) means render/separation drop; on enst it may")
    log("  be a genuine label error. If %suspect is high everywhere, the audio metric")
    log("  is suspect, not the labels.")
    return out


def _selftest():
    n = 600
    env = np.full(n, 0.2)  # flat baseline
    env[300] = 5.0  # a clear transient at frame 300 (t=4.0s)
    hf = np.zeros(n)
    hf[300] = 3.0
    r = classify_onset(env, hf, 300 / FPS)
    assert r["has_transient"], r
    r = classify_onset(env, hf, 100 / FPS)  # flat region -> no transient (suspect label)
    assert not r["has_transient"], r
    r = classify_onset(env, hf, 5000 / FPS)  # off the end -> edge
    assert r["edge"], r
    # a soft-but-real onset on a SPARSE stem (median base ~0): local rise still passes
    sparse = np.zeros(n)
    sparse[300] = 0.6
    r = classify_onset(sparse, np.zeros(n), 300 / FPS)
    assert r["has_transient"], r
    print("SELFTEST OK (classify_onset: transient / flat-suspect / edge / sparse-soft)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Crash label-quality audit (transient at the label?)")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000, help="match the training pool")
    ap.add_argument("--splits", default="train,val", help="train, val, or train,val")
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled",
                    help="only for the onset memo (_onsets.json); no features read")
    ap.add_argument("--max-clips", type=int, default=0, help="cap crash stems (0 = all); for a dry run")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_label_audit.json")
    ap.add_argument("--selftest", action="store_true", help="validate the classifier; no audio/GPU")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    from head_capacity_sweep import build_specs

    from drumjot_training import runtime
    runtime.tee_stdio(Path(args.out_json).with_suffix(".log"))
    log = lambda s: print(s, flush=True)  # noqa: E731
    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cache = Path(args.cache)
    log(f"=== crash label audit: sources={sources} splits={splits} cap={args.pool_cap} ===")

    tr_specs, va_specs = build_specs(sources, args.pool_cap, cache)
    specs = []
    if "train" in splits:
        specs += tr_specs
    if "val" in splits:
        specs += va_specs
    agg = audit(specs, log, max_clips=args.max_clips)
    out = _report(agg, log)
    Path(args.out_json).write_text(json.dumps({"config": vars(args), "audit": out}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
