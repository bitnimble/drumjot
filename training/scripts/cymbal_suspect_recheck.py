"""Re-run the ORIGINAL hand-rolled suspect audit over the SNAPPED+FILTERED onsets.

The first crash audit (commit b8ffe8f) flagged ~30% of crash labels as "suspect"
(no transient at the labelled time: rel<0.15 of clip-max AND local-SNR<3). The
snap+filter pipeline (`align_dataset_onsets.py`) then DISCARDED only 9.1% crash /
13.7% ride. Those are different criteria -- snapping RELOCATES a mistimed label
onto the nearest real onset instead of discarding it, so a label that was suspect
(off the transient) becomes "snapped," not "discarded." This script answers the
obvious follow-up: after snap+filter, how many SURVIVING labels still fail the
original strict suspect test? If snapping genuinely fixed them, %suspect collapses;
if it merely parked labels on bleed/noise peaks, %suspect stays ~30%.

It computes each stem's onset-strength envelope ONCE and scores BOTH the raw and
the aligned onset lists against it, so the `raw` column reproduces the ~30%
baseline (a sanity anchor) and `aligned` is the apples-to-apples residual. Crash
and ride, per source.

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=8 python training/scripts/cymbal_suspect_recheck.py \
      --aligned-onsets /codebox-workspace/datasets/_onsets_aligned.json \
      --out-json /codebox-workspace/cymbal_suspect_recheck.json

Validate the transient classifier without audio/GPU:  python ... --selftest
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
LANES = ("cr", "rd")


def classify_onset(env, hf, t, *, core_s=0.05, ctx_s=0.4, snr_thr=3.0, rel_thr=0.15):
    """Is there a transient at time `t` (s) in onset-strength `env`?  VERBATIM from
    the original audit (commit b8ffe8f) so the suspect criterion is identical."""
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


SIBLING = {"cr": "rd", "rd": "cr"}  # within-cymbal-stem confusion partner


def _score(onsets, env, hf, counters, *, sibling=None, coincide_s=0.05):
    """Tally suspect/edge for one onset list against a precomputed envelope. When
    `sibling` (the other cymbal lane's onset times) is given, each SUSPECT onset is
    further bucketed: `wrong` (a sibling onset within +/-coincide_s -> likely a
    rd<->cr mislabel), `dead` (snr<1.5 -> no transient = separation drop), else
    `soft` (a weak-but-real rise we deliberately keep)."""
    sib = np.asarray(sibling, dtype=np.float64) if sibling is not None else None
    for t in onsets:
        r = classify_onset(env, hf, float(t))
        if r["edge"]:
            counters["edge"] += 1
            continue
        counters["n"] += 1
        if r["has_transient"]:
            continue
        counters["suspect"] += 1
        if sib is None:
            continue
        if sib.size and float(np.min(np.abs(sib - float(t)))) <= coincide_s:
            counters["wrong"] += 1
        elif r["snr"] < 1.5:
            counters["dead"] += 1
        else:
            counters["soft"] += 1


def audit_pair(raw_specs, aligned_by_path, log, *, max_clips=0):
    """For every stem with a crash/ride lane, compute the envelope ONCE and score
    both the raw and the aligned onset lists. agg[(source, lane)] -> {raw, aligned}."""
    stems = [(a, o) for (a, o, _full) in raw_specs if any(o.get(ln) for ln in LANES)]
    if max_clips:
        stems = stems[:max_clips]
    log(f"stems to audit (have crash or ride): {len(stems)}")
    agg = {}
    t0 = time.perf_counter()
    for k, (audio_path, raw_on) in enumerate(stems, 1):
        src = _source_of(audio_path)
        try:
            env, hf = _envelopes(audio_path)
        except Exception as e:  # noqa: BLE001
            log(f"  skip {Path(audio_path).name}: {e!r}")
            continue
        aligned_on = aligned_by_path.get(audio_path, {})
        for ln in LANES:
            if not (raw_on.get(ln) or aligned_on.get(ln)):
                continue
            cell = agg.setdefault((src, ln), {
                "raw": {"n": 0, "suspect": 0, "edge": 0},
                "aligned": {"n": 0, "suspect": 0, "edge": 0,
                            "wrong": 0, "dead": 0, "soft": 0}})
            _score(raw_on.get(ln, []), env, hf, cell["raw"])
            _score(aligned_on.get(ln, []), env, hf, cell["aligned"],
                   sibling=aligned_on.get(SIBLING.get(ln, ""), []))
        if k % 100 == 0:
            log(f"  audited {k}/{len(stems)} stems ({time.perf_counter() - t0:.0f}s)")
    return agg


def _report(agg, log):
    out = {}
    log("\n==== suspect-label recheck: RAW vs SNAPPED+FILTERED (rel>=0.15 OR snr>=3) ====")
    log(f"  {'src/lane':10s} | {'raw n':>7s} {'raw susp':>9s} | "
        f"{'algn n':>7s} {'algn susp':>10s} | {'Δsusp':>7s}")
    tot = {}
    for (src, ln) in sorted(agg):
        c = agg[(src, ln)]
        rn, an = c["raw"]["n"], c["aligned"]["n"]
        rs = c["raw"]["suspect"] / rn if rn else 0.0
        as_ = c["aligned"]["suspect"] / an if an else 0.0
        log(f"  {src + '/' + ln:10s} | {rn:7d} {rs:9.1%} | {an:7d} {as_:10.1%} | {as_ - rs:+7.1%}")
        out[f"{src}/{ln}"] = {"raw": {"n": rn, "pct_suspect": rs},
                              "aligned": {"n": an, "pct_suspect": as_,
                                          "wrong": c["aligned"]["wrong"], "dead": c["aligned"]["dead"],
                                          "soft": c["aligned"]["soft"]}}
        for ln_key in (ln, "ALL"):
            t = tot.setdefault(ln_key, {"rn": 0, "rs": 0, "an": 0, "as": 0,
                                        "wrong": 0, "dead": 0, "soft": 0})
            t["rn"] += rn
            t["rs"] += c["raw"]["suspect"]
            t["an"] += an
            t["as"] += c["aligned"]["suspect"]
            for b in ("wrong", "dead", "soft"):
                t[b] += c["aligned"][b]
    log("  " + "-" * 60)
    for ln_key in sorted(tot):
        t = tot[ln_key]
        rs = t["rs"] / t["rn"] if t["rn"] else 0.0
        as_ = t["as"] / t["an"] if t["an"] else 0.0
        log(f"  {'ALL/' + ln_key:10s} | {t['rn']:7d} {rs:9.1%} | {t['an']:7d} {as_:10.1%} | {as_ - rs:+7.1%}")
        out[f"ALL/{ln_key}"] = {"raw": {"n": t["rn"], "pct_suspect": rs},
                                "aligned": {"n": t["an"], "pct_suspect": as_,
                                            "wrong": t["wrong"], "dead": t["dead"], "soft": t["soft"]}}
    # decomposition of the SURVIVING (aligned) suspect bucket
    log("\n==== aligned-suspect breakdown: wrong-lane (rd<->cr) / dead / soft ====")
    log(f"  {'src/lane':10s} | {'suspect':>7s} | {'wrong':>6s} {'dead':>6s} {'soft':>6s} "
        f"| {'wrong%':>7s} {'dead%':>6s} {'soft%':>6s}  (% of ALL aligned onsets)")
    for (src, ln) in sorted(agg):
        c = agg[(src, ln)]["aligned"]
        an = c["n"]
        if not an:
            continue
        susp = c["suspect"]
        log(f"  {src + '/' + ln:10s} | {susp:7d} | {c['wrong']:6d} {c['dead']:6d} {c['soft']:6d} "
            f"| {c['wrong'] / an:7.1%} {c['dead'] / an:6.1%} {c['soft'] / an:6.1%}")
    log("  " + "-" * 70)
    for ln_key in sorted(k for k in tot):
        t = tot[ln_key]
        an = t["an"]
        if not an:
            continue
        log(f"  {'ALL/' + ln_key:10s} | {t['as']:7d} | {t['wrong']:6d} {t['dead']:6d} {t['soft']:6d} "
            f"| {t['wrong'] / an:7.1%} {t['dead'] / an:6.1%} {t['soft'] / an:6.1%}")
    log("\n  raw susp reproduces the ~30% baseline; algn susp = surviving suspects after")
    log("  snap+filter. Of those: WRONG = a sibling cymbal onset within 50 ms (likely a")
    log("  rd<->cr mislabel, genuinely bad); DEAD = no transient (snr<1.5, separation")
    log("  drop -- a target with no learnable signal); SOFT = weak-but-real rise (keep).")
    log("  NB cross-KIT bleed (vs kick/snare/hat) isn't visible -- per-stem files carry")
    log("  only the cymbal lanes, so WRONG captures rd<->cr confusion only.")
    return out


def _selftest():
    n = 600
    env = np.full(n, 0.2)
    env[300] = 5.0
    hf = np.zeros(n)
    hf[300] = 3.0
    assert classify_onset(env, hf, 300 / FPS)["has_transient"]
    assert not classify_onset(env, hf, 100 / FPS)["has_transient"]
    assert classify_onset(env, hf, 5000 / FPS)["edge"]
    # _score tallies suspect for a flat-region onset, not for the transient
    c = {"n": 0, "suspect": 0, "edge": 0}
    _score([300 / FPS, 100 / FPS], env, hf, c)
    assert c == {"n": 2, "suspect": 1, "edge": 0}, c
    # breakdown: a flat-region suspect with a sibling onset 20 ms away -> wrong;
    # a flat-region suspect with no sibling and snr~1 -> dead.
    cb = {"n": 0, "suspect": 0, "edge": 0, "wrong": 0, "dead": 0, "soft": 0}
    _score([100 / FPS], env, hf, cb, sibling=[100 / FPS + 0.02])
    assert cb["wrong"] == 1 and cb["dead"] == 0 and cb["soft"] == 0, cb
    cb = {"n": 0, "suspect": 0, "edge": 0, "wrong": 0, "dead": 0, "soft": 0}
    _score([100 / FPS], env, hf, cb, sibling=[])  # flat region, snr~1 -> dead
    assert cb["dead"] == 1 and cb["wrong"] == 0 and cb["soft"] == 0, cb
    print("SELFTEST OK (classify_onset + _score + breakdown)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Suspect-label recheck: raw vs snapped+filtered")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000, help="match the training pool")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--cache", default="/codebox-workspace/mert_cache")
    ap.add_argument("--aligned-onsets", default="/codebox-workspace/datasets/_onsets_aligned.json",
                    help="the snapped+filtered onsets to recheck against the raw labels")
    ap.add_argument("--max-clips", type=int, default=0, help="cap stems (0 = all); for a dry run")
    ap.add_argument("--out-json", default="/codebox-workspace/cymbal_suspect_recheck.json")
    ap.add_argument("--selftest", action="store_true")
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
    log(f"=== suspect recheck: sources={sources} splits={splits} cap={args.pool_cap} "
        f"aligned={args.aligned_onsets} ===")

    raw_tr, raw_va = build_specs(sources, args.pool_cap, cache)
    al_tr, al_va = build_specs(sources, args.pool_cap, cache, aligned_path=args.aligned_onsets)
    raw_specs, al_specs = [], []
    if "train" in splits:
        raw_specs += raw_tr
        al_specs += al_tr
    if "val" in splits:
        raw_specs += raw_va
        al_specs += al_va
    aligned_by_path = {a: o for (a, o, _full) in al_specs}
    agg = audit_pair(raw_specs, aligned_by_path, log, max_clips=args.max_clips)
    out = _report(agg, log)
    Path(args.out_json).write_text(json.dumps({"config": vars(args), "recheck": out}, indent=2))
    log(f"\nresults -> {args.out_json}")


if __name__ == "__main__":
    main()
