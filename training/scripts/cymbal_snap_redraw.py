"""Redraw suspect snippets with SNAP + FILTER applied, for visual verification.

Two corrections to the raw labels, both audio-referenced:
  - SNAP    : a label near a real audio onset -> move it onto that onset.
  - DISCARD : a label with NO real transient within the window -> drop it (a false
              onset, e.g. 000 @ 3.0s where the real crash is ~400ms away).

"Real onset" = a local max of the onset-strength envelope that passes the
validated transient test (local rise OR global salience -- the same metric that
flagged the ~30% suspects, confirmed by the user's review). Snap target = the
NEAREST real onset within +/-window (nearest, not loudest, so a good label isn't
yanked onto a louder neighbour). No real onset in window -> discard.

The decision functions (`real_onset_times`, `align_or_discard`) are imported by
the dataset-wide aligner so the two can't drift.

This re-renders the SAME 100 snippets (manifest from cymbal_snip_suspects) drawing
every crash label: original (red dashed), snapped (green solid), discarded
(magenta dotted) -- so snap+filter can be eyeballed before going dataset-wide.

  OMP_NUM_THREADS=8 python training/scripts/cymbal_snap_redraw.py \
      --in-dir /codebox-workspace/crash_suspect_samples \
      --out-dir /codebox-workspace/crash_suspect_samples_snapped

Validate the snap/filter logic without audio:  python ... --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # head_capacity_sweep
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/

SR = 44100
DISP_HOP = 512  # spectrogram/display envelope hop
SNAP_HOP = 64   # forced_align's high-res envelope hop (~1.45 ms)
# A real onset = an envelope local max whose PROMINENCE (rise above its local
# baseline) clears PROM_FRAC of the clip's loudest onset. Prominence (not absolute
# height) is what the transcriber's picker uses -- it keeps a soft-but-real hit
# while rejecting flat noise. Calibration (ride stem): background noise peaks sit
# at ~3.3% prominence (p90), so a few % puts the bar just above the noise; soft
# rides below it are near-inaudible and fine to drop.
PROM_FRAC = 0.05


def real_onset_times(env, fps, *, min_sep_s=0.03, prom_frac=PROM_FRAC):
    """Times (s) of real audio onsets: envelope local maxima >= `min_sep_s` apart
    whose prominence >= `prom_frac` * clip-max. The audio's own onset set, to align
    labels to."""
    from scipy.signal import find_peaks

    if env.size == 0:
        return np.empty(0)
    peaks, _ = find_peaks(env, distance=max(1, round(min_sep_s * fps)),
                          prominence=prom_frac * float(env.max()))
    return peaks.astype(np.float64) / fps


def align_or_discard(label_times, real_times, window_s):
    """For each label: snap to the NEAREST real onset within +/-window_s, else
    discard. Dedup: if two labels snap to the same real onset, keep one (the
    closer). Returns list of (kind, original_t, new_t) in input order; kind is
    'snap' (new_t = onset, may equal original) or 'discard' (new_t = original)."""
    real = np.sort(np.asarray(real_times, dtype=np.float64))
    out = []
    claimed: dict[int, tuple[int, float]] = {}  # real-idx -> (label-idx, dist)
    for li, t in enumerate(label_times):
        if real.size == 0:
            out.append(("discard", float(t), float(t)))
            continue
        j = int(np.argmin(np.abs(real - t)))
        d = abs(real[j] - t)
        if d <= window_s:
            prev = claimed.get(j)
            if prev is None or d < prev[1]:
                if prev is not None:
                    out[prev[0]] = ("discard", out[prev[0]][1], out[prev[0]][1])  # bump the farther one
                claimed[j] = (li, d)
                out.append(("snap", float(t), float(real[j])))
            else:
                out.append(("discard", float(t), float(t)))  # this real onset already taken, closer
        else:
            out.append(("discard", float(t), float(t)))
    return out


def render(y, sr, decisions, title, png_path):
    """2-panel PNG: red=original, green=snapped, magenta dotted=discarded."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=DISP_HOP)), ref=np.max)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=DISP_HOP)
    times = librosa.times_like(env, sr=sr, hop_length=DISP_HOP)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    librosa.display.specshow(S, sr=sr, hop_length=DISP_HOP, x_axis="time", y_axis="log", ax=ax1)
    for ax in (ax1, ax2):
        for kind, ot, nt, sub in decisions:
            ax.axvline(ot, color="red", lw=1.0, ls="--", alpha=0.45)
            if kind == "snap":
                ax.axvline(nt, color="lime", lw=1.4, ls="-", alpha=0.9)
            elif sub and sub.startswith("wrong"):
                ax.axvline(ot, color="orange", lw=1.6, ls=":", alpha=0.95)  # wrong-lane
            else:
                ax.axvline(ot, color="magenta", lw=1.6, ls=":", alpha=0.95)  # ghost
    ax1.set(title=title, ylabel="freq (Hz)")
    ax2.plot(times, env, lw=0.9, color="0.3")
    ax2.set(xlabel="time (s)  [red=orig, green=snap, magenta=ghost, orange=wrong-lane]",
            ylabel="onset str")
    ax2.margins(x=0)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def _full_by_audio(specs):
    """audio_path -> full per-lane onset dict (all lanes), for crash stems."""
    m = {}
    for audio_path, o, full in specs:
        if o.get("cr"):
            m[str(audio_path)] = full
    return m


def discard_reason(t, full, *, exclude="cr", tol=0.05):
    """Why a discarded crash label is bogus: 'wrong:<lanes>' if another lane has an
    onset within +/-tol of `t` (likely the right hit in the wrong lane), else 'ghost'."""
    hits = [ln for ln, ts in full.items()
            if ln != exclude and any(abs(float(x) - t) <= tol for x in ts)]
    return f"wrong:{'+'.join(sorted(hits))}" if hits else "ghost"


def _selftest():
    fps = SR / SNAP_HOP
    n = 6000
    env = np.full(n, 0.05)
    for pt in (1.0, 2.0, 3.0):  # three real onsets
        env[int(round(pt * fps))] = 5.0
    reals = real_onset_times(env, fps)
    assert len(reals) == 3 and np.allclose(np.round(reals), [1, 2, 3]), reals
    # label 80ms late -> snap; label at 3.5 (>window from any real) -> discard
    dec = align_or_discard([1.08, 3.5], reals, window_s=0.12)
    assert dec[0][0] == "snap" and abs(dec[0][2] - 1.0) < 0.005, dec
    assert dec[1][0] == "discard", dec
    # two labels near the SAME onset -> the farther one is discarded (dedup)
    dec = align_or_discard([2.01, 2.05], reals, window_s=0.12)
    kinds = sorted(d[0] for d in dec)
    assert kinds == ["discard", "snap"], dec
    # PROMINENCE gate: a loud onset (high prominence) is kept; a low-prominence
    # noise ripple is dropped; a soft-but-real hit ABOVE the prominence bar is kept.
    pg = np.zeros(n)
    pg[int(round(2.0 * fps))] = 5.0    # loud onset (sets clip ref) -> kept
    pg[int(round(4.0 * fps))] = 0.10   # noise ripple at 2% of ref (< PROM_FRAC) -> dropped
    pg[int(round(5.0 * fps))] = 0.40   # soft-but-real at 8% of ref (> PROM_FRAC) -> kept
    got = np.round(real_onset_times(pg, fps)).tolist()
    assert got == [2.0, 5.0], f"prominence gate: {got} (want loud+soft kept, noise dropped)"
    print("SELFTEST OK (snap/discard/dedup + prominence gate)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Redraw suspect snippets with snap + filter")
    ap.add_argument("--in-dir", default="/codebox-workspace/crash_suspect_samples")
    ap.add_argument("--out-dir", default="/codebox-workspace/crash_suspect_samples_snapped")
    ap.add_argument("--snap-window", type=float, default=0.12, help="+/- snap window (s)")
    ap.add_argument("--pool-sources", default="star,enst,egmd")
    ap.add_argument("--pool-cap", type=int, default=3000)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--cache", default="/codebox-workspace/datasets/_cache_mert_pooled")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    import librosa
    from head_capacity_sweep import build_specs

    log = lambda s: print(s, flush=True)  # noqa: E731
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((in_dir / "manifest.json").read_text())
    log(f"=== snap+filter redraw: {len(manifest)} snippets, window +/-{args.snap_window}s ===")

    sources = [s.strip() for s in args.pool_sources.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    tr, va = build_specs(sources, args.pool_cap, Path(args.cache))
    specs = (tr if "train" in splits else []) + (va if "val" in splits else [])
    full_of = _full_by_audio(specs)

    n_snap = n_ghost = n_wrong = n_tot = 0
    wrong_lanes: dict[str, int] = {}
    by_src = {}
    for e in manifest:
        ap_str = e["audio_path"]
        snip_start = float(e["snip_start"])
        full = full_of.get(ap_str)
        if full is None:
            continue
        all_cr = np.sort(np.asarray(full.get("cr", []), dtype=np.float64))
        try:
            y6, sr = librosa.load(ap_str, sr=SR, mono=True, offset=snip_start, duration=6.0)
        except Exception as exc:  # noqa: BLE001
            log(f"  load fail {Path(ap_str).name}: {exc!r}")
            continue
        end = snip_start + len(y6) / sr
        rel = [float(t - snip_start) for t in all_cr[(all_cr >= snip_start) & (all_cr <= end)]]
        env = librosa.onset.onset_strength(y=y6, sr=sr, hop_length=SNAP_HOP)
        reals = real_onset_times(env, sr / SNAP_HOP)
        decisions = []  # (kind, ot, nt, sub)
        for kind, ot, nt in align_or_discard(rel, reals, args.snap_window):
            sub = None
            if kind == "discard":
                sub = discard_reason(snip_start + ot, full)  # abs time for lane coincidence
                if sub.startswith("wrong"):
                    n_wrong += 1
                    wrong_lanes[sub] = wrong_lanes.get(sub, 0) + 1
                else:
                    n_ghost += 1
            else:
                n_snap += 1
            decisions.append((kind, ot, nt, sub))
        n_tot += len(decisions)
        ns = sum(1 for d in decisions if d[0] == "snap")
        nd = len(decisions) - ns
        s = by_src.setdefault(e["source"], [0, 0])
        s[0] += ns
        s[1] += nd
        title = (f"{e['idx']:03d} {e['source']}  {len(rel)} labels: {ns} snap / {nd} DISCARD  "
                 f"{Path(ap_str).name}")
        png = out_dir / f"{Path(e['png']).stem}_snapfilt.png"
        try:
            render(y6, sr, decisions, title, png)
        except Exception as exc:  # noqa: BLE001
            log(f"  render fail {e['idx']}: {exc!r}")
        if (e["idx"] + 1) % 20 == 0:
            log(f"  redrawn {e['idx'] + 1}")

    log(f"\n==== snap/discard over {n_tot} crash labels in the 100 windows ====")
    for src in sorted(by_src):
        ns, nd = by_src[src]
        tot = ns + nd
        if tot:
            log(f"  {src:5s}  {ns:4d} snap / {nd:4d} discard  ({nd / tot:.1%})")
    log(f"  ALL    {n_snap} snap / {n_ghost + n_wrong} discard "
        f"({(n_ghost + n_wrong) / n_tot:.1%}); of discards: {n_ghost} ghost / {n_wrong} wrong-lane")
    if wrong_lanes:
        log("  wrong-lane breakdown (discarded crash coincides with another lane):")
        for k, v in sorted(wrong_lanes.items(), key=lambda kv: -kv[1]):
            log(f"    {k}: {v}")
    log(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
