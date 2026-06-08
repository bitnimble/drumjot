"""Evaluate the learned onset model on a folder of ParaDB / Paradiddle maps.

Test set = hand-charted ground truth on real songs. Two phases over the folder
(separate-all, then score-all, so the separator and MERT never share VRAM):
  1. parse the hardest `.rlrr` -> per-lane GT onsets + referenced audio tracks
  2. reconstruct the full original song from the tracks: add the drum track only
     if the song tracks are drumless backing (drum/song correlation), so a
     full-mix map's drums aren't double-counted (`build_mix`)
  3. run OUR drum separation (BS-Roformer) on it -> drum stem. We deliberately
     IGNORE the mapper's own drum split (it may be low-quality / inconsistent).
  4. ALIGNMENT/QUALITY CHECK: fraction of GT onsets sitting on a real drum-stem
     transient + the best global time offset, to flag corrupted / mis-synced
     charts before trusting them.
  5. run the model on the drum stem (windowed for full songs) -> per-lane onsets
  6. onset-F1 vs GT with optimistic, per-map hat/cymbal folding
     (rlrr.comparison_pairs); raw model vs +envelope post-filter.

Must run where BOTH the transcriber app (for separation + the audio-separator
models) and drumjot_training are importable, with a GPU, e.g. the sandbox. Point
MODELS_DIR at the provisioned models-cache so separation skips the weight
download. Stems are cached under --stems-cache so re-runs skip separation.

Usage:
  MODELS_DIR=/codebox-workspace/drumjot/models-cache \
  python3 eval_paradb.py --maps-dir <folder-of-zips> --checkpoint <dir> \
      [--window-seconds 30] [--stems-cache <dir>]
"""
import argparse
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transcriber"))

from drumjot_training import (  # noqa: E402
    clean,
    embeddings,
    forced_align,
    inference,
    metrics,
    postfilter,
    rlrr,
)

_AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".flac", ".m4a", ".aac"}
SEP_SR = 44100  # separate at full band so cymbal/hi-hat content survives

# stems_per pitch -> the model lanes that legitimately belong to that isolated
# stem. Onsets the model fires in any OTHER lane when fed this stem are
# cross-instrument leakage (hallucination): discarded + counted. (MDX23C merges
# ride+crash+china into one "cymbals" stem -> c; misc-perc has no stem.)
STEM_TO_LANES = {
    "k": ("k",),
    "s": ("s", "ss"),
    "h": ("hc", "hp", "ho"),
    "c": ("rd", "cr", "mc"),
    "t": ("t",),
}


def _pick_rlrr(root: Path) -> Path | None:
    """Hardest chart in an extracted map dir (highest complexity)."""
    charts = list(root.rglob("*.rlrr"))
    if not charts:
        return None
    return max(charts, key=lambda p: rlrr.complexity(p))


def _sum_tracks(root: Path, names: list[str], sr: int):
    """Load + sum referenced tracks (mono, resampled). Returns waveform or None."""
    import librosa

    ys = []
    for name in names:
        base = Path(name).name
        hits = [p for p in root.rglob(base) if p.suffix.lower() in _AUDIO_EXTS]
        if not hits:
            print(f"    WARN track not found in zip: {name}", flush=True)
            continue
        y, _ = librosa.load(str(hits[0]), sr=sr, mono=True)
        ys.append(y.astype(np.float32))
    if not ys:
        return None
    n = max(len(y) for y in ys)
    out = np.zeros(n, dtype=np.float32)
    for y in ys:
        out[: len(y)] += y
    return out


def _pad_sum(a, b):
    n = max(len(a), len(b))
    out = np.zeros(n, dtype=np.float32)
    out[: len(a)] += a
    out[: len(b)] += b
    return out


def _containment(song, drums, sr, max_seconds):
    """Raw-sample correlation of the drum track with the song mix.

    ~0 when the song is drumless backing (those drums are NOT in it); clearly
    positive when the song already contains those drums (a full mix = backing +
    drums, so corr ~= drum energy fraction). Unlike an onset-support test this
    isn't fooled by non-drum instruments hitting on the same beats.
    """
    n = min(len(song), len(drums))
    if max_seconds is not None:
        n = min(n, int(max_seconds * sr))
    a = song[:n].astype(np.float64) - float(np.mean(song[:n]))
    b = drums[:n].astype(np.float64) - float(np.mean(drums[:n]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return abs(float(a @ b) / denom)


def build_mix(root, song_names, drum_names, sr, out_wav, max_seconds, corr_thresh):
    """Reconstruct the full original song without double-counting drums:

    - song tracks ONLY if they already contain the drums (full-mix map, even one
      that also ships redundant drum stems -> don't add them);
    - song + drum tracks if the song tracks are drumless backing (stems map).

    Decided by drum/song signal correlation (see `_containment`), which is
    robust to the coincident-onset problem that breaks an onset-support test.
    Returns (ok, case_label).
    """
    import soundfile as sf

    song = _sum_tracks(root, song_names, sr) if song_names else None
    drums = _sum_tracks(root, drum_names, sr) if drum_names else None

    if song is not None and drums is not None:
        corr = _containment(song, drums, sr, max_seconds)
        if corr > corr_thresh:  # song already contains these drums (full mix)
            mix, case = song, f"song-only; drums already in song (corr {corr:.2f})"
        else:  # drumless backing -> add the drum track to rebuild the full song
            mix, case = _pad_sum(song, drums), f"backing+drums (corr {corr:.2f})"
    elif song is not None:
        mix, case = song, "song-only"
    elif drums is not None:
        mix, case = drums, "drums-only"
    else:
        return False, "no audio"

    peak = float(np.max(np.abs(mix)) or 1.0)
    sf.write(str(out_wav), mix / peak * 0.98, sr)  # float wav, headroom
    return True, case


def _global_offset(gt, env, env_fps, floor, window_s, search_s):
    """Robust global GT->audio offset + chart-quality support.

    offset = MEDIAN signed distance from each GT onset to its nearest envelope
    peak (>= floor) within +/-search_s. The median is robust to dense drum peaks
    and straggler onsets, unlike argmax-of-support (which, on a near-saturated
    support plateau, overshoots the true offset by chasing a few outliers).
    support = fraction of onsets within +/-window_s of a qualifying peak at
    offset 0, used purely as a chart-accuracy / corruption signal.
    """
    half = round(search_s * env_fps)
    n = env.size
    deltas = []
    for ts in gt.values():
        for t in ts:
            c = int(round(t * env_fps))
            lo, hi = max(0, c - half), min(n, c + half + 1)
            if lo >= hi:
                continue
            idx = lo + int(np.argmax(env[lo:hi]))
            if float(env[idx]) >= floor:
                deltas.append(idx / env_fps - t)
    off = float(np.median(deltas)) if deltas else 0.0
    s0 = clean.support_score(gt, env, env_fps, window_s=window_s, support_floor=floor)["fraction"]
    return off, s0


def main():
    ap = argparse.ArgumentParser(description="Evaluate learned model on ParaDB maps")
    ap.add_argument("--maps-dir", required=True, help="folder of .zip ParaDB maps")
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir (model.pt + meta.json)")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap analysis to first N s (default: whole song)")
    ap.add_argument("--window-seconds", type=float, default=30.0, help="MERT encode chunk size for long songs")
    ap.add_argument("--min-support", type=float, default=0.8, help="flag maps below this GT support")
    ap.add_argument("--support-percentile", type=float, default=60.0)
    ap.add_argument("--align-window", type=float, default=0.03, help="support/post-filter window (s)")
    ap.add_argument("--offset-window", type=float, default=0.05, help="+/- median-offset search (s)")
    ap.add_argument("--offset-correct-min", type=float, default=0.025,
                    help="apply offset correction only if |median offset| exceeds this (s)")
    ap.add_argument("--stems-cache", default=None, help="dir to cache separated drum stems")
    ap.add_argument("--drum-corr-threshold", type=float, default=0.5,
                    help="drum/song correlation above which the song is treated as already containing the drums "
                    "(biased high: when unsure we sum the drum track, which re-separation cleans up anyway)")
    ap.add_argument("--no-offset-correct", dest="offset_correct", action="store_false",
                    help="score against the raw chart times instead of shifting GT by the detected best offset")
    ap.add_argument("--full-drum", action="store_true",
                    help="run the model once on the whole BS-Roformer drum stem (all lanes) instead of the "
                    "MDX23C per-instrument split; no cross-instrument isolation/leakage")
    args = ap.parse_args()

    import gc

    import torch
    from app.pipeline.separate import Separator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stems_cache:
        stems_cache = Path(args.stems_cache)
        stems_cache.mkdir(parents=True, exist_ok=True)
        stems_tmp = None
    else:  # need stems to persist across the two phases; use a run-lifetime tmp
        stems_tmp = tempfile.TemporaryDirectory()
        stems_cache = Path(stems_tmp.name)

    zips = sorted(Path(args.maps_dir).glob("*.zip"))
    print(
        f"{len(zips)} maps; checkpoint={args.checkpoint}; max_seconds={args.max_seconds}; "
        "adaptive hat/cymbal folding (per-map, optimistic)",
        flush=True,
    )

    # ---- Phase A: per song, mix -> drum stem (stems_all) -> per-instrument
    # stems (stems_per: kick/snare/hi-hat/cymbals/toms). MERT is NOT loaded yet
    # so the separator and MERT never coexist on a small GPU. ----
    maps: list[tuple] = []  # (zip_path, gt, drum_stem, {pitch: stem_path})
    sep = Separator()
    sep.load()
    for zp in zips:
        print(f"\n=== {zp.name} (separate) ===", flush=True)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with zipfile.ZipFile(zp) as z:
                z.extractall(root)
            chart = _pick_rlrr(root)
            if chart is None:
                print("  no .rlrr; skipping", flush=True)
                continue
            gt = {
                ln: [t for t in ts if args.max_seconds is None or t < args.max_seconds]
                for ln, ts in rlrr.onsets_by_lane(chart).items()
            }
            # only score sparse aux-perc lanes if the kit actually charts them
            aux_keep = {ln for ln in ("mp", "mc") if rlrr.has_lane_track(chart, ln)}
            drum_cached = stems_cache / f"{zp.stem}.drum.flac"
            # the BS-Roformer drum stem is needed in both modes (full-drum scores
            # it directly; the split mode also uses it for the alignment envelope)
            if not drum_cached.exists():
                mix_wav = root / "_mix.wav"
                ok, case = build_mix(
                    root, rlrr.song_tracks(chart), rlrr.drum_tracks(chart),
                    SEP_SR, mix_wav, args.max_seconds, args.drum_corr_threshold,
                )
                if not ok:
                    print("  no resolvable audio; skipping", flush=True)
                    continue
                print(f"  mix: {case}", flush=True)
                drum_cached.write_bytes(Path(sep.run_stems_all(mix_wav, root).drum_stem).read_bytes())
            if args.full_drum:
                print("  full drum stem (no per-instrument split)", flush=True)
                pieces = {}
            else:
                piece_cached = {p: stems_cache / f"{zp.stem}.{p}.flac" for p in STEM_TO_LANES}
                if all(pp.exists() for pp in piece_cached.values()):
                    print("  using cached per-instrument stems", flush=True)
                    pieces = dict(piece_cached)
                else:
                    per = sep.run_stems_per(drum_cached, root).per_instrument  # {pitch: path}
                    pieces = {}
                    for p, path in per.items():
                        if p in STEM_TO_LANES:
                            piece_cached[p].write_bytes(Path(path).read_bytes())
                            pieces[p] = piece_cached[p]
                    print(f"  per-instrument stems: {sorted(pieces)}", flush=True)
            maps.append((zp, gt, drum_cached, pieces, aux_keep))

    # free the separator's GPU memory before loading MERT (no coexistence)
    del sep
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Phase B: run the model on EACH isolated instrument stem; keep only
    # the onsets in lanes that belong to that stem, count the rest as leakage. ----
    model, meta = inference.load_model(args.checkpoint, device)
    encoder = embeddings.MertEncoder(name=meta["encoder"], layer=meta["encoder_layer"])
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    leak: dict[str, dict] = defaultdict(lambda: {"matched": 0, "leaked": 0, "to": Counter()})
    flagged = []

    for zp, gt, drum_stem, pieces, aux_keep in maps:
        print(f"\n=== {zp.name} (score) ===", flush=True)
        # alignment / quality check against the combined drum-stem envelope
        env, env_fps = forced_align.onset_envelope(drum_stem, max_seconds=args.max_seconds)
        floor = postfilter.support_floor_from_env(env, args.support_percentile)
        off, s0 = _global_offset(gt, env, env_fps, floor, args.align_window, args.offset_window)
        n_gt = sum(len(v) for v in gt.values())
        status = "ok" if s0 >= args.min_support else "SUSPECT"
        if status == "SUSPECT":
            flagged.append((zp.name, s0, off))
        # The median offset is the true global GT->audio shift. Apply it only
        # when it's a meaningful systematic offset (not the ~7ms flux-lag of an
        # already-aligned chart, which would just nudge GT off where the model
        # fires); the envelope peak is a slightly-late reference vs the onset.
        apply_off = off if (args.offset_correct and abs(off) > args.offset_correct_min) else 0.0
        gt_scored = (
            {ln: [t + apply_off for t in ts] for ln, ts in gt.items()} if apply_off else gt
        )
        print(
            f"  GT onsets={n_gt}  support@0={s0:.2f}  offset={off * 1000:+.0f}ms  [{status}]"
            + (f"  (corrected @ {apply_off * 1000:+.0f}ms)" if apply_off else ""),
            flush=True,
        )

        # run the model on each isolated stem; trust only matching lanes. Also
        # gate each kept lane through ITS OWN stem's onset-strength envelope
        # (the deterministic filter) -> est_filt, so ride is gated on the
        # cymbals-only stem, hats on the hi-hat stem, etc.
        est: dict[str, list[float]] = {lane: [] for lane in meta["lanes"]}
        est_filt: dict[str, list[float]] = {lane: [] for lane in meta["lanes"]}
        if args.full_drum:
            # one model pass over the whole drum stem; keep ALL lanes (no
            # per-instrument isolation, so no cross-instrument leakage). Filter
            # against the same drum-stem envelope used for alignment.
            raw = inference.transcribe(
                drum_stem, model, meta, encoder,
                max_seconds=args.max_seconds, window_seconds=args.window_seconds,
            )
            for lane, ts in raw.items():
                if not ts:
                    continue
                est[lane].extend(ts)
                est_filt[lane].extend(
                    postfilter.filter_lane(np.asarray(ts, dtype=float), env, env_fps, args.align_window, floor)
                )
        else:
            for pitch, stem_path in pieces.items():
                matching = STEM_TO_LANES.get(pitch, ())
                raw = inference.transcribe(
                    stem_path, model, meta, encoder,
                    max_seconds=args.max_seconds, window_seconds=args.window_seconds,
                )
                senv, sfps = forced_align.onset_envelope(stem_path, max_seconds=args.max_seconds)
                sfloor = postfilter.support_floor_from_env(senv, args.support_percentile)
                for lane, ts in raw.items():
                    if not ts:
                        continue
                    if lane in matching:
                        est[lane].extend(ts)
                        est_filt[lane].extend(
                            postfilter.filter_lane(np.asarray(ts, dtype=float), senv, sfps, args.align_window, sfloor)
                        )
                        leak[pitch]["matched"] += len(ts)
                    else:  # cross-instrument hallucination: discard + count
                        leak[pitch]["leaked"] += len(ts)
                        leak[pitch]["to"][lane] += len(ts)
        for d in (est, est_filt):
            for ts in d.values():
                ts.sort()

        filt_pairs = {lbl: el for lbl, _r, el in rlrr.comparison_pairs(gt_scored, est_filt)}
        for label, ref, est_l in rlrr.comparison_pairs(gt_scored, est):
            if not ref:
                continue
            if label in ("mp", "mc") and label not in aux_keep:
                continue  # no charted percussion track for this lane -> don't score it
            mr = metrics.onset_f1(ref, np.asarray(est_l, dtype=float), meta["onset_tolerance_s"])
            mf = metrics.onset_f1(ref, np.asarray(filt_pairs.get(label, []), dtype=float), meta["onset_tolerance_s"])
            agg[label]["f"].append(mr["f"])
            agg[label]["ff"].append(mf["f"])
            agg[label]["p"].append(mr["p"])
            agg[label]["r"].append(mr["r"])
            agg[label]["pf"].append(mf["p"])
            agg[label]["rf"].append(mf["r"])

    # ---- reports ----
    def _m(v):
        return sum(v) / len(v) if v else 0.0

    print("\n==== ParaDB per-lane onset-F1: raw model vs +per-stem envelope filter ====", flush=True)
    print("  label  F_raw  F_filt    dF    P_raw>P_filt  R_raw>R_filt  maps", flush=True)
    print("  (h/cym = chart lumped the group; hc/ho/rd/cr = chart split it)", flush=True)
    for label in rlrr.REPORT_ORDER:
        a = agg[label]
        if not a["f"]:
            continue
        fr, ff = _m(a["f"]), _m(a["ff"])
        print(
            f"  {label:4s} {fr:6.3f} {ff:6.3f} {ff - fr:+6.3f}   {_m(a['p']):.3f}>{_m(a['pf']):.3f}  "
            f"{_m(a['r']):.3f}>{_m(a['rf']):.3f}  {len(a['f'])}",
            flush=True,
        )

    print("\n==== cross-instrument leakage (onsets fired in the WRONG lane, discarded) ====", flush=True)
    print("  stem  matched  leaked  leak%   top wrong lanes", flush=True)
    for pitch in ("k", "s", "h", "c", "t"):
        d = leak[pitch]
        tot = d["matched"] + d["leaked"]
        if not tot:
            continue
        top = ", ".join(f"{ln}:{n}" for ln, n in d["to"].most_common(3))
        print(f"  {pitch:4s} {d['matched']:7d} {d['leaked']:7d} {100 * d['leaked'] / tot:5.1f}%   {top}", flush=True)

    if flagged:
        print(f"\n{len(flagged)} SUSPECT maps (low support or large offset, review/exclude):", flush=True)
        for name, s0, off in flagged:
            print(f"  {name}: support@0={s0:.2f} offset={off * 1000:+.0f}ms", flush=True)


if __name__ == "__main__":
    main()
