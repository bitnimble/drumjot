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

LICENSE / SCOPE: ParaDB songs are copyrighted + the charts unlicensed. This is a
RESEARCH measurement harness ONLY (held-out eval) -- not a step toward shipping a
model trained on ParaDB data.
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
    embeddings,
    forced_align,
    inference,
    metrics,
    paradb,
    postfilter,
    rlrr,
)
from drumjot_training.parampred import eval_gap, hybrid, regressor, report  # noqa: E402

SEP_SR = paradb.SEP_SR  # separate at full band so cymbal/hi-hat content survives

# stems_per pitch -> the model lanes that legitimately belong to that isolated
# stem. Onsets the model fires in any OTHER lane when fed this stem are
# cross-instrument leakage (hallucination): discarded + counted. (MDX23C merges
# ride+crash into one "cymbals" stem -> c; misc-perc has no stem.) Shared with
# the cull + training via the paradb module so the three can't drift.
STEM_TO_LANES = paradb.PERSTEM_TO_LANES

# 6-way drumsep (aufr33-jarredou v0.1) eval: the merged `c` cymbal stem is
# replaced by isolated `rd` (ride) + `cr` (crash) stems, and hi-hat comes from the
# 6-way `hh` stem. Selected by --six-way so a model trained on 6-way stems is
# scored in its matching separation domain (needs a 6-way stems-cache; see
# tmp/presep_eval_6way.py). k/s/t are the reused 5-way stems, as in training.
STEM_TO_LANES_6WAY: dict[str, tuple[str, ...]] = {
    "k": ("k",), "s": ("s", "ss"), "h": ("hc", "ho"),
    "rd": ("rd",), "cr": ("cr",), "t": ("t",),
}

# Mix reconstruction + chart offset live in `drumjot_training.paradb` (shared with
# the corpus-cull gate and the separation step). Thin aliases keep the call sites
# below unchanged.
_pick_rlrr = paradb.pick_chart
build_mix = paradb.build_mix
_global_offset = paradb.global_offset


def _requested_lanes(args):
    """The lanes the oracle report is restricted to (None = all checkpoint lanes)."""
    return {ln.strip() for ln in args.lanes.split(",")} if args.lanes else None


def _accumulate_gap(gap_records, model, meta, encoder, gt_scored, drum_stem, pieces, args, predictor):
    """Read each stem's raw activation curves and append per-lane oracle-gap
    records (current vs oracle, plus predicted when a predictor is loaded). A
    separate MERT pass from the scoring path -- this mode is a deliberate
    analysis, not the deployed hot path."""
    req = _requested_lanes(args)
    # (stem audio, lanes that legitimately belong to it) -- mirrors how `est` is
    # built so the gap is measured on the same isolated-stem inputs. With --lanes,
    # drop stems carrying no requested lane so the model never runs on them.
    if args.full_drum:
        targets_ = [(drum_stem, req)]
    else:
        targets_ = []
        for pitch, stem_path in pieces.items():
            restrict = set(STEM_TO_LANES.get(pitch, ()))
            if req is not None:
                restrict &= req
            if restrict:
                targets_.append((stem_path, restrict))
    for stem_path, restrict in targets_:
        probs, fps = inference.stitched_probs(
            stem_path, model, meta, encoder, args.max_seconds, args.window_seconds
        )
        wave = sr = None
        if predictor is not None:
            import librosa

            wave, sr = librosa.load(str(stem_path), sr=SEP_SR, mono=True)
        gap_records.extend(eval_gap.lane_gap_records(
            probs, fps, meta["lanes"], meta["thresholds"], gt_scored,
            default_threshold=meta["peak_threshold"], tolerance=meta["onset_tolerance_s"],
            restrict_lanes=restrict, predictor=predictor, waveform=wave, sr=sr,
        ))


def print_reports(agg, leak, gap_records, flagged, *, oracle_report, predictor):
    """Render the per-lane F1 + leakage + oracle-gap + flagged reports from the
    accumulated maps. Shared by the single-process run and the shard merge
    (merge_paradb_shards.py) so both print identically. `predictor` is used only
    for its truthiness (whether to add the predicted/hybrid columns)."""
    def _m(v):
        return sum(v) / len(v) if v else 0.0

    print("\n==== ParaDB per-lane onset-F1: bare peak-pick vs +shared deterministic picker ====", flush=True)
    print("  label  F_bare F_pick   dF    P_bare>P_pick  R_bare>R_pick  maps", flush=True)
    print("  (F_pick = deployed picker = the headline number; h/cym folded, hc/ho/rd/cr split)", flush=True)
    for label in rlrr.REPORT_ORDER:
        a = agg[label]
        if not a["f"]:
            continue
        fb, fp = _m(a["fb"]), _m(a["f"])
        print(
            f"  {label:4s} {fb:6.3f} {fp:6.3f} {fp - fb:+6.3f}   {_m(a['pb']):.3f}>{_m(a['p']):.3f}  "
            f"{_m(a['rb']):.3f}>{_m(a['r']):.3f}  {len(a['f'])}",
            flush=True,
        )

    print("\n==== cross-instrument leakage (onsets fired in the WRONG lane, discarded) ====", flush=True)
    print("  stem  matched  leaked  leak%   top wrong lanes", flush=True)
    for pitch in STEM_TO_LANES:  # 5-way k/s/h/c/t or 6-way k/s/h/rd/cr/t
        d = leak[pitch]
        tot = d["matched"] + d["leaked"]
        if not tot:
            continue
        top = ", ".join(f"{ln}:{n}" for ln, n in d["to"].most_common(3))
        print(f"  {pitch:4s} {d['matched']:7d} {d['leaked']:7d} {100 * d['leaked'] / tot:5.1f}%   {top}", flush=True)

    if oracle_report and gap_records:
        gaps = report.aggregate(gap_records)
        order = [ln for ln in rlrr.REPORT_ORDER if ln in gaps] or sorted(gaps)
        print("\n" + report.format_report(gaps, lane_order=order), flush=True)
        tot = sum(g.gap for g in gaps.values()) / len(gaps)
        cap = sum(g.captured for g in gaps.values()) / len(gaps)
        src = "predicted" if predictor else "(no predictor: predicted == current)"
        print(f"  mean oracle gap {tot:+.3f} F1; mean captured {cap:+.3f} {src}", flush=True)
        if predictor:  # the hybrid routes hc->determ, cymbals->learned (needs the predictor)
            print("\n" + hybrid.format_hybrid(gaps, hybrid.DEFAULT_ROUTING, lane_order=order), flush=True)
            print(f"  mean captured {hybrid.captured(gaps, hybrid.DEFAULT_ROUTING):+.3f} hybrid", flush=True)

    if flagged:
        print(f"\n{len(flagged)} SUSPECT maps (low support or large offset, review/exclude):", flush=True)
        for name, s0, off in flagged:
            print(f"  {name}: support@0={s0:.2f} offset={off * 1000:+.0f}ms", flush=True)


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
    ap.add_argument("--oracle-report", action="store_true",
                    help="also report per-lane onset-F1 at the per-song ORACLE peakpick params (the ceiling) "
                    "vs today's global params -- the adaptive-param gap gate (design spec build-order step 1). "
                    "Adds an extra MERT pass per stem to read the raw activation curves.")
    ap.add_argument("--param-predictor", default=None,
                    help="optional ParamRegressor joblib artifact; when set, the oracle report adds a "
                    "'predicted' column scoring the predictor's per-song params (label-free inference).")
    ap.add_argument("--lanes", default=None,
                    help="comma-separated lanes to restrict the --oracle-report to (e.g. hc,ho,rd,cr for a "
                    "hi-hat+cymbal checkpoint); stems carrying none of them are skipped. Default: all "
                    "checkpoint lanes.")
    ap.add_argument("--log", default=None,
                    help="tee stdout+stderr to this file (self-log; no manual redirect needed)")
    ap.add_argument("--shard", default=None,
                    help="process only a stride 'I/N' of the maps (maps[I::N]) -- run N "
                    "processes over disjoint subsets in parallel, then combine with "
                    "--dump + merge_paradb_shards.py. Cache-hit shards load neither the "
                    "separator nor MERT, so they share one GPU.")
    ap.add_argument("--dump", default=None,
                    help="pickle (agg, leak, gap_records, flagged) here instead of printing "
                    "the report; merge_paradb_shards.py combines shard dumps into one report.")
    ap.add_argument("--maps-list", default=None,
                    help="file of newline-separated .zip paths to process (overrides the "
                    "--maps-dir glob + --shard); the parallel orchestrator writes one per "
                    "worker so it can pin all uncached songs to one GPU encoder worker.")
    ap.add_argument("--require-cached", action="store_true",
                    help="skip (don't score) any song whose MERT features aren't fully cached, "
                    "so this worker never loads the encoder -- for the cache-only workers in a "
                    "parallel run (the single encoder worker omits this and does the uncached).")
    ap.add_argument("--six-way", action="store_true",
                    help="score a 6-way-drumsep-trained model: route ride/crash from isolated "
                    "rd/cr stems + hi-hat from the 6-way hh stem (needs a 6-way --stems-cache).")
    args = ap.parse_args()

    if args.six_way:  # module global read by the piece cache + scoring loop + _accumulate_gap
        globals()["STEM_TO_LANES"] = STEM_TO_LANES_6WAY

    import gc

    import torch
    from app.pipeline.separate import Separator

    from drumjot_training import runtime
    runtime.tee_stdio(args.log)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stems_cache:
        stems_cache = Path(args.stems_cache)
        stems_cache.mkdir(parents=True, exist_ok=True)
        stems_tmp = None
    else:  # need stems to persist across the two phases; use a run-lifetime tmp
        stems_tmp = tempfile.TemporaryDirectory()
        stems_cache = Path(stems_tmp.name)

    if args.maps_list:
        zips = [Path(p) for p in Path(args.maps_list).read_text().splitlines() if p.strip()]
    else:
        zips = sorted(Path(args.maps_dir).glob("*.zip"))
        if args.shard:
            si, sn = (int(x) for x in args.shard.split("/"))
            zips = zips[si::sn]
    _scope = f"shard {args.shard}" if args.shard else ("from maps-list" if args.maps_list else "")
    print(
        f"{len(zips)} maps{f' ({_scope})' if _scope else ''}; "
        f"checkpoint={args.checkpoint}; max_seconds={args.max_seconds}; "
        "adaptive hat/cymbal folding (per-map, optimistic)",
        flush=True,
    )

    # ---- Phase A: per song, mix -> drum stem (stems_all) -> per-instrument
    # stems (stems_per: kick/snare/hi-hat/cymbals/toms). MERT is NOT loaded yet
    # so the separator and MERT never coexist on a small GPU. ----
    maps: list[tuple] = []  # (zip_path, gt, drum_stem, {pitch: stem_path})
    # Lazily construct the separator on the first cache MISS: a fully-cached run
    # (eval over pre-separated stems) never loads BS-Roformer/MDX23C, so several
    # sharded eval processes can share one GPU.
    sep = None

    def _sep():
        nonlocal sep
        if sep is None:
            sep = Separator()
            sep.load()
        return sep

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
                drum_cached.write_bytes(Path(_sep().run_stems_all(mix_wav, root).drum_stem).read_bytes())
            if args.full_drum:
                print("  full drum stem (no per-instrument split)", flush=True)
                pieces = {}
            else:
                piece_cached = {p: stems_cache / f"{zp.stem}.{p}.flac" for p in STEM_TO_LANES}
                if all(pp.exists() for pp in piece_cached.values()):
                    print("  using cached per-instrument stems", flush=True)
                    pieces = dict(piece_cached)
                else:
                    per = _sep().run_stems_per(drum_cached, root).per_instrument  # {pitch: path}
                    pieces = {}
                    for p, path in per.items():
                        if p in STEM_TO_LANES:
                            piece_cached[p].write_bytes(Path(path).read_bytes())
                            pieces[p] = piece_cached[p]
                    print(f"  per-instrument stems: {sorted(pieces)}", flush=True)
            maps.append((zp, gt, drum_cached, pieces, aux_keep))

    # free the separator's GPU memory before loading MERT (no coexistence)
    sep = None
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
    predictor = regressor.ParamRegressor.load(args.param_predictor) if args.param_predictor else None
    gap_records: list[report.GapRecord] = []

    for zp, gt, drum_stem, pieces, aux_keep in maps:
        if args.require_cached:  # cache-only worker: never load the encoder
            to_check = [drum_stem] if args.full_drum else list(pieces.values())
            if not all(embeddings.windows_cached(sp, encoder, meta, args.window_seconds, args.max_seconds)
                       for sp in to_check):
                print(f"\n=== {zp.name} (skip: not fully cached; encoder worker handles it) ===", flush=True)
                continue
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

        # run the model on each isolated stem; keep only matching lanes (count the
        # rest as cross-instrument leakage). Two picks from the SAME model output,
        # for an ablation: `est` = the shared per-lane deterministic picker
        # (per-lane min-distance + prominence + decay-reset; what we deploy) and
        # `est_bare` = a bare height+min-distance pick (baseline). The picker's
        # value shows as F_pick - F_bare per lane. (Replaces the old onset-envelope
        # support gate, which was a measured no-op.)
        est: dict[str, list[float]] = {lane: [] for lane in meta["lanes"]}
        est_bare: dict[str, list[float]] = {lane: [] for lane in meta["lanes"]}
        if args.full_drum:
            # one model pass over the whole drum stem; keep ALL lanes (no
            # per-instrument isolation, so no cross-instrument leakage).
            bare, full = inference.transcribe_dual(
                drum_stem, model, meta, encoder,
                max_seconds=args.max_seconds, window_seconds=args.window_seconds,
            )
            for lane in meta["lanes"]:
                est[lane].extend(full[lane])
                est_bare[lane].extend(bare[lane])
        else:
            for pitch, stem_path in pieces.items():
                matching = STEM_TO_LANES.get(pitch, ())
                bare, full = inference.transcribe_dual(
                    stem_path, model, meta, encoder,
                    max_seconds=args.max_seconds, window_seconds=args.window_seconds,
                )
                for lane in meta["lanes"]:
                    ft = full[lane]
                    if lane in matching:
                        est[lane].extend(ft)
                        est_bare[lane].extend(bare[lane])
                        leak[pitch]["matched"] += len(ft)
                    elif ft:  # cross-instrument hallucination: discard + count
                        leak[pitch]["leaked"] += len(ft)
                        leak[pitch]["to"][lane] += len(ft)
        for d in (est, est_bare):
            for ts in d.values():
                ts.sort()

        if args.oracle_report:
            _accumulate_gap(gap_records, model, meta, encoder, gt_scored, drum_stem, pieces, args, predictor)

        bare_pairs = {lbl: el for lbl, _r, el in rlrr.comparison_pairs(gt_scored, est_bare)}
        for label, ref, est_l in rlrr.comparison_pairs(gt_scored, est):
            if not ref:
                continue
            if label in ("mp", "mc") and label not in aux_keep:
                continue  # no charted percussion track for this lane -> don't score it
            mr = metrics.onset_f1(ref, np.asarray(est_l, dtype=float), meta["onset_tolerance_s"])
            mb = metrics.onset_f1(ref, np.asarray(bare_pairs.get(label, []), dtype=float), meta["onset_tolerance_s"])
            agg[label]["f"].append(mr["f"])  # full picker (deployed, headline = past F_raw)
            agg[label]["fb"].append(mb["f"])  # bare baseline
            agg[label]["p"].append(mr["p"])
            agg[label]["r"].append(mr["r"])
            agg[label]["pb"].append(mb["p"])
            agg[label]["rb"].append(mb["r"])

    # ---- reports (or dump the raw accumulators for a sharded run, which
    # merge_paradb_shards.py combines into one identical report) ----
    if args.dump:
        import pickle

        Path(args.dump).write_bytes(pickle.dumps((
            {k: dict(v) for k, v in agg.items()},  # drop the lambda factories so it pickles
            {k: {"matched": d["matched"], "leaked": d["leaked"], "to": dict(d["to"])}
             for k, d in leak.items()},
            gap_records, flagged,
        )))
        print(f"\ndumped {len(maps)} maps' records -> {args.dump}", flush=True)
        return
    print_reports(agg, leak, gap_records, flagged,
                  oracle_report=args.oracle_report, predictor=predictor)


if __name__ == "__main__":
    main()
