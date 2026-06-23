"""Phase-3 ParaDB training-tree builder: kept maps -> per-stem `paradb-sep` tree.

Consumes the corpus-cull manifest (`build_paradb_manifest.py`) and the cached
BS-Roformer drum stems it already produced, and builds the per-instrument tree
the trainer consumes (mirrors enst-sep / mdb-sep):

    <out>/perstem/<pitch>/<map_id>.flac   # pitch in k/s/h/c/t (MDX23C)
    <out>/onsets/<map_id>.json            # offset-corrected 9-lane GT onsets

Only the **kept-training** maps are built: kept = status ok AND support/recall
above the chosen cut (`--min-support`/`--min-recall`); the **held-out eval** ids
(deterministic by map-id hash, `--holdout-frac`) are excluded and FROZEN to
`--eval-ids-out` (commit it, it's the benchmark split, shared by the param
corpus + eval_paradb).

Reuses the gate's cached BS-Roformer drum stem (`--stems-cache`), so it runs only
MDX23C (the per-instrument split) -- saving the BS-Roformer pass, though MDX23C
itself is ~gate-scale (~100 s/map on the 1660). Distributed (atomic per-map
claim, same as the gate) + resumable (skips maps already gated -> stem_scores
written) + pipelined (GPU split on the main thread; chart-parse on a producer
thread; per-stem P/R gate + FLAC + onsets write on a writer thread).

PER-STEM P/R GATE: after MDX23C, each isolated instrument stem is scored
precision + recall (see `_score_stem`); stems below `--stem-min-support`/
`--stem-min-recall` are dropped (so a map that omits an instrument's lane can't
poison that stem's training). All per-stem scores -> `stem_scores/<id>.json`.

  MODELS_DIR=/codebox-workspace/drumjot/models-cache \
  scripts/sandbox-run env PYTHONPATH=training:dsp:transcriber python3 \
      training/scripts/separate_paradb_dataset.py \
      --maps-dir /codebox-workspace/datasets/paradb/zips \
      --stems-cache /codebox-workspace/datasets/paradb/_drumstems \
      --manifest /codebox-workspace/datasets/paradb/paradb_manifest.json \
      --out-dir /codebox-workspace/datasets/paradb-sep \
      --eval-ids-out training/paradb_eval_ids.json \
      --min-support 0.97 --min-recall 0.90 --holdout-frac 0.05

LICENSE / SCOPE: ParaDB songs are copyrighted + the charts unlicensed. This
training tree is RESEARCH-ONLY; we do NOT intend to ship a model trained on
ParaDB data, and the held-out eval ids are for measurement only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in ("training", "dsp", "transcriber"):
    sys.path.insert(0, str(_REPO / _p))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from drumjot_training import (  # noqa: E402
    clean,
    forced_align,
    metrics,
    paradb,
    postfilter,
    rlrr,
    runtime,
)

_PERSTEM_PITCHES = ("k", "s", "h", "c", "t")
# stem pitch -> a representative lane whose peak-pick params (min-distance +
# decay-reset) suit that stem: k/s/t = clean, h = hat (ring), c = cymbal (ring).
_STEM_PARAM_LANE = {"k": "k", "s": "s", "t": "t", "h": "hc", "c": "rd"}
_RUNNER = f"{os.uname().nodename}:{os.getpid()}"


def _score_stem(stem_path: Path, restricted: dict, drum_conf_floor: float, pitch: str, args):
    """Per-stem precision + recall against the ISOLATED instrument stem.

    precision (support): fraction of the chart's onsets for this stem's lanes that
    land on a transient in this stem. recall: fraction of the stem's own
    HIGH-CONFIDENCE onsets the chart covers -- catches a map that dropped this
    instrument's lane (stem audio has clear hits, chart lane empty -> recall ~0).

    Confident onsets on an ISOLATED stem need care (a naive percentile floor
    over-fires on the sparse stem + on cymbal/hi-hat ring):
      - PEAK-RELATIVE height: `peak_frac` x the stem's own max envelope (robust to
        the sparse-stem percentile), floored by `abs_frac` x the DRUM stem's
        percentile so a SILENT stem (song lacks this instrument) yields 0 confident
        onsets -> recall 1.0 (kept, valid negative), not a false drop.
      - the lane's min-distance + decay-reset (hat/cym ring collapses to ONE onset,
        not a phantom stream). prominence is omitted (its [0,1]-activation scale
        doesn't map to the onset-strength envelope; height+min-dist+reset suffice).
    Returns (support, recall, n_confident)."""
    env, fps = forced_align.onset_envelope(stem_path)
    if env.size == 0 or float(np.max(env)) <= 0.0:
        return 1.0, 1.0, 0  # silent stem -> vacuously fine (empty-negative example)
    supp_floor = postfilter.support_floor_from_env(env, args.stem_support_percentile)
    support = clean.support_score(
        restricted, env, fps, window_s=args.stem_window, support_floor=supp_floor)["fraction"]
    pp = metrics.LANE_PEAK_PARAMS.get(_STEM_PARAM_LANE.get(pitch, ""), metrics.DEFAULT_PEAK_PARAMS)
    conf_floor = max(args.stem_recall_peak_frac * float(np.max(env)),
                     args.stem_recall_abs_frac * drum_conf_floor)
    rec = clean.recall_score(
        restricted, env, fps, confident_floor=conf_floor, window_s=args.stem_window,
        min_distance_s=pp["min_distance_s"], decay_reset_frac=pp["decay_reset_frac"],
        decay_reset_floor=conf_floor,
    )
    return support, rec["fraction"], rec["n_confident"]


def _claim(claims: Path, map_id: str, stale_s: float) -> bool:
    """Atomic O_EXCL claim (same scheme as the gate); reclaim a stale lock."""
    lock = claims / f"{map_id}.lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
        return True
    except FileExistsError:
        try:
            if time.time() - lock.stat().st_mtime < stale_s:
                return False
            lock.unlink()
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            return True
        except (FileExistsError, FileNotFoundError):
            return False


def _release(claims: Path, map_id: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        (claims / f"{map_id}.lock").unlink()


def _done(out: Path, map_id: str) -> bool:
    """Done once the per-stem gate has run (stem_scores written). Kept stems are
    written; stems dropped by the gate are intentionally absent, so we must NOT
    require all 5 -- that would re-process gated maps forever."""
    return (out / "stem_scores" / f"{map_id}.json").exists()


def _chart_onsets(zip_path: Path, offset_ms: float, offset_correct_min: float) -> dict:
    """Parse the hardest chart from the zip and apply the SAME global offset the
    gate scored with (so the stored labels line up with the audio). Extracts only
    the `.rlrr` members (skips audio + macOS junk) to a temp dir -> cheap."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with zipfile.ZipFile(zip_path) as z:
            for n in z.namelist():
                if n.lower().endswith(".rlrr") and "__MACOSX" not in n and not Path(n).name.startswith("._"):
                    z.extract(n, tdp)
        chart = paradb.pick_chart(tdp)
        if chart is None:
            raise RuntimeError("no chart in zip")
        gt = rlrr.onsets_by_lane(chart)
    off = offset_ms / 1000.0
    apply_off = off if abs(off) > offset_correct_min else 0.0
    return paradb.shift_onsets(gt, apply_off)


def _write_flac(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(str(src))
    tmp = dst.with_suffix(f".{os.getpid()}.tmp")
    sf.write(str(tmp), y, sr, format="FLAC")
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps-dir", required=True, help="folder of maps__*.zip (for chart onsets)")
    ap.add_argument("--stems-cache", required=True, help="cached BS-Roformer drum stems from the gate")
    ap.add_argument("--manifest", required=True, help="paradb_manifest.json from build_paradb_manifest.py")
    ap.add_argument("--out-dir", required=True, help="output paradb-sep per-stem tree root")
    ap.add_argument("--eval-ids-out", required=True, help="write the frozen held-out eval ids here (commit it)")
    ap.add_argument("--min-support", type=float, default=0.97, help="cull: keep support_corr >= this")
    ap.add_argument("--min-recall", type=float, default=0.90, help="cull: keep recall >= this")
    ap.add_argument("--min-onsets", type=int, default=0, help="cull: keep n_onsets >= this")
    ap.add_argument("--holdout-frac", type=float, default=0.05, help="fraction of kept maps held out for eval")
    ap.add_argument("--offset-correct-min", type=float, default=0.025,
                    help="apply chart offset only if |offset| exceeds this (match the gate)")
    # PER-STEM precision+recall gate: scored against each ISOLATED instrument stem,
    # then drop individual stems below threshold (catches maps that omit a whole
    # instrument lane -> that stem's audio has hits the chart never charted).
    # Defaults 0.0 = score+record but drop nothing (pick the cut from the printed
    # distribution, like the map gate, then re-run with thresholds).
    ap.add_argument("--stem-min-support", type=float, default=0.0, help="drop a stem below this precision")
    ap.add_argument("--stem-min-recall", type=float, default=0.0, help="drop a stem below this recall")
    ap.add_argument("--stem-support-percentile", type=float, default=60.0)
    ap.add_argument("--stem-recall-percentile", type=float, default=92.0,
                    help="DRUM-stem percentile used as the absolute reference for the silent-stem floor")
    ap.add_argument("--stem-recall-peak-frac", type=float, default=0.25,
                    help="confident-onset height = this x the stem's own max envelope (peak-relative; "
                    "robust to the sparse-stem percentile that over-fires)")
    ap.add_argument("--stem-recall-abs-frac", type=float, default=0.15,
                    help="lower bound on the confident-onset floor = this x the DRUM-stem percentile, so "
                    "a SILENT stem yields 0 confident onsets (recall 1.0, kept) not a false drop")
    ap.add_argument("--stem-window", type=float, default=0.05, help="+/- window for per-stem support/recall (s)")
    ap.add_argument("--work-dir", default=None, help="claims dir for multi-runner (default <out-dir>/_sep_claims)")
    ap.add_argument("--stale-minutes", type=float, default=60.0)
    ap.add_argument("--scratch-dir", default=None, help="fast local dir for the MDX23C temp output")
    ap.add_argument("--limit", type=int, default=0, help="cap maps this runner claims (0=all)")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    runtime.tee_stdio(args.log)
    log = lambda s: print(s, flush=True)  # noqa: E731

    manifest = paradb.load_manifest(args.manifest)
    kept = paradb.kept_map_ids(manifest, min_support=args.min_support,
                               min_recall=args.min_recall, min_onsets=args.min_onsets)
    train_ids, eval_ids = paradb.holdout_split(kept, args.holdout_frac)
    Path(args.eval_ids_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_ids_out).write_text(json.dumps(
        {"holdout_frac": args.holdout_frac, "min_support": args.min_support,
         "min_recall": args.min_recall, "eval_ids": eval_ids}, indent=2))
    log(f"manifest: {len(manifest)} maps | kept {len(kept)} | train {len(train_ids)} | "
        f"held-out eval {len(eval_ids)} -> {args.eval_ids_out}")

    out = Path(args.out_dir)
    claims = Path(args.work_dir) if args.work_dir else out / "_sep_claims"
    claims.mkdir(parents=True, exist_ok=True)
    (out / "onsets").mkdir(parents=True, exist_ok=True)
    (out / "stem_scores").mkdir(parents=True, exist_ok=True)
    stems_cache = Path(args.stems_cache)
    maps_dir = Path(args.maps_dir)
    stale_s = args.stale_minutes * 60.0
    todo = [m for m in train_ids if not _done(out, m)]
    log(f"[{_RUNNER}] {len(train_ids)} train maps, {len(train_ids)-len(todo)} already built, {len(todo)} to do")
    if not todo:
        return

    from app.pipeline.separate import Separator

    ready_q: queue.Queue = queue.Queue(maxsize=3)
    write_q: queue.Queue = queue.Queue(maxsize=3)
    counts = {"claimed": 0, "done": 0, "skipped": 0, "dropped_stems": 0}
    stem_stats = {p: {"support": [], "recall": [], "dropped": 0} for p in _PERSTEM_PITCHES}

    def produce():
        claimed = 0
        for mid in todo:
            if args.limit and claimed >= args.limit:
                break
            if not _claim(claims, mid, stale_s):
                continue
            claimed += 1
            drum = stems_cache / f"{mid}.drum.flac"
            zp = maps_dir / f"maps__{mid}.zip"
            job = {"id": mid, "drum": drum, "onsets": None, "td": None, "status": "ok"}
            try:
                if not drum.exists() or not zp.exists():
                    job["status"] = "missing_inputs"
                else:
                    job["onsets"] = _chart_onsets(zp, manifest[mid].get("offset_ms", 0.0),
                                                  args.offset_correct_min)
            except Exception as exc:  # noqa: BLE001  bad chart -> skip this map
                job["status"] = f"parse_failed: {type(exc).__name__}"
            counts["claimed"] = claimed
            ready_q.put(job)
        ready_q.put(None)

    def write():
        while True:
            job = write_q.get()
            if job is None:
                break
            mid = job["id"]
            try:
                if job["status"] == "ok":
                    drum_env, _fps = forced_align.onset_envelope(job["drum"])
                    drum_conf = (postfilter.support_floor_from_env(drum_env, args.stem_recall_percentile)
                                 if drum_env.size else 0.0)
                    scores, kept_any = {}, False
                    for p, src in job["per"].items():
                        if p not in _PERSTEM_PITCHES or not src:
                            continue
                        restricted = {ln: job["onsets"].get(ln, []) for ln in paradb.PERSTEM_TO_LANES[p]}
                        sup, rec, nconf = _score_stem(Path(src), restricted, drum_conf, p, args)
                        keep = sup >= args.stem_min_support and rec >= args.stem_min_recall
                        scores[p] = {"support": round(sup, 4), "recall": round(rec, 4),
                                     "n_confident": nconf, "kept": keep}
                        stem_stats[p]["support"].append(sup)
                        stem_stats[p]["recall"].append(rec)
                        if keep:
                            _write_flac(Path(src), out / "perstem" / p / f"{mid}.flac")
                            kept_any = True
                        else:
                            counts["dropped_stems"] += 1
                            stem_stats[p]["dropped"] += 1
                    # stem_scores is the done-marker (written even if all stems dropped)
                    (out / "stem_scores" / f"{mid}.json").write_text(json.dumps(scores))
                    if kept_any:
                        (out / "onsets" / f"{mid}.json").write_text(json.dumps(job["onsets"]))
                    counts["done"] += 1
                else:
                    counts["skipped"] += 1
                    log(f"  [{_RUNNER}] skip {mid}: {job['status']}")
            except Exception as exc:  # noqa: BLE001
                counts["skipped"] += 1
                log(f"  [{_RUNNER}] write FAILED {mid}: {exc!r}")
            finally:
                _release(claims, mid)
                if job["td"] is not None:
                    shutil.rmtree(job["td"], ignore_errors=True)
            if counts["done"] % 25 == 0 and job["status"] == "ok":
                log(f"  [{_RUNNER}] {counts['done']} built")

    sep = Separator()
    sep.load()
    producer = threading.Thread(target=produce, daemon=True)
    writer = threading.Thread(target=write, daemon=True)
    producer.start()
    writer.start()
    while True:
        job = ready_q.get()
        if job is None:
            break
        if job["status"] == "ok":
            try:
                td = Path(tempfile.mkdtemp(prefix="paradbsep_", dir=args.scratch_dir or None))
                job["td"] = td
                job["per"] = sep.run_stems_per(job["drum"], td).per_instrument  # {pitch: path}
            except Exception as exc:  # noqa: BLE001
                job["status"] = f"sep_failed: {type(exc).__name__}: {exc}"
        write_q.put(job)
    write_q.put(None)
    writer.join()
    log(f"[{_RUNNER}] claimed {counts['claimed']}, processed {counts['done']}, "
        f"skipped {counts['skipped']}, dropped {counts['dropped_stems']} stems -> {out}")
    # per-stem precision/recall distribution (this runner's maps) -> tune the cut
    log("  per-stem P/R (this run):  pitch  n     sup med/min    rec med/min    dropped")
    for p in _PERSTEM_PITCHES:
        s, r = stem_stats[p]["support"], stem_stats[p]["recall"]
        if not s:
            continue
        log(f"    {p:2s}  {len(s):4d}   sup {np.median(s):.2f}/{min(s):.2f}   "
            f"rec {np.median(r):.2f}/{min(r):.2f}   dropped {stem_stats[p]['dropped']}")


if __name__ == "__main__":
    main()
