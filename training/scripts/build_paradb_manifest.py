"""Phase-1 ParaDB corpus cull: score every map's chart against its audio.

A lot of ParaDB maps are trash (auto-converted from MIDI, mis-synced, wrong
audio). This gate scores each map with the SAME cleanliness mechanism the
training pipeline uses, the support fraction (`clean.support_score`): the
fraction of charted onsets that land on a real drum transient. An auto-from-MIDI
chart rendered off the wrong tempo/offset has onsets between transients, so its
support collapses -> we cull it.

The honest signal is the isolated DRUM stem, not the full mix (dense music has
energy everywhere, so a misaligned chart falsely "supports" against the mix). So
for every map we run BS-Roformer (mix -> drum stem) and score against the drum
stem's onset-strength envelope, after a robust global chart->audio offset
correction. The drum stem is CACHED so the survivors' MDX23C split
(separate_paradb_dataset.py) reuses it -- nothing wasted on what we keep.

TWO signals, because support only catches PRECISION:
  - support (precision): fraction of charted onsets on a real transient. Catches
    auto-from-MIDI / mis-synced charts (notes where there's no drum hit).
  - recall: fraction of HIGH-CONFIDENCE audio onsets the chart covers. Catches a
    chart SIMPLER than the performance (100% precision but missing real hits) --
    only the unambiguous drum transients count, so we flag missing OBVIOUS hits.
A map is kept only if BOTH clear their thresholds (chosen from the histograms).

PIPELINED so the GPU never idles: a producer thread claims+extracts+builds the
next map's mix while the main thread separates the current map and a scorer
thread scores+writes the previous one (mirrors separate_egmd_dataset.py).

DISTRIBUTED (per-map claiming). Run this on as many boxes as you like at once
(1660 + 3080 + sandbox); they cooperate over the shared `--work-dir` with NO
coordination beyond the filesystem:
  - `results/<map_id>.json`  the scored entry; its existence == this map is DONE.
  - `claims/<map_id>.lock`   an atomic O_EXCL create == "I'm working on this".
A runner claims the next map with no result and no live claim, scores it, writes
the result, releases the lock. A crashed runner's lock goes stale after
`--stale-minutes` and is reclaimed. Faster boxes naturally do more maps. The
merged `paradb_manifest.json` is rebuilt from `results/` at the end of every run
(it's the source of truth for the cull; this script does NOT hard-drop anything).

  MODELS_DIR=/codebox-workspace/drumjot/models-cache \
  scripts/sandbox-run env PYTHONPATH=training:dsp:transcriber python3 \
      training/scripts/build_paradb_manifest.py \
      --maps-dir /codebox-workspace/datasets/paradb/zips \
      --stems-cache /codebox-workspace/datasets/paradb/_drumstems \
      --work-dir /codebox-workspace/datasets/paradb/_gate \
      --out-json /codebox-workspace/datasets/paradb/paradb_manifest.json

Same command on every box. `--merge`/`--report-only` just rebuild + print from
`results/` without separating.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # training/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transcriber"))

from drumjot_training import (  # noqa: E402
    clean,
    forced_align,
    paradb,
    postfilter,
    rlrr,
    runtime,
)

# A provisional keep flag is written at this support so the manifest is usable
# without a second pass; the REAL threshold is chosen from the histogram later.
DEFAULT_KEEP_SUPPORT = 0.6
_RUNNER = f"{socket.gethostname()}:{os.getpid()}"


# ---------------------------------------------------------------------------
# distributed per-map claiming (filesystem-only coordination)
# ---------------------------------------------------------------------------
def _result_path(work_dir: Path, map_id: str) -> Path:
    return work_dir / "results" / f"{map_id}.json"


def _claim_path(work_dir: Path, map_id: str) -> Path:
    return work_dir / "claims" / f"{map_id}.lock"


def _try_claim(work_dir: Path, map_id: str, stale_s: float) -> bool:
    """Atomically claim `map_id` for THIS runner. Returns True if we own it.

    The claim is an O_EXCL create of `claims/<id>.lock`: exactly one racing
    runner wins. A lock older than `stale_s` with no result is a crashed
    runner's leftover -> reclaim it (delete + retry the exclusive create once).
    """
    if _result_path(work_dir, map_id).exists():
        return False  # already done
    lock = _claim_path(work_dir, map_id)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            return False  # vanished (just released) -> let the next scan see the result
        if age < stale_s or _result_path(work_dir, map_id).exists():
            return False  # live claim, or finished while we looked
        try:
            lock.unlink()  # reclaim a stale (crashed-runner) lock
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except (FileExistsError, FileNotFoundError):
            return False  # another runner reclaimed it first
    with os.fdopen(fd, "w") as f:
        f.write(f"{_RUNNER} {time.time():.0f}\n")
    return True


def _release(work_dir: Path, map_id: str) -> None:
    """Drop our claim lock (result file is the durable done-marker)."""
    with contextlib.suppress(FileNotFoundError):
        _claim_path(work_dir, map_id).unlink()


def _publish_drum_flac(src: Path, dst: Path) -> None:
    """FLAC-encode the separator's drum WAV into the shared stems cache, atomically.

    Runs on the SCORER thread (off the GPU thread) so the NFS write + FLAC encode
    overlap the next separation. FLAC ~3x smaller than the raw WAV the separator
    emits, cutting both the NFS write and on-disk corpus size."""
    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(str(src))
    tmp = dst.with_suffix(f".{os.getpid()}.tmp")
    sf.write(str(tmp), y, sr, format="FLAC")
    os.replace(tmp, dst)


def _write_result(work_dir: Path, map_id: str, entry: dict) -> None:
    """Atomically write one map's scored entry (tmp + os.replace)."""
    dst = _result_path(work_dir, map_id)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(entry, f)
    os.replace(tmp, dst)


def _merge_results(work_dir: Path) -> dict:
    """Collect every `results/<id>.json` into a single manifest dict."""
    manifest: dict = {}
    for rp in sorted((work_dir / "results").glob("*.json")):
        try:
            entry = json.loads(rp.read_text())
        except Exception:  # noqa: BLE001  a half-written file (shouldn't happen w/ os.replace)
            continue
        manifest[entry.get("map_id", rp.stem)] = entry
    return manifest


def _extract_parse(zp: Path, td: Path, args) -> tuple[dict, dict | None, Path | None]:
    """Stage 1 (CPU): extract the zip into `td`, pick the hardest chart, parse
    onsets + sanity. Returns `(entry, gt, chart_path)`; gt/chart are None for a
    terminal status (no_chart / too_few_onsets). Raises only on a corrupt zip /
    malformed chart, which the producer catches into an `error` result."""
    map_id = paradb.map_id_of_zip(zp)
    entry: dict = {"zip": zp.name, "map_id": map_id, "status": "ok", "runner": _RUNNER}
    with zipfile.ZipFile(zp) as z:
        z.extractall(td)
    chart = paradb.pick_chart(td)
    if chart is None:
        entry["status"] = "no_chart"
        return entry, None, None
    entry["chart"] = str(chart.relative_to(td))
    gt = rlrr.onsets_by_lane(chart)
    per_lane_n = {ln: len(ts) for ln, ts in gt.items() if ts}
    entry["n_onsets"] = sum(per_lane_n.values())
    entry["per_lane_n"] = per_lane_n
    if entry["n_onsets"] < args.min_onsets:
        entry["status"] = "too_few_onsets"
        return entry, None, None
    return entry, gt, chart


def _score_from_stem(entry: dict, gt: dict, drum_path: Path, args) -> None:
    """Stage 3 (CPU): score `gt` against the drum stem's onset envelope, mutating
    `entry`. Two cleanliness signals at the offset-corrected chart:
      - support (PRECISION): fraction of charted onsets on a real transient.
      - recall: fraction of HIGH-CONFIDENCE audio onsets the chart covers
        (catches a chart simpler than the performance; missing real hits)."""
    env, env_fps = forced_align.onset_envelope(drum_path, max_seconds=args.max_seconds)
    if env.size == 0 or float(np.max(env)) <= 0.0:
        entry["status"] = "silent_stem"
        return
    floor = postfilter.support_floor_from_env(env, args.support_percentile)
    off, s0 = paradb.global_offset(gt, env, env_fps, floor, args.align_window, args.offset_window)
    apply_off = off if abs(off) > args.offset_correct_min else 0.0
    gt_off = paradb.shift_onsets(gt, apply_off)
    s_corr = clean.support_score(
        gt_off, env, env_fps, window_s=args.align_window, support_floor=floor,
    )["fraction"]
    conf_floor = postfilter.support_floor_from_env(env, args.recall_percentile)
    rec = clean.recall_score(
        gt_off, env, env_fps, confident_floor=conf_floor, window_s=args.recall_window,
        min_distance_s=args.recall_min_distance, prominence=(args.recall_prominence or None),
    )
    entry["duration_s"] = round(float(env.size / env_fps), 2)
    entry["offset_ms"] = round(off * 1000.0, 1)
    entry["support_0"] = round(float(s0), 4)
    entry["support_corr"] = round(float(s_corr), 4)
    entry["recall"] = round(float(rec["fraction"]), 4)
    entry["n_confident"] = rec["n_confident"]
    entry["keep"] = bool(s_corr >= args.keep_support and rec["fraction"] >= args.keep_recall)


def _histogram(vals: np.ndarray, label: str, log) -> None:
    log(f"\n  {label} over {len(vals)} scored maps "
        f"(median {np.median(vals):.3f}, mean {vals.mean():.3f}):")
    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(vals, bins=edges)
    peak = max(hist.max(), 1)
    for i in range(len(hist)):
        bar = "#" * int(round(40 * hist[i] / peak))
        log(f"    {edges[i]:.1f}-{edges[i+1]:.1f} | {hist[i]:5d} {bar}")


def _print_report(manifest: dict, keep_support: float, keep_recall: float, log) -> None:
    scored = [e for e in manifest.values() if "support_corr" in e]
    statuses: dict[str, int] = {}
    for e in manifest.values():
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    log(f"\n==== ParaDB manifest: {len(manifest)} maps ====")
    for st, n in sorted(statuses.items(), key=lambda kv: -kv[1]):
        log(f"  status {st:16s} {n:5d}")
    if not scored:
        return
    sup = np.array([e["support_corr"] for e in scored])
    rec = np.array([e.get("recall", 1.0) for e in scored])
    _histogram(sup, "support@corrected (precision)", log)
    _histogram(rec, "recall (confident audio onsets covered)", log)
    # combined keep grid: how many maps survive each (support, recall) cut pair
    log("\n  combined keep (support>=row, recall>=col):")
    cols = (0.4, 0.5, 0.6, 0.7)
    log("    sup\\rec |" + "".join(f"{c:>7.2f}" for c in cols))
    for srow in (0.5, 0.6, 0.7, 0.8):
        cells = "".join(f"{int(((sup >= srow) & (rec >= c)).sum()):>7d}" for c in cols)
        log(f"    {srow:>7.2f} |{cells}")
    log(f"\n  provisional keep (support>={keep_support:.2f} AND recall>={keep_recall:.2f}): "
        f"{sum(1 for e in scored if e.get('keep'))} maps")
    log("\n  sample LOW-recall maps (chart simpler than the audio):")
    for e in sorted(scored, key=lambda e: e.get("recall", 1.0))[:8]:
        log(f"    rec={e.get('recall',1.0):.2f} sup={e['support_corr']:.2f} "
            f"conf={e.get('n_confident',0):4d} n={e.get('n_onsets',0):5d}  {e['zip']}")
    log("  sample LOW-support maps (charted notes not in the audio):")
    for e in sorted(scored, key=lambda e: e["support_corr"])[:8]:
        log(f"    sup={e['support_corr']:.2f} rec={e.get('recall',1.0):.2f} "
            f"off={e.get('offset_ms',0):+.0f}ms n={e.get('n_onsets',0):5d}  {e['zip']}")


def _merge_and_report(work_dir: Path, out_path: Path, keep_support: float, keep_recall: float, log) -> dict:
    manifest = _merge_results(work_dir)
    out_path.write_text(json.dumps(manifest, indent=2))
    log(f"\nmerged {len(manifest)} results -> {out_path}")
    _print_report(manifest, keep_support, keep_recall, log)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="ParaDB corpus cull: distributed per-map chart support scoring")
    ap.add_argument("--maps-dir", required=True, help="folder of maps__*.zip ParaDB packs")
    ap.add_argument("--stems-cache", required=True,
                    help="dir for cached BS-Roformer drum stems (reused by separation)")
    ap.add_argument("--work-dir", default=None,
                    help="shared claims/+results/ dir for multi-runner sharding (default: <out-json parent>/_gate)")
    ap.add_argument("--out-json", required=True, help="merged manifest output (source of truth for the cull)")
    ap.add_argument("--limit", type=int, default=0, help="cap maps THIS runner claims (0=all; smoke)")
    ap.add_argument("--stale-minutes", type=float, default=60.0,
                    help="reclaim a claim lock older than this with no result (crashed runner)")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap per-song audio (debug)")
    ap.add_argument("--min-onsets", type=int, default=50, help="charts below this many onsets are trash")
    ap.add_argument("--keep-support", type=float, default=DEFAULT_KEEP_SUPPORT,
                    help="provisional support (precision) keep threshold (real cut chosen from histogram)")
    ap.add_argument("--keep-recall", type=float, default=0.5,
                    help="provisional recall keep threshold (real cut chosen from histogram)")
    ap.add_argument("--support-percentile", type=float, default=60.0, help="adaptive support floor percentile")
    ap.add_argument("--align-window", type=float, default=0.03, help="support window (s)")
    ap.add_argument("--offset-window", type=float, default=0.05, help="+/- median-offset search (s)")
    ap.add_argument("--offset-correct-min", type=float, default=0.025,
                    help="apply offset correction only if |median offset| exceeds this (s)")
    ap.add_argument("--drum-corr-threshold", type=float, default=0.5,
                    help="drum/song correlation above which the song already contains the drums")
    # recall gate: only EXTREMELY confident audio onsets (high percentile) count,
    # so we penalise a chart for missing OBVIOUS hits, not soft/ambiguous ones.
    ap.add_argument("--recall-percentile", type=float, default=92.0,
                    help="envelope percentile for 'confident' audio onsets (higher = stricter)")
    ap.add_argument("--recall-window", type=float, default=0.05,
                    help="+/- window for a chart onset to count as covering an audio onset (s)")
    ap.add_argument("--recall-min-distance", type=float, default=0.05,
                    help="min spacing between confident audio onsets (s)")
    ap.add_argument("--recall-prominence", type=float, default=0.0,
                    help="prominence for confident-onset peaks (0=off)")
    ap.add_argument("--scratch-dir", default=None,
                    help="dir for per-map temp work (zip extract + mix wav + the separator's raw "
                    "6-stem WAV dump). Point at a FAST local disk; on WSL the default /tmp can be a "
                    "slow DrvFs mount, making the per-clip stem write dominate on a fast GPU.")
    ap.add_argument("--merge", action="store_true", help="just rebuild the manifest from results/ and exit")
    ap.add_argument("--report-only", action="store_true", help="rebuild + print the report, no separation")
    ap.add_argument("--log", default=None, help="tee stdout+stderr to this file")
    args = ap.parse_args()

    runtime.tee_stdio(args.log)
    log = lambda s: print(s, flush=True)  # noqa: E731

    out_path = Path(args.out_json)
    work_dir = Path(args.work_dir) if args.work_dir else out_path.parent / "_gate"
    (work_dir / "claims").mkdir(parents=True, exist_ok=True)
    (work_dir / "results").mkdir(parents=True, exist_ok=True)
    if args.scratch_dir:
        Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)

    if args.merge or args.report_only:
        _merge_and_report(work_dir, out_path, args.keep_support, args.keep_recall, log)
        return

    stems_cache = Path(args.stems_cache)
    stems_cache.mkdir(parents=True, exist_ok=True)
    stale_s = args.stale_minutes * 60.0
    zips = paradb.iter_zips(args.maps_dir)
    n_done = len(list((work_dir / "results").glob("*.json")))
    log(f"[{_RUNNER}] {len(zips)} maps total; {n_done} already have results; work-dir={work_dir}")

    _run_pipeline(zips, work_dir, stems_cache, stale_s, args, log)
    _merge_and_report(work_dir, out_path, args.keep_support, args.keep_recall, log)


def _run_pipeline(zips, work_dir: Path, stems_cache: Path, stale_s: float, args, log) -> None:
    """3-stage GPU-fed pipeline (mirrors separate_egmd_dataset.py) so the GPU
    never idles on disk/IO: a producer thread claims+extracts+builds the next
    map's mix while the main thread separates the current one and a scorer thread
    scores+writes the previous one. Each map flows as a mutable `job` dict; every
    stage guards its own failures into the job's `entry` so one bad map can't kill
    the runner, and the scorer always writes a result + releases the claim."""
    import queue
    import shutil
    import threading

    from app.pipeline.separate import Separator

    ready_q: queue.Queue = queue.Queue(maxsize=3)   # prepped (CPU) -> GPU
    score_q: queue.Queue = queue.Queue(maxsize=3)   # separated -> scorer (CPU)
    counts = {"claimed": 0, "done": 0}

    def produce():
        claimed = 0
        for zp in zips:
            if args.limit and claimed >= args.limit:
                break
            map_id = paradb.map_id_of_zip(zp)
            if not _try_claim(work_dir, map_id, stale_s):
                continue  # done by, or in flight on, another runner
            claimed += 1
            td = Path(tempfile.mkdtemp(prefix="paradbgate_", dir=args.scratch_dir or None))
            job = {"map_id": map_id, "td": td, "gt": None, "mix": None,
                   "drum": None, "to_gpu": False, "publish": False,
                   "entry": {"zip": zp.name, "map_id": map_id, "status": "ok", "runner": _RUNNER}}
            try:
                entry, gt, chart = _extract_parse(zp, td, args)
                job["entry"] = entry
                job["gt"] = gt
                if gt is not None:  # passed sanity
                    drum_cached = stems_cache / f"{map_id}.drum.flac"
                    if drum_cached.exists():
                        job["drum"] = drum_cached  # already separated -> score from cache (no re-publish)
                    else:
                        mix = td / "_mix.wav"
                        ok, case = paradb.build_mix(
                            td, rlrr.song_tracks(chart), rlrr.drum_tracks(chart),
                            paradb.SEP_SR, mix, args.max_seconds, args.drum_corr_threshold,
                        )
                        entry["mix_case"] = case
                        if ok:
                            job["mix"] = mix
                            job["to_gpu"] = True
                        else:
                            entry["status"] = "no_audio"
            except Exception as exc:  # noqa: BLE001  corrupt zip / malformed chart
                job["entry"]["status"] = "error"
                job["entry"]["error"] = f"{type(exc).__name__}: {exc}"
            counts["claimed"] = claimed
            ready_q.put(job)
        ready_q.put(None)

    def score():
        while True:
            job = score_q.get()
            if job is None:
                break
            entry, map_id = job["entry"], job["map_id"]
            try:
                if entry["status"] == "ok" and job["drum"] is not None:
                    _score_from_stem(entry, job["gt"], job["drum"], args)
                    if entry["status"] == "ok":
                        entry["drum_stem"] = f"{map_id}.drum.flac"
                        if job["publish"]:  # freshly separated -> cache it (FLAC, off the GPU thread)
                            _publish_drum_flac(job["drum"], stems_cache / f"{map_id}.drum.flac")
            except Exception as exc:  # noqa: BLE001  odd audio must not kill the runner
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                _write_result(work_dir, map_id, entry)
                _release(work_dir, map_id)
                if job["td"] is not None:
                    shutil.rmtree(job["td"], ignore_errors=True)
            counts["done"] += 1
            s = entry.get("support_corr")
            r = entry.get("recall")
            log(f"[{_RUNNER}] ({counts['done']}) {map_id}  status={entry['status']}"
                + (f"  sup={s:.2f} rec={r:.2f} off={entry.get('offset_ms',0):+.0f}ms"
                   if s is not None else ""))

    sep = Separator()
    sep.load()
    producer = threading.Thread(target=produce, daemon=True)
    scorer = threading.Thread(target=score, daemon=True)
    producer.start()
    scorer.start()
    # main thread = GPU: pull prepped jobs, separate (only those needing it), hand off.
    while True:
        job = ready_q.get()
        if job is None:
            break
        if job["to_gpu"]:
            try:
                # GPU thread does ONLY separation; the drum stem stays on local
                # disk (job["td"]) and the scorer publishes it to the shared cache
                # off-thread, so the NFS write never blocks the next separation.
                job["drum"] = Path(sep.run_stems_all(job["mix"], job["td"], build_no_drums=False).drum_stem)
                job["publish"] = True
            except Exception as exc:  # noqa: BLE001
                job["entry"]["status"] = "sep_failed"
                job["entry"]["error"] = f"{type(exc).__name__}: {exc}"
        score_q.put(job)
    score_q.put(None)
    scorer.join()
    log(f"[{_RUNNER}] claimed {counts['claimed']} maps this run, {counts['done']} scored")


if __name__ == "__main__":
    main()
