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


def _score_map(zp: Path, sep, stems_cache: Path, args) -> dict:
    """Separate (cached) + score one map -> a manifest entry dict.

    Known failures (no chart, no audio, separation error, silent stem) are
    returned as a `status` rather than raised. Unexpected raises (corrupt zip,
    malformed chart) are caught by the caller, which records an `error` result so
    the corpus run is crash-proof and the bad map isn't re-claimed forever."""
    map_id = paradb.map_id_of_zip(zp)
    entry: dict = {"zip": zp.name, "map_id": map_id, "status": "ok", "runner": _RUNNER}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        chart = paradb.pick_chart(root)
        if chart is None:
            entry["status"] = "no_chart"
            return entry
        entry["chart"] = str(chart.relative_to(root))
        gt = rlrr.onsets_by_lane(chart)
        per_lane_n = {ln: len(ts) for ln, ts in gt.items() if ts}
        n_onsets = sum(per_lane_n.values())
        entry["n_onsets"] = n_onsets
        entry["per_lane_n"] = per_lane_n
        if n_onsets < args.min_onsets:
            entry["status"] = "too_few_onsets"
            return entry

        # BS-Roformer drum stem (cached -> reused by separate_paradb_dataset.py).
        drum_cached = stems_cache / f"{map_id}.drum.flac"
        if not drum_cached.exists():
            mix_wav = root / "_mix.wav"
            ok, case = paradb.build_mix(
                root, rlrr.song_tracks(chart), rlrr.drum_tracks(chart),
                paradb.SEP_SR, mix_wav, args.max_seconds, args.drum_corr_threshold,
            )
            entry["mix_case"] = case
            if not ok:
                entry["status"] = "no_audio"
                return entry
            try:
                drum = sep.run_stems_all(mix_wav, root, build_no_drums=False).drum_stem
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "sep_failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                return entry
            drum_cached.parent.mkdir(parents=True, exist_ok=True)
            # atomic publish so a concurrent runner never reads a half-written stem
            tmp_stem = drum_cached.with_suffix(f".{os.getpid()}.tmp")
            tmp_stem.write_bytes(Path(drum).read_bytes())
            os.replace(tmp_stem, drum_cached)
        entry["drum_stem"] = drum_cached.name

        # support: fraction of charted onsets on a real drum transient, at offset
        # 0 and after the robust global offset correction (a globally-shifted but
        # otherwise-good chart is still good once corrected).
        env, env_fps = forced_align.onset_envelope(drum_cached, max_seconds=args.max_seconds)
        if env.size == 0 or float(np.max(env)) <= 0.0:
            entry["status"] = "silent_stem"
            return entry
        floor = postfilter.support_floor_from_env(env, args.support_percentile)
        off, s0 = paradb.global_offset(gt, env, env_fps, floor, args.align_window, args.offset_window)
        apply_off = off if abs(off) > args.offset_correct_min else 0.0
        s_corr = clean.support_score(
            paradb.shift_onsets(gt, apply_off), env, env_fps,
            window_s=args.align_window, support_floor=floor,
        )["fraction"]
        entry["duration_s"] = round(float(env.size / env_fps), 2)
        entry["offset_ms"] = round(off * 1000.0, 1)
        entry["support_0"] = round(float(s0), 4)
        entry["support_corr"] = round(float(s_corr), 4)
        entry["keep"] = bool(s_corr >= args.keep_support)
    return entry


def _print_report(manifest: dict, keep_support: float, log) -> None:
    scored = [e for e in manifest.values() if "support_corr" in e]
    statuses: dict[str, int] = {}
    for e in manifest.values():
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    log(f"\n==== ParaDB manifest: {len(manifest)} maps ====")
    for st, n in sorted(statuses.items(), key=lambda kv: -kv[1]):
        log(f"  status {st:16s} {n:5d}")
    if not scored:
        return
    sc = np.array([e["support_corr"] for e in scored])
    log(f"\n  support@corrected over {len(sc)} scored maps "
        f"(median {np.median(sc):.3f}, mean {sc.mean():.3f}):")
    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(sc, bins=edges)
    peak = max(hist.max(), 1)
    for i in range(len(hist)):
        bar = "#" * int(round(40 * hist[i] / peak))
        log(f"    {edges[i]:.1f}-{edges[i+1]:.1f} | {hist[i]:5d} {bar}")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        log(f"  keep >= {thr:.2f}: {int((sc >= thr).sum()):5d} maps")
    log(f"\n  provisional keep (>= {keep_support:.2f}): "
        f"{sum(1 for e in scored if e.get('keep'))} maps")
    by_supp = sorted(scored, key=lambda e: e["support_corr"])
    log("\n  sample LOW-support maps (likely trash):")
    for e in by_supp[:8]:
        log(f"    {e['support_corr']:.2f}  off={e.get('offset_ms',0):+.0f}ms  "
            f"n={e.get('n_onsets',0):5d}  {e['zip']}")
    log("  sample HIGH-support maps (likely good):")
    for e in by_supp[-8:]:
        log(f"    {e['support_corr']:.2f}  off={e.get('offset_ms',0):+.0f}ms  "
            f"n={e.get('n_onsets',0):5d}  {e['zip']}")


def _merge_and_report(work_dir: Path, out_path: Path, keep_support: float, log) -> dict:
    manifest = _merge_results(work_dir)
    out_path.write_text(json.dumps(manifest, indent=2))
    log(f"\nmerged {len(manifest)} results -> {out_path}")
    _print_report(manifest, keep_support, log)
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
                    help="provisional keep threshold written to the manifest (real cut chosen from histogram)")
    ap.add_argument("--support-percentile", type=float, default=60.0, help="adaptive support floor percentile")
    ap.add_argument("--align-window", type=float, default=0.03, help="support window (s)")
    ap.add_argument("--offset-window", type=float, default=0.05, help="+/- median-offset search (s)")
    ap.add_argument("--offset-correct-min", type=float, default=0.025,
                    help="apply offset correction only if |median offset| exceeds this (s)")
    ap.add_argument("--drum-corr-threshold", type=float, default=0.5,
                    help="drum/song correlation above which the song already contains the drums")
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

    if args.merge or args.report_only:
        _merge_and_report(work_dir, out_path, args.keep_support, log)
        return

    stems_cache = Path(args.stems_cache)
    stems_cache.mkdir(parents=True, exist_ok=True)
    stale_s = args.stale_minutes * 60.0
    zips = paradb.iter_zips(args.maps_dir)
    n_done = len(list((work_dir / "results").glob("*.json")))
    log(f"[{_RUNNER}] {len(zips)} maps total; {n_done} already have results; "
        f"work-dir={work_dir}")

    from app.pipeline.separate import Separator

    sep = Separator()
    sep.load()
    claimed = 0
    for zp in zips:
        if args.limit and claimed >= args.limit:
            break
        map_id = paradb.map_id_of_zip(zp)
        if not _try_claim(work_dir, map_id, stale_s):
            continue  # done by, or in flight on, another runner
        claimed += 1
        try:
            try:
                entry = _score_map(zp, sep, stems_cache, args)
            except Exception as exc:  # noqa: BLE001  one bad map must never kill the runner
                # _score_map guards known failures; this catches the unexpected
                # (BadZipFile, malformed chart, odd audio). Write an `error`
                # result so the map is marked DONE, not re-claimed forever.
                entry = {"zip": zp.name, "map_id": map_id, "status": "error",
                         "error": f"{type(exc).__name__}: {exc}", "runner": _RUNNER}
            _write_result(work_dir, map_id, entry)
        finally:
            _release(work_dir, map_id)
        s = entry.get("support_corr")
        log(f"[{_RUNNER}] ({claimed}) {map_id}  status={entry['status']}"
            + (f"  support={s:.2f} off={entry.get('offset_ms',0):+.0f}ms n={entry.get('n_onsets',0)}"
               if s is not None else ""))

    log(f"[{_RUNNER}] claimed {claimed} maps this run")
    _merge_and_report(work_dir, out_path, args.keep_support, log)


if __name__ == "__main__":
    main()
