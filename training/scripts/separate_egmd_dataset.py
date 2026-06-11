"""Separation-AWARE E-GMD training data: run our drum separator over E-GMD's
drum-module audio so the training audio carries the SAME artifacts (bleed,
residual, smearing) as the real-separator output we feed at inference, keeping
E-GMD's exact MIDI labels.

E-GMD is drums-only (Roland TD-17 module audio), so there's nothing to mix in
first -- the audio goes straight to BS-Roformer -> MDX23C, like ENST's
drums-only takes. Three things make E-GMD special and shape this script:

1. BALANCED-FIRST. E-GMD is 444 h, dominated by kick/snare grooves. Clips are
   ORDERED by greedy marginal rare-lane coverage (same algorithm as
   extract_star_balanced.py), so the rarest lanes (ride/crash/misc-cymbal, the
   hat articulations, toms) are pulled in FIRST. By default the whole dataset is
   separated (train --train-min 0 = everything) in that priority order, so a
   ctrl-C at any point still leaves the most training-valuable clips done; a
   duration cap (--train-min/--val-min minutes) just truncates the same order.

2. RESUMABLE. Two persisted states: a lane-count cache (`_lane_counts.json`, so
   the 45k MIDIs are scanned once) and per-clip output existence. A rerun
   re-derives the same balanced order, skips clips already on disk, and
   processes the rest -- so you can stop the job (e.g. to use the GPU at night)
   and pick up exactly where it left off.

3. BATCHED. E-GMD clips are tiny (median 3.5 s); separating each one (padded to
   30 s for the model) wastes ~15x the GPU. Instead we pack many clips into one
   ~5-min buffer (silence gaps between them), separate ONCE, and slice each
   clip's stems back out by time.

Output (flat, STAR/ENST-consistent), so `--dataset egmd`/`egmd_perstem` consume
it via egmd.read_index / egmd.perstem_index:
  annotation/<uid>.midi                 # copied label
  audio/sep_drum/<uid>.flac             # BS-Roformer drum stem
  audio/perstem/<pitch>/<uid>.flac      # MDX23C 5-class (pitch in k/s/h/c/t)
  e-gmd-v1.0.0.csv                       # rebuilt each run from completed clips
where uid = the source audio path with "/" -> "__" (unique per clip).

Run in the CUDA sandbox (transcriber `app` separator + GPU + MODELS_DIR).
PYTHONPATH must include the dsp + training packages.

Usage: separate_egmd_dataset.py <egmd_root> <out_dir>
         [--train-min N] [--val-min N] [--buffer-sec N] [--gap-sec N]
"""
import argparse
import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Quiet the chatty separator stack for batch runs: WARNING+ only from
# audio-separator, and no tqdm chunk bars. Set before importing the app/lib.
os.environ.setdefault("DRUMJOT_SEP_LOG_LEVEL", "WARNING")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "transcriber"))  # transcriber app

from drumjot_training import egmd, midi_labels  # noqa: E402
from drumjot_training.lanes import LANES  # noqa: E402

PITCHES = tuple(egmd.PERSTEM_TO_LANES)  # ("k", "s", "h", "c", "t")
MIN_SEP_SECONDS = 30.0  # pad a buffer shorter than this (only the last small batch)


# --- pure helpers (unit-testable, no audio/GPU) ----------------------------


def _fmt_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def uid_for(audio_filename: str) -> str:
    """Source rel path -> unique flat id: drummer1/eval_session/1_x.wav -> drummer1__eval_session__1_x."""
    return "__".join(Path(audio_filename).with_suffix("").parts)


def greedy_select(counts: np.ndarray, durations: np.ndarray, max_seconds: float) -> list[int]:
    """Indices selected by greedy marginal rare-lane coverage, capped by total
    duration. `counts` is (n_clips, n_lanes); each step picks the clip with the
    highest `sum_lane count/(1+covered)` so under-covered (rare) lanes pull their
    clips in first. Deterministic (argmax ties -> lowest index), and
    prefix-stable, so a larger cap just extends the same order."""
    n = counts.shape[0]
    if n == 0:
        return []
    cf = counts.astype(np.float64)
    covered = np.zeros(counts.shape[1], dtype=np.float64)
    avail = np.ones(n, dtype=bool)
    order: list[int] = []
    total = 0.0
    while total < max_seconds and avail.any():
        scores = (cf / (1.0 + covered)).sum(axis=1)
        scores[~avail] = -np.inf
        i = int(np.argmax(scores))
        avail[i] = False
        order.append(i)
        covered += cf[i]
        total += float(durations[i])
    return order


def plan_batches(durations: list[float], buffer_sec: float, gap_sec: float) -> list[list[int]]:
    """Group consecutive clip indices (priority order preserved) into batches
    whose buffered length (sum durations + inter-clip gaps) stays near
    `buffer_sec`. A clip longer than `buffer_sec` becomes its own batch."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0.0
    for i, d in enumerate(durations):
        add = d + (gap_sec if cur else 0.0)
        if cur and cur_len + add > buffer_sec:
            batches.append(cur)
            cur, cur_len = [], 0.0
            add = d
        cur.append(i)
        cur_len += add
    if cur:
        batches.append(cur)
    return batches


def slice_seconds(y: np.ndarray, sr: int, start_s: float, len_s: float) -> np.ndarray:
    """Slice [start_s, start_s+len_s) from `y` (samples, ...) at rate `sr`."""
    a = int(round(start_s * sr))
    b = a + int(round(len_s * sr))
    return y[a:b]


# --- selection (balanced + cached counts) ----------------------------------


def load_lane_counts(rows, root: Path, cache_path: Path, log) -> dict:
    """{audio_filename: {"counts": {lane:int}, "dur": float}} for every row,
    reading each clip's MIDI once and caching to `cache_path` (so reruns and an
    interrupted scan both resume)."""
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    out: dict = {}
    scanned = 0
    for i, r in enumerate(rows):
        key = r["audio_filename"]
        if key in cache:
            out[key] = cache[key]
            continue
        try:
            onsets = midi_labels.onsets_from_path(root / r["midi_filename"])
            counts = {ln: len(onsets[ln]) for ln in LANES}
        except Exception as e:  # noqa: BLE001
            log(f"  skip count {key}: {e!r}")
            counts = dict.fromkeys(LANES, 0)
        out[key] = cache[key] = {"counts": counts, "dur": float(r["duration"])}
        scanned += 1
        if scanned % 2000 == 0:
            _atomic_bytes(json.dumps(cache).encode(), cache_path)  # atomic so an interrupted scan resumes
            log(f"  scanned {scanned} new MIDIs ({i + 1}/{len(rows)})")
    _atomic_bytes(json.dumps(cache).encode(), cache_path)
    return out


def select(rows, counts_by_key, max_seconds: float) -> list[dict]:
    """Balanced, duration-capped selection of `rows` (CSV dicts), in priority
    order."""
    if not rows:
        return []
    counts = np.array([[counts_by_key[r["audio_filename"]]["counts"][ln] for ln in LANES] for r in rows])
    durs = np.array([counts_by_key[r["audio_filename"]]["dur"] for r in rows])
    return [rows[i] for i in greedy_select(counts, durs, max_seconds)]


# --- output paths + completion ---------------------------------------------


def out_paths(out: Path, uid: str):
    return {
        "midi": out / "annotation" / f"{uid}.midi",
        "drum": out / "audio" / "sep_drum" / f"{uid}.flac",
        "per": {p: out / "audio" / "perstem" / p / f"{uid}.flac" for p in PITCHES},
    }


def is_done(paths) -> bool:
    return paths["drum"].exists() and paths["midi"].exists() and all(p.exists() for p in paths["per"].values())


def _write_flac(y, sr, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, format="FLAC")


def _atomic_bytes(data: bytes, dst: Path) -> None:
    """Write `data` to `dst` atomically (tmp + os.replace), so a ctrl-C mid-write
    can never leave a half-written file that resume/the loader would trust."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


# --- batched separation -----------------------------------------------------


# The batch lifecycle is split into three stages so a prefetch pipeline can run
# them on different threads and keep the GPU fed (the GPU otherwise idles through
# the NFS-bound build + writeback, the big flat gaps in `nvidia-smi`):
#   build_buffer  -- CPU/NFS: read clips, concatenate into one buffer  (producer)
#   gpu_separate  -- GPU: roformer -> mdx23c, read stems back into RAM (main)
#   write_outputs -- CPU/NFS: slice + write per-clip stems + midi      (writer)
# build_buffer returns the clips it actually packed (valid_batch), aligned 1:1
# with `offsets`, so a dropped clip (sr mismatch) can't desync the writeback.


def build_buffer(batch, gap_sec: float, log):
    """Read + concatenate `batch` clips (silence gaps between them) into one
    buffer. Returns (valid_batch, buf, sr, offsets) or None if nothing packed.
    `batch` entries: dicts with uid, src_audio (abs), midi (abs)."""
    pieces: list[np.ndarray] = []
    offsets: list[tuple[float, float]] = []  # (start_s, len_s) per kept clip
    valid: list[dict] = []                   # clips actually packed, aligned to offsets
    sr = None
    pos = 0.0
    for e in batch:
        y, s = sf.read(str(e["src_audio"]), always_2d=True)
        if sr is None:
            sr = s
        elif s != sr:
            log(f"  skip {e['uid']}: sr {s} != batch sr {sr}")
            continue
        if pieces:
            pieces.append(np.zeros((int(gap_sec * sr), y.shape[1]), dtype=np.float32))
            pos += gap_sec
        offsets.append((pos, y.shape[0] / sr))
        pieces.append(y.astype(np.float32))
        pos += y.shape[0] / sr
        valid.append(e)
    if not pieces or sr is None:
        return None
    buf = np.concatenate(pieces, axis=0)
    need = int(MIN_SEP_SECONDS * sr)
    if buf.shape[0] < need:  # pad only an undersized final batch
        buf = np.concatenate([buf, np.zeros((need - buf.shape[0], buf.shape[1]), dtype=buf.dtype)], axis=0)
    return valid, buf, sr, offsets


def gpu_separate(sep, buf: np.ndarray, sr: int, n_clips: int, log):
    """Separate one buffer: roformer -> mdx23c. Reads every stem back INTO RAM
    (dy, pys) before the temp dir is torn down, so the writer needs no temp
    files. Returns (dy, dsr, pys) where pys = {pitch: (array, sr)}."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        mixd = tdp / "_mixin"
        mixd.mkdir()
        mix = mixd / "mix.wav"  # NEUTRAL name (separator picks drum stem by "drum" in filename)
        sf.write(str(mix), buf, sr)
        drum = sep.run_stems_all(mix, tdp / "s1", build_no_drums=False).drum_stem  # separate work dirs
        log(f"  roformer OK ({n_clips} clips); splitting into 5 stems...")
        per = sep.run_stems_per(drum, tdp / "s2").per_instrument
        log("  mdx23c per-stem OK")
        dy, dsr = sf.read(str(drum), always_2d=True)
        # The separator skips writing empty/near-silent stems (e.g. kick on a
        # cymbal-heavy buffer), so a pitch may be missing or its file unreadable;
        # load defensively and silence-fill in write_outputs so every clip gets 5.
        pys: dict = {}
        for p, src in per.items():
            try:
                arr, ss = sf.read(str(src), always_2d=True)
                if arr.size:
                    pys[p] = (arr, ss)
            except Exception:  # noqa: BLE001  (missing/empty/corrupt stem)
                pass
    return dy, dsr, pys


def write_outputs(valid_batch, offsets, dy, dsr, pys, out: Path) -> int:
    """Slice each clip's stems out of the separated buffer and write them, midi
    last (the atomic completion marker). `valid_batch` is aligned 1:1 to `offsets`."""
    done = 0
    for k, e in enumerate(valid_batch):
        start_s, len_s = offsets[k]
        paths = out_paths(out, e["uid"])
        _write_flac(slice_seconds(dy, dsr, start_s, len_s), dsr, paths["drum"])
        for p in PITCHES:
            if p in pys:
                yy, ss = pys[p]
                _write_flac(slice_seconds(yy, ss, start_s, len_s), ss, paths["per"][p])
            else:  # instrument absent in this buffer -> a silent stem (accurate)
                _write_flac(np.zeros((int(round(len_s * dsr)), 1), dtype="float32"), dsr, paths["per"][p])
        # midi written LAST and atomically = the completion marker: is_done()
        # is true only once all stems exist AND this rename has landed.
        _atomic_bytes(Path(e["midi"]).read_bytes(), paths["midi"])
        done += 1
    return done


# --- CSV (rebuilt from completed clips) -------------------------------------


def write_csv(out: Path, selection: list[dict]) -> int:
    """(Re)write the sep-tree CSV from selected clips whose outputs are on disk,
    so `egmd.read_index` / `--dataset egmd` see exactly the completed set."""
    fields = [
        "drummer", "session", "id", "style", "bpm", "beat_type", "time_signature",
        "duration", "split", "midi_filename", "audio_filename", "kit_name",
    ]
    rows_out = []
    for e in selection:
        if not is_done(out_paths(out, e["uid"])):
            continue
        r = dict(e["row"])
        r["split"] = e["split"]
        r["midi_filename"] = f"annotation/{e['uid']}.midi"
        r["audio_filename"] = f"audio/sep_drum/{e['uid']}.flac"
        rows_out.append({k: r.get(k, "") for k in fields})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)
    _atomic_bytes(buf.getvalue().encode(), out / "e-gmd-v1.0.0.csv")  # atomic: loader never sees a half-write
    return len(rows_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("egmd_root", type=Path, help="extracted E-GMD root (with e-gmd-v1.0.0.csv)")
    ap.add_argument("out_dir", type=Path, help="output dir for the separation-aware tree")
    ap.add_argument("--train-min", type=float, default=0.0,
                    help="balanced train minutes to select; 0 (default) = EVERYTHING, in balanced "
                    "priority order, so a ctrl-C/resume always has the most valuable clips done first")
    ap.add_argument("--val-min", type=float, default=30.0,
                    help="balanced val minutes from validation+test (0 = everything); kept modest by "
                    "default since eval rarely needs all ~10k held-out clips")
    ap.add_argument("--buffer-sec", type=float, default=300.0, help="separation buffer length per batch")
    ap.add_argument("--gap-sec", type=float, default=2.0, help="silence gap between concatenated clips")
    args = ap.parse_args()

    root, out = args.egmd_root, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731

    with open(root / "e-gmd-v1.0.0.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    log(f"E-GMD: {len(rows)} clips")
    counts_by_key = load_lane_counts(rows, root, out / "_lane_counts.json", log)

    train_rows = [r for r in rows if r["split"].strip() == "train"]
    val_rows = [r for r in rows if r["split"].strip() in ("validation", "test")]
    train_cap = float("inf") if args.train_min <= 0 else args.train_min * 60
    val_cap = float("inf") if args.val_min <= 0 else args.val_min * 60
    train_sel = select(train_rows, counts_by_key, train_cap)
    val_sel = select(val_rows, counts_by_key, val_cap)
    log(f"selected (balanced): {len(train_sel)} train, {len(val_sel)} val clips")

    # val FIRST so an early ctrl-C/resume always leaves a complete eval set, then
    # train grinds in balanced priority order (most valuable clips first). val is
    # relabelled "validation" for the loader.
    selection = (
        [{"row": r, "split": "validation"} for r in val_sel]
        + [{"row": r, "split": "train"} for r in train_sel]
    )
    for e in selection:
        e["uid"] = uid_for(e["row"]["audio_filename"])
        e["src_audio"] = root / e["row"]["audio_filename"]
        e["midi"] = root / e["row"]["midi_filename"]

    pending = [e for e in selection if not is_done(out_paths(out, e["uid"]))]
    log(f"{len(selection) - len(pending)} already done, {len(pending)} pending")
    if not pending:
        n = write_csv(out, selection)
        log(f"nothing to do; CSV has {n} clips -> {out}")
        return

    from app.pipeline.separate import Separator

    sep = Separator()
    sep.load()

    batches = plan_batches([float(e["row"]["duration"]) for e in pending],
                           args.buffer_sec, args.gap_sec)
    nb = len(batches)
    batch_clips = [[pending[i] for i in idxs] for idxs in batches]
    log(f"{len(pending)} clips in {nb} batches (~{args.buffer_sec:.0f}s buffers); pipelined "
        f"(build || GPU || write) to keep the GPU fed")

    # Prefetch pipeline: a producer thread builds the next buffer and a writer
    # thread writes the previous batch's stems WHILE the GPU separates the
    # current one, so the GPU isn't stalled on NFS reads/writes between batches.
    # Only the main thread touches the GPU/`sep` (single CUDA context); the other
    # two stages are pure CPU/IO (soundfile + numpy release the GIL, so they
    # genuinely overlap). Bounded queues cap how many buffers/stem-sets sit in RAM.
    import queue
    import threading

    ready_q: queue.Queue = queue.Queue(maxsize=2)  # built input buffers (producer -> GPU)
    write_q: queue.Queue = queue.Queue(maxsize=1)  # separated stems in RAM (GPU -> writer)
    stats = {"done": 0}

    def produce():
        for bi, batch in enumerate(batch_clips, 1):
            try:
                built = build_buffer(batch, args.gap_sec, log)
            except Exception as e:  # noqa: BLE001  bad clip -> skip batch, retried on resume
                log(f"  !! batch {bi} build FAILED: {e!r} (its clips retried on resume)")
                built = None
            ready_q.put((bi, built))
        ready_q.put(None)  # sentinel

    def consume_writes():
        t0 = time.perf_counter()
        while True:
            item = write_q.get()
            if item is None:
                break
            bi, res = item
            valid_batch, offsets, dy, dsr, pys = res
            try:
                stats["done"] += write_outputs(valid_batch, offsets, dy, dsr, pys, out)
            except Exception as e:  # noqa: BLE001  partial writes lack the midi marker -> retried
                log(f"  !! batch {bi} write FAILED: {e!r} (its clips retried on resume)")
            write_csv(out, selection)  # refresh after each batch so an interrupt leaves a usable CSV
            d = stats["done"]
            eta = _fmt_eta((time.perf_counter() - t0) / d * (len(pending) - d)) if d else "?"
            log(f"[{d}/{len(pending)} clips] batch {bi}/{nb} written  eta {eta}")

    producer = threading.Thread(target=produce, daemon=True)
    writer = threading.Thread(target=consume_writes, daemon=True)
    producer.start()
    writer.start()

    while True:
        item = ready_q.get()
        if item is None:
            break
        bi, built = item
        if built is None:
            continue  # build failed/empty
        valid_batch, buf, sr, offsets = built
        log(f"[{stats['done']}/{len(pending)} clips] batch {bi}/{nb} ({len(valid_batch)} clips) separating...")
        try:
            dy, dsr, pys = gpu_separate(sep, buf, sr, len(valid_batch), log)
            write_q.put((bi, (valid_batch, offsets, dy, dsr, pys)))
        except Exception as e:  # noqa: BLE001  keep the unattended run alive; clips retried on resume
            log(f"  !! batch {bi} GPU FAILED: {e!r} (its clips retried on resume)")
    write_q.put(None)  # sentinel after the last GPU result
    writer.join()

    n = write_csv(out, selection)
    log(f"DONE. {stats['done']} clips this run; CSV has {n} completed -> {out}")


if __name__ == "__main__":
    main()
