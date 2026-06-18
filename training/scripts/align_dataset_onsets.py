"""Dataset-wide onset SNAP + FILTER: align every label to its stem audio.

Validated on 100 crash snippets (OVERNIGHT_LOG 2026-06-19): labels are frequently
mistimed vs the separated stem, and a chunk are outright false (88% of the false
crash labels coincide with ANOTHER instrument = lane-mislabels). This applies the
same audio-referenced snap+filter to EVERY lane of EVERY per-instrument stem in
all three sep datasets, so training can run on aligned/cleaned targets.

Per stem, per lane (the lanes that belong to the stem's pitch):
  - build the onset-strength envelope of the STEM (forced_align.onset_envelope);
  - find the real audio onsets (cymbal_snap_redraw.real_onset_times);
  - SNAP each label to the nearest real onset within the lane's window, else
    DISCARD it (classified ghost vs wrong-lane via other-lane coincidence).

NON-DESTRUCTIVE + reversible: writes a NEW `_onsets_aligned.json` (keyed by stem
audio path -> {lane: [aligned times]}); originals are untouched. Training opts in
via a flag (default off). Resumable: skips stems already in the output.

  DRUMJOT_STAR=/codebox-workspace/datasets/star_balanced_sep \
  DRUMJOT_ENST=/codebox-workspace/datasets/enst-sep \
  DRUMJOT_EGMD=/codebox-workspace/datasets/egmd_sep \
  OMP_NUM_THREADS=8 python training/scripts/align_dataset_onsets.py \
      --out /codebox-workspace/datasets/_onsets_aligned.json

Validate the per-source iteration wiring without audio:  python ... --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # cymbal_snap_redraw
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dsp"))  # dsp/

# Per-lane snap windows (s). cym validated at 0.12; tighter on denser lanes so a
# good label isn't yanked onto a neighbour. Unknown lanes fall back to 0.05.
SNAP_WINDOW = {"k": 0.05, "s": 0.05, "ss": 0.05, "t": 0.05,
               "hc": 0.06, "hp": 0.06, "ho": 0.07, "rd": 0.10, "cr": 0.12,
               "mc": 0.12, "mp": 0.05}
DEFAULT_WINDOW = 0.05


def iter_source(name, log):
    """Yield (audio_path:str, pitch, lanes, full_onsets) for every perstem clip of
    a source. full_onsets (all lanes, memoized per annotation) drives wrong-lane
    classification; lanes = the stem's own lanes (what we snap)."""
    from drumjot_training import egmd, enst, midi_labels, paths, star

    cfg = {
        "star": (star.perstem_index, star.PERSTEM_TO_LANES,
                 lambda c: c.annotation_path, star.onsets_by_lane),
        "enst": (enst.perstem_index, enst.PERSTEM_TO_LANES,
                 lambda c: c.annotation_path, enst.onsets_by_lane),
        "egmd": (egmd.perstem_index, egmd.PERSTEM_TO_LANES,
                 lambda c: c.midi_path, midi_labels.onsets_from_path),
    }[name]
    index_fn, p2l, label_path, reader = cfg
    root = paths.dataset_path(name)
    clips = index_fn(root)
    log(f"  {name}: {len(clips)} perstem clips under {root}")
    cache: dict[str, dict] = {}
    for c in clips:
        lp = str(label_path(c))
        full = cache.get(lp)
        if full is None:
            full = {ln: list(v) for ln, v in reader(lp).items()}
            cache[lp] = full
        yield str(c.audio_path), c.pitch, p2l.get(c.pitch, ()), full


def align_stem(audio_path, lanes, full, log, *, no_filter=False):
    """Snap (+optionally filter) a stem's lanes against its audio. Returns (aligned
    dict, stats), stats = {lane: [n_in, n_snap, n_ghost, n_wrong]}. With `no_filter`
    (snap-only) a would-be discard is KEPT at its original time instead of dropped --
    the conservative variant that aligns without risking real soft hits."""
    import librosa
    from cymbal_snap_redraw import align_or_discard, discard_reason, real_onset_times

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if y.size == 0:
        return {}, {}
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=64)
    reals = real_onset_times(env, sr / 64)
    aligned, stats = {}, {}
    for lane in lanes:
        onsets = sorted(float(x) for x in full.get(lane, []))
        n_snap = n_ghost = n_wrong = 0
        kept = []
        for kind, ot, nt in align_or_discard(onsets, reals, SNAP_WINDOW.get(lane, DEFAULT_WINDOW)):
            if kind == "snap":
                kept.append(round(nt, 4))
                n_snap += 1
            else:
                if no_filter:
                    kept.append(round(ot, 4))  # snap-only: keep the (unsnapped) original
                if discard_reason(ot, full, exclude=lane).startswith("wrong"):
                    n_wrong += 1
                else:
                    n_ghost += 1
        aligned[lane] = sorted(kept)
        stats[lane] = [len(onsets), n_snap, n_ghost, n_wrong]
    return aligned, stats


def _accum(total, stats):
    for lane, s in stats.items():
        t = total.setdefault(lane, [0, 0, 0, 0])
        for i in range(4):
            t[i] += s[i]


def _report(total, log):
    log("\n==== alignment stats per lane (over all stems) ====")
    log(f"  {'lane':5s} {'labels':>8s} {'snap':>8s} {'ghost':>7s} {'wrong':>7s} "
        f"{'%snap':>6s} {'%disc':>6s}")
    grand = [0, 0, 0, 0]
    for lane in sorted(total):
        n, sn, g, w = total[lane]
        if not n:
            continue
        for i, v in enumerate((n, sn, g, w)):
            grand[i] += v
        log(f"  {lane:5s} {n:8d} {sn:8d} {g:7d} {w:7d} {sn / n:6.1%} {(g + w) / n:6.1%}")
    n, sn, g, w = grand
    if n:
        log(f"  {'ALL':5s} {n:8d} {sn:8d} {g:7d} {w:7d} {sn / n:6.1%} {(g + w) / n:6.1%}")
        log(f"  => discards: {g} ghost / {w} wrong-lane "
            f"({w / (g + w):.0%} of discards are wrong-lane)" if (g + w) else "")


def _selftest():
    # window table covers every lane the per-stem maps reference
    from drumjot_training import egmd, enst, star
    lanes = set()
    for p2l in (star.PERSTEM_TO_LANES, enst.PERSTEM_TO_LANES, egmd.PERSTEM_TO_LANES):
        for v in p2l.values():
            lanes.update(v)
    missing = [ln for ln in lanes if ln not in SNAP_WINDOW]
    assert not missing, f"no snap window for lanes: {missing}"
    print(f"SELFTEST OK (snap windows cover all perstem lanes: {sorted(lanes)})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Dataset-wide onset snap+filter")
    ap.add_argument("--sources", default="star,enst,egmd")
    ap.add_argument("--out", default="/codebox-workspace/datasets/_onsets_aligned.json")
    ap.add_argument("--stats-out", default="/codebox-workspace/datasets/_onsets_aligned_stats.json")
    ap.add_argument("--flush-every", type=int, default=200, help="incremental write cadence (resume)")
    ap.add_argument("--max-stems", type=int, default=0, help="cap stems processed (0=all; dry run)")
    ap.add_argument("--no-filter", action="store_true",
                    help="snap-only: keep would-be discards at their original time (conservative; "
                    "aligns without risking real soft hits on snare/ride/tom)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    from drumjot_training import runtime
    runtime.tee_stdio(Path(args.out).with_suffix(".log"))
    log = lambda s: print(s, flush=True)  # noqa: E731
    out_path = Path(args.out)
    aligned = json.loads(out_path.read_text()) if out_path.exists() else {}
    log(f"=== dataset onset alignment: sources={args.sources} (resume: {len(aligned)} stems done) ===")

    total: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    done = 0
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        for audio_path, _pitch, lanes, full in iter_source(name, log):
            if audio_path in aligned:
                continue
            try:
                a, stats = align_stem(audio_path, lanes, full, log, no_filter=args.no_filter)
            except Exception as e:  # noqa: BLE001
                log(f"  skip {Path(audio_path).name}: {e!r}")
                continue
            aligned[audio_path] = a
            _accum(total, stats)
            done += 1
            if done % args.flush_every == 0:
                out_path.write_text(json.dumps(aligned))
                log(f"  {done} stems aligned ({time.perf_counter() - t0:.0f}s); flushed")
            if args.max_stems and done >= args.max_stems:
                break
        if args.max_stems and done >= args.max_stems:
            break
    out_path.write_text(json.dumps(aligned))
    Path(args.stats_out).write_text(json.dumps(total, indent=2))
    log(f"\naligned {done} new stems ({len(aligned)} total) -> {out_path}")
    _report(total, log)


if __name__ == "__main__":
    main()
