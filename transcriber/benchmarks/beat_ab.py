"""Deterministic A/B between beat-tracking front-ends.

Compares **madmom** (`RNNDownBeatProcessor`) and **Beat Transformer** (both
through the shared production grid: madmom DBN -> gated
`align_beats_to_onsets` -> `_finalize_bar_tempos`) against **Beat This!**
(ISMIR 2024, DBN-free / meter-agnostic, scored on its native output with
the same onset-offset alignment), versus E-GMD MIDI ground truth.

Drumjot-relevant verdict metrics: **downbeat_f** (bar alignment) +
**bar_len_ok** (right beats-per-bar so bars don't drift) + octave-safe
tempo. beat_f / amlt are diagnostics only, denominator and 3/4-vs-6/8
feel don't affect note placement (see the `beat-tracker-eval-priority`
memory).

See `docs/superpowers/specs/2026-06-30-beat-tracker-ab-design.md`.

Run (CPU, synthetic onsets, while the GPU is busy), from `transcriber/`:

    .venv/bin/python -m benchmarks.beat_ab --onsets synthetic

`--onsets adtof` is the truthful production path but needs the GPU. The
Beat This! arm needs `beat-this` installed in the venv (eval-only dep;
not in pyproject, `uv pip install beat_this`).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import os
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .beat_gt import GtGrid, LaneOnset, gt_grid, lane_onsets, parse_time_sig, sanity_coverage
from .onset_synth import gt_align_onsets, stable_seed, synthesize_align_onsets

log = logging.getLogger(__name__)

CSV_NAME = "e-gmd-v1.0.0.csv"
TRACKERS = ("madmom", "beat_transformer", "beat_this")
TEMPO_BANDS: tuple[tuple[float, float], ...] = ((0, 90), (90, 120), (120, 150), (150, 1e9))
SANITY_MIN_COVERAGE = 0.50  # GT downbeats needing a nearby onset to keep a clip


@dataclass(frozen=True, slots=True)
class Clip:
    track_id: str
    audio_path: Path
    midi_path: Path
    bpm: float
    time_sig: tuple[int, int]
    duration: float

    @property
    def band(self) -> str:
        for lo, hi in TEMPO_BANDS:
            if lo <= self.bpm < hi:
                return f"{int(lo)}-{int(hi) if hi < 1e9 else '+'}"
        return "?"

    @property
    def is_four_four(self) -> bool:
        return self.time_sig == (4, 4)


# ---------- selection ----------

# Documented project default (`training/data_paths.toml.example`,
# `fetch_egmd.sh` downloads here). Used when neither --root, $DRUMJOT_EGMD,
# nor training/data_paths.toml resolves.
_CODEBOX_DEFAULT = Path("/codebox-workspace/datasets/e-gmd-v1.0.0")


def resolve_root(cli_root: Path | None) -> Path:
    """Resolve the E-GMD root the way the training pipeline does.

    Order: explicit --root, $DRUMJOT_EGMD, `training/data_paths.toml`
    (`egmd` key), the documented codebox default, then the in-tree
    benchmarks dataset dir.
    """
    if cli_root is not None:
        return cli_root
    env = os.environ.get("DRUMJOT_EGMD")
    if env:
        return Path(env)
    toml_path = Path(__file__).resolve().parents[2] / "training" / "data_paths.toml"
    if toml_path.exists():
        egmd = tomllib.loads(toml_path.read_text()).get("egmd")
        if egmd:
            return Path(egmd)
    if (_CODEBOX_DEFAULT / CSV_NAME).exists():
        return _CODEBOX_DEFAULT
    return Path(__file__).resolve().parent / "datasets" / "e-gmd"


def _read_rows(root: Path, split: str) -> list[dict]:
    csv_path = root / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"E-GMD CSV missing at {csv_path}. Set $DRUMJOT_EGMD / "
            f"training/data_paths.toml, or pass --root."
        )
    # E-GMD's `test` split is 100% 4/4; non-4/4 meters live only in
    # train/validation. madmom + Beat Transformer are pretrained (never
    # trained on E-GMD), so there's no leakage in sampling across splits,
    # and `all` is the default so the non-4/4 quota can be filled.
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if split != "all":
        rows = [r for r in rows if r.get("split") == split]
    return rows


def _to_clip(root: Path, row: dict) -> Clip | None:
    try:
        bpm = float(row["bpm"])
        duration = float(row["duration"])
        time_sig = parse_time_sig(row["time_signature"])
        audio = root / row["audio_filename"]
        midi = root / row["midi_filename"]
    except (KeyError, ValueError):
        return None
    if bpm <= 0 or duration <= 0:
        return None
    return Clip(row["audio_filename"], audio, midi, bpm, time_sig, duration)


def _n_bars(c: Clip) -> float:
    num, den = c.time_sig
    beat_sec = (60.0 / c.bpm) * (4.0 / den)
    bar_sec = beat_sec * num
    return c.duration / bar_sec if bar_sec > 0 else 0.0


def _band_index(bpm: float) -> int:
    for i, (lo, hi) in enumerate(TEMPO_BANDS):
        if lo <= bpm < hi:
            return i
    return len(TEMPO_BANDS) - 1


def _stratified(clips: list[Clip], k: int) -> list[Clip]:
    """Round-robin across tempo bands, deterministic within a band."""
    if k <= 0:
        return []
    buckets: list[list[Clip]] = [[] for _ in TEMPO_BANDS]
    for c in clips:
        buckets[_band_index(c.bpm)].append(c)
    for b in buckets:
        b.sort(key=lambda c: c.track_id)
    chosen: list[Clip] = []
    cursors = [0] * len(buckets)
    while len(chosen) < k and any(cursors[i] < len(buckets[i]) for i in range(len(buckets))):
        for i, b in enumerate(buckets):
            if cursors[i] < len(b):
                chosen.append(b[cursors[i]])
                cursors[i] += 1
                if len(chosen) >= k:
                    break
    return chosen


def _select_nonfourfour(nff: list[Clip], quota: int, min_per_meter: int) -> list[Clip]:
    """Non-4/4 selection that guarantees every meter is represented.

    First reserve `min_per_meter` clips of each meter present (capped by
    availability), so rare meters (E-GMD has only 43 each of 5/4 & 5/8)
    can't be crowded out by 3/4 & 6/8; then fill the rest of the quota
    tempo-stratified across whatever's left. The per-meter guarantee wins
    if it would exceed the quota.
    """
    by_meter: dict[tuple[int, int], list[Clip]] = {}
    for c in nff:
        by_meter.setdefault(c.time_sig, []).append(c)
    chosen: list[Clip] = []
    chosen_ids: set[str] = set()
    for meter in sorted(by_meter):
        for c in _stratified(by_meter[meter], min(min_per_meter, len(by_meter[meter]))):
            if c.track_id not in chosen_ids:
                chosen.append(c)
                chosen_ids.add(c.track_id)
    remaining = [c for c in nff if c.track_id not in chosen_ids]
    for c in _stratified(remaining, max(0, quota - len(chosen))):
        chosen.append(c)
        chosen_ids.add(c.track_id)
    return chosen


def select_clips(
    clips: Iterable[Clip],
    n_total: int = 96,
    nonfourfour_quota: int = 24,
    min_bars: int = 8,
    min_per_meter: int = 4,
) -> list[Clip]:
    eligible = [c for c in clips if _n_bars(c) >= min_bars]
    ff = [c for c in eligible if c.is_four_four]
    nff = [c for c in eligible if not c.is_four_four]
    nff_sel = _select_nonfourfour(nff, nonfourfour_quota, min_per_meter)
    ff_sel = _stratified(ff, n_total - len(nff_sel))
    return sorted(nff_sel + ff_sel, key=lambda c: c.track_id)


# ---------- scoring ----------

def _median_diff(xs) -> float:
    import numpy as np
    a = np.asarray(xs, dtype=float)
    return float(np.median(np.diff(a))) if a.size >= 2 else 0.0


def _score(grid: GtGrid, beats: list[float], downbeats: list[float]) -> dict:
    import mir_eval
    import numpy as np

    ref_b = np.asarray(grid.beats, dtype=float)
    ref_d = np.asarray(grid.downbeats, dtype=float)
    est_b = np.asarray(beats, dtype=float)
    est_d = np.asarray(downbeats, dtype=float)

    beat_f = float(mir_eval.beat.f_measure(ref_b, est_b)) if est_b.size else 0.0
    downbeat_f = float(mir_eval.beat.f_measure(ref_d, est_d)) if est_d.size and ref_d.size else 0.0
    if est_b.size >= 2 and ref_b.size >= 2:
        _cmlc, cmlt, _amlc, amlt = mir_eval.beat.continuity(ref_b, est_b)
    else:
        cmlt = amlt = 0.0

    # Tempo derived uniformly from each tracker's own beat spacing (median
    # inter-beat interval), so all three are compared apples-to-apples.
    est_bpm = 60.0 / _median_diff(beats) if _median_diff(beats) > 0 else 0.0
    tempo_err = abs(est_bpm - grid.bpm)
    tempo_oct = min(abs(est_bpm - grid.bpm), abs(2 * est_bpm - grid.bpm), abs(est_bpm / 2 - grid.bpm)) if est_bpm else grid.bpm

    # Bar-grouping correctness (the Drumjot-relevant metric): does the
    # detected bar span the right number of beats so bars don't drift?
    # 6/8-vs-3/4 internal feel is irrelevant; a wrong beats-per-bar (5/4
    # read as 4/4 -> ratio 0.8) breaks alignment.
    gt_bar = _median_diff(grid.downbeats)
    det_bar = _median_diff(downbeats)
    bar_len_ratio = det_bar / gt_bar if gt_bar > 0 and det_bar > 0 else 0.0
    return {
        "downbeat_f": downbeat_f,
        "bar_len_ratio": bar_len_ratio,
        "bar_len_ok": abs(bar_len_ratio - 1.0) <= 0.06,
        "tempo_oct_within4": tempo_oct <= 0.04 * grid.bpm,
        "tempo_err": float(tempo_err),
        "tempo_oct": float(tempo_oct),
        "amlt": float(amlt),
        "beat_f": beat_f,
        "cmlt": float(cmlt),
        "est_bpm": float(est_bpm),
        "gt_bpm": float(grid.bpm),
    }


# ---------- run ----------

def _build_align_onsets(
    clip: Clip, mode: str, onsets: list[LaneOnset]
) -> list[tuple[float, float]]:
    if mode == "gt":
        return gt_align_onsets(onsets)
    if mode == "synthetic":
        return synthesize_align_onsets(onsets, stable_seed(clip.track_id), clip.duration)
    if mode == "adtof":
        from app.pipeline.adtof_onsets import detect_drum_onsets_for_alignment
        return detect_drum_onsets_for_alignment(clip.audio_path)
    raise ValueError(f"unknown onsets mode {mode!r}")


_BEAT_THIS = None


def _beat_this_model():
    global _BEAT_THIS
    if _BEAT_THIS is None:
        from beat_this.inference import File2Beats
        _BEAT_THIS = File2Beats(device="cpu", dbn=False)
    return _BEAT_THIS


def _uniform_offset(beats: list[float], align_onsets: list[tuple[float, float]],
                    max_distance: float = 0.05, min_cov: float = 0.30) -> float:
    """Median beat→strongest-nearby-onset offset (same as the production
    fine aligner), so Beat This!'s grid gets the same lag correction the
    DBN trackers get. Returns 0 below the coverage gate."""
    import numpy as np
    if not align_onsets or not beats:
        return 0.0
    arr = sorted(align_onsets)
    times = np.array([t for t, _ in arr])
    strengths = np.array([s for _, s in arr])
    deltas = []
    for bt in beats:
        lo = int(np.searchsorted(times, bt - max_distance, "left"))
        hi = int(np.searchsorted(times, bt + max_distance, "right"))
        if lo < hi:
            deltas.append(float(times[lo + int(np.argmax(strengths[lo:hi]))] - bt))
    if not deltas or len(deltas) / len(beats) < min_cov:
        return 0.0
    return float(np.median(deltas))


def _run_tracker(clip: Clip, tracker: str, align_onsets: list[tuple[float, float]]) -> dict:
    if tracker == "beat_this":
        beats_arr, db_arr = _beat_this_model()(str(clip.audio_path))
        beats = [float(x) for x in beats_arr]
        downbeats = [float(x) for x in db_arr]
        off = _uniform_offset(beats, align_onsets)
        return {"beats": [b + off for b in beats], "downbeats": [d + off for d in downbeats]}

    from app.config import settings
    from app.pipeline.beats import analyze_beats

    prev = settings.beat_tracker
    settings.beat_tracker = tracker  # type: ignore[assignment]
    try:
        structure = analyze_beats(clip.audio_path, clip.duration, align_onsets)
    finally:
        settings.beat_tracker = prev  # type: ignore[assignment]
    beats = [b.time for b in structure.beats]
    downbeats = [b.time for b in structure.beats if b.beat_in_bar == 1]
    return {"beats": beats, "downbeats": downbeats}


def run(
    root: Path,
    mode: str,
    out_dir: Path,
    n_total: int,
    nonfourfour_quota: int,
    min_bars: int,
    split: str,
    min_per_meter: int,
) -> None:
    rows = _read_rows(root, split)
    clips = [c for c in (_to_clip(root, r) for r in rows) if c is not None]
    # Select from CSV metadata first, then stat only the chosen clips. Stat'ing
    # all ~45k clips up front hammers the HDD-backed NFS for no reason.
    selected = select_clips(clips, n_total, nonfourfour_quota, min_bars, min_per_meter)
    missing = [c for c in selected if not (c.audio_path.exists() and c.midi_path.exists())]
    if missing:
        log.warning("%d selected clips missing on disk, dropping: %s",
                    len(missing), [c.track_id for c in missing])
        selected = [c for c in selected if c not in missing]
    meters = collections.Counter(f"{c.time_sig[0]}/{c.time_sig[1]}" for c in selected)
    log.info("selected %d clips from %d candidates; meters: %s",
             len(selected), len(clips), dict(sorted(meters.items())))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_clips.txt").write_text(
        "\n".join(f"{c.track_id}\t{c.bpm:.1f}\t{c.time_sig[0]}/{c.time_sig[1]}\t{c.band}"
                  for c in selected) + "\n"
    )

    per_clip: list[dict] = []
    with (out_dir / "per_clip.jsonl").open("w") as jf:
        for i, clip in enumerate(selected, 1):
            onsets = lane_onsets(clip.midi_path)
            grid = gt_grid(clip.midi_path, clip.time_sig)
            cov = sanity_coverage(grid, onsets)
            if cov < SANITY_MIN_COVERAGE:
                log.warning("[%d/%d] DROP %s: GT downbeat coverage %.0f%% < %.0f%% (loose timing)",
                            i, len(selected), clip.track_id, cov * 100, SANITY_MIN_COVERAGE * 100)
                continue
            align = _build_align_onsets(clip, mode, onsets)
            row: dict = {"track_id": clip.track_id, "band": clip.band,
                         "time_sig": f"{clip.time_sig[0]}/{clip.time_sig[1]}",
                         "is_4_4": clip.is_four_four, "sanity_cov": cov}
            for tracker in TRACKERS:
                pred = _run_tracker(clip, tracker, align)
                row[tracker] = _score(grid, pred["beats"], pred["downbeats"])
            jf.write(json.dumps(row) + "\n")
            jf.flush()
            per_clip.append(row)
            log.info("[%d/%d] %s  downbeat_f mad=%.2f bt=%.2f bthis=%.2f | bar_ok %d/%d/%d",
                     i, len(selected), clip.track_id,
                     row["madmom"]["downbeat_f"], row["beat_transformer"]["downbeat_f"],
                     row["beat_this"]["downbeat_f"],
                     row["madmom"]["bar_len_ok"], row["beat_transformer"]["bar_len_ok"],
                     row["beat_this"]["bar_len_ok"])

    _write_summary(out_dir, mode, per_clip)
    log.info("wrote report to %s", out_dir / "summary.md")


# ---------- report ----------

def _med(rows: list[dict], tracker: str, metric: str) -> float:
    import statistics
    vals = [float(r[tracker][metric]) for r in rows]
    return statistics.median(vals) if vals else 0.0


# Drumjot-relevant metrics: bar alignment + bar grouping + octave-safe tempo.
# Denominator / 3-4-vs-6-8 feel are cosmetic, so AMLt & beat_f are diagnostics.
REPORT_METRICS = ["downbeat_f", "bar_len_ok", "tempo_oct_within4", "beat_f", "amlt"]


def _table(rows: list[dict]) -> list[str]:
    head = "| metric | " + " | ".join(TRACKERS) + " |"
    lines = [head, "|" + "---|" * (len(TRACKERS) + 1)]
    for m in REPORT_METRICS:
        cells = " | ".join(f"{_med(rows, t, m):.3f}" for t in TRACKERS)
        lines.append(f"| {m} | {cells} |")
    return lines


def _write_summary(out_dir: Path, mode: str, rows: list[dict]) -> None:
    out = [f"# Beat-tracker A/B, onsets={mode}", "",
           f"Scored clips: **{len(rows)}** "
           f"({sum(r['is_4_4'] for r in rows)}× 4/4, {sum(not r['is_4_4'] for r in rows)}× non-4/4). "
           "Values are **medians** (data is bimodal; means mislead).", "",
           "Drumjot-relevant primaries: **downbeat_f** (bar alignment) + "
           "**bar_len_ok** (right beats-per-bar so bars don't drift) + "
           "**tempo_oct_within4**. beat_f / amlt are diagnostics "
           "(denominator & 3/4-vs-6/8 feel don't matter for note placement).", "",
           "## Overall", ""]
    out += _table(rows)

    ff = [r for r in rows if r["is_4_4"]]
    nff = [r for r in rows if not r["is_4_4"]]
    out += ["", "## 4/4 vs non-4/4", ""]
    for label, sub in (("4/4", ff), ("non-4/4", nff)):
        if sub:
            out += [f"### {label} ({len(sub)} clips)", ""] + _table(sub) + [""]

    out += ["## Per time signature", ""]
    for ts in sorted({r["time_sig"] for r in rows}):
        sub = [r for r in rows if r["time_sig"] == ts]
        out += [f"### {ts} ({len(sub)} clips)", ""] + _table(sub) + [""]

    out += ["## Per tempo band", ""]
    for band in sorted({r["band"] for r in rows}):
        sub = [r for r in rows if r["band"] == band]
        out += [f"### {band} BPM ({len(sub)} clips)", ""] + _table(sub) + [""]
    (out_dir / "summary.md").write_text("\n".join(out))


# ---------- cli ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="madmom vs Beat Transformer vs Beat This! beat-tracker A/B")
    p.add_argument("--onsets", choices=["synthetic", "gt", "adtof"], default="synthetic",
                   help="align-onset source (synthetic & gt are CPU-only; adtof needs the GPU)")
    p.add_argument("--root", type=Path, default=None,
                   help="E-GMD root; defaults to $DRUMJOT_EGMD / data_paths.toml / codebox")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "out" / "beat_ab")
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--nonfourfour", type=int, default=24)
    p.add_argument("--min-per-meter", type=int, default=4,
                   help="guaranteed min clips per non-4/4 meter (3/4, 6/8, 5/4, 5/8)")
    p.add_argument("--min-bars", type=int, default=8)
    p.add_argument("--split", default="all",
                   help="E-GMD split filter, or 'all' (default; needed for non-4/4 coverage)")
    args = p.parse_args(argv)

    if args.onsets != "adtof":
        # GPU is reserved for training; force BT inference onto CPU and pin
        # threads (local box is a 12-thread 7800X3D). Must precede any torch
        # import, which analyze_beats does lazily.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("OMP_NUM_THREADS", "8")
    # BT's default checkpoint path is the docker `/app/checkpoints`; point it
    # at the writable in-tree dir so the local run can download/load there.
    os.environ.setdefault(
        "BEAT_TRANSFORMER_CHECKPOINT",
        str(Path(__file__).resolve().parents[1] / "checkpoints" / "beat_transformer.pt"),
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _seed_determinism()
    run(resolve_root(args.root), args.onsets,
        args.output_dir.with_name(args.output_dir.name + f"_{args.onsets}"),
        args.n, args.nonfourfour, args.min_bars, args.split, args.min_per_meter)


def _seed_determinism() -> None:
    try:
        import torch
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as exc:  # torch optional at import time on a bare box
        log.warning("torch determinism setup skipped: %s", exc)


if __name__ == "__main__":
    main()
