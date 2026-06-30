"""Deterministic A/B between the two beat-tracking front-ends.

madmom (`RNNDownBeatProcessor`) vs Beat Transformer, both run through the
*identical* production grid (shared madmom DBN -> gated
`align_beats_to_onsets` -> `_finalize_bar_tempos`) and scored against
E-GMD MIDI-derived ground truth with mir_eval.

The verdict's primaries are **downbeat F-measure** (drives time-sig + bar
drift) and **AMLt / tempo** (the only BPM source). Beat F-measure is a
secondary diagnostic only.

See `docs/superpowers/specs/2026-06-30-beat-tracker-ab-design.md`.

Run (CPU, synthetic onsets, while the GPU is busy), from `transcriber/`:

    .venv/bin/python -m benchmarks.beat_ab --onsets synthetic

`--onsets adtof` is the truthful production path but needs the GPU.
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
TRACKERS = ("madmom", "beat_transformer")
TEMPO_BANDS: tuple[tuple[float, float], ...] = ((0, 90), (90, 120), (120, 150), (150, 1e9))
SANITY_MIN_COVERAGE = 0.50  # GT beats needing a nearby MIDI onset to keep a clip

# Metrics where higher is better (vs tempo error, where lower is better).
HIGHER_BETTER = ("downbeat_f", "amlt", "beat_f", "cmlt")


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

def _score(grid: GtGrid, beats: list[float], downbeats: list[float], bpm: float) -> dict:
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
    tempo_err = abs(bpm - grid.bpm)
    return {
        "downbeat_f": downbeat_f,
        "amlt": float(amlt),
        "beat_f": beat_f,
        "cmlt": float(cmlt),
        "tempo_err": float(tempo_err),
        "tempo_within4": tempo_err <= 0.04 * grid.bpm,
        "tempo_within8": tempo_err <= 0.08 * grid.bpm,
        "est_bpm": float(bpm),
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


def _run_tracker(clip: Clip, tracker: str, align_onsets: list[tuple[float, float]]) -> dict:
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
    return {"beats": beats, "downbeats": downbeats, "bpm": structure.initial_tempo}


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
    clips = [c for c in clips if c.audio_path.exists() and c.midi_path.exists()]
    selected = select_clips(clips, n_total, nonfourfour_quota, min_bars, min_per_meter)
    meters = collections.Counter(f"{c.time_sig[0]}/{c.time_sig[1]}" for c in selected)
    log.info("selected %d clips from %d eligible; meters: %s",
             len(selected), len(clips), dict(sorted(meters.items())))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_clips.txt").write_text(
        "\n".join(f"{c.track_id}\t{c.bpm:.1f}\t{c.time_sig[0]}/{c.time_sig[1]}\t{c.band}"
                  for c in selected) + "\n"
    )

    per_clip: list[dict] = []
    with (out_dir / "per_clip.jsonl").open("w") as jf:
        for i, clip in enumerate(selected, 1):
            # Tempo reference from the MIDI tempo map (quarter-note BPM, the
            # same unit the trackers report) rather than the CSV's possibly
            # eighth-relative label.
            grid = gt_grid(clip.midi_path, clip.time_sig)
            onsets = lane_onsets(clip.midi_path)
            cov = sanity_coverage(grid, onsets)
            if cov < SANITY_MIN_COVERAGE:
                log.warning("[%d/%d] DROP %s: GT beat coverage %.0f%% < %.0f%% (loose timing / phase)",
                            i, len(selected), clip.track_id, cov * 100, SANITY_MIN_COVERAGE * 100)
                continue
            align = _build_align_onsets(clip, mode, onsets)
            row: dict = {"track_id": clip.track_id, "band": clip.band,
                         "time_sig": f"{clip.time_sig[0]}/{clip.time_sig[1]}",
                         "is_4_4": clip.is_four_four, "sanity_cov": cov}
            for tracker in TRACKERS:
                pred = _run_tracker(clip, tracker, align)
                row[tracker] = _score(grid, pred["beats"], pred["downbeats"], pred["bpm"])
            jf.write(json.dumps(row) + "\n")
            jf.flush()
            per_clip.append(row)
            log.info("[%d/%d] %s  dbF madmom=%.3f bt=%.3f  AMLt=%.3f/%.3f",
                     i, len(selected), clip.track_id,
                     row["madmom"]["downbeat_f"], row["beat_transformer"]["downbeat_f"],
                     row["madmom"]["amlt"], row["beat_transformer"]["amlt"])

    _write_summary(out_dir, mode, per_clip)
    log.info("wrote report to %s", out_dir / "summary.md")


# ---------- report ----------

def _mean(rows: list[dict], tracker: str, metric: str) -> float:
    vals = [float(r[tracker][metric]) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _wilcoxon(rows: list[dict], metric: str) -> str:
    if len(rows) < 6:
        return "n<6"
    from scipy.stats import wilcoxon
    diffs = [float(r["beat_transformer"][metric]) - float(r["madmom"][metric]) for r in rows]
    if all(d == 0 for d in diffs):
        return "p=1.000 (no diff)"
    try:
        _, p = wilcoxon(diffs)
        return f"p={p:.3f}"
    except ValueError as exc:
        return f"n/a ({exc})"


def _table(rows: list[dict], metrics: list[str]) -> list[str]:
    lines = ["| metric | madmom | beat_transformer | delta (bt-mad) | wilcoxon |",
             "|---|---|---|---|---|"]
    for m in metrics:
        mad = _mean(rows, "madmom", m)
        bt = _mean(rows, "beat_transformer", m)
        lines.append(f"| {m} | {mad:.3f} | {bt:.3f} | {bt - mad:+.3f} | {_wilcoxon(rows, m)} |")
    return lines


def _write_summary(out_dir: Path, mode: str, rows: list[dict]) -> None:
    metrics = ["downbeat_f", "amlt", "tempo_err", "tempo_within4", "tempo_within8", "beat_f", "cmlt"]
    out = [f"# Beat-tracker A/B, onsets={mode}", "",
           f"Scored clips: **{len(rows)}** "
           f"({sum(r['is_4_4'] for r in rows)}× 4/4, {sum(not r['is_4_4'] for r in rows)}× non-4/4).",
           "",
           "Primary verdict metrics: **downbeat_f**, **amlt**, **tempo_***. "
           "beat_f / cmlt are diagnostics.", "",
           "## Overall", ""]
    out += _table(rows, metrics)

    ff = [r for r in rows if r["is_4_4"]]
    nff = [r for r in rows if not r["is_4_4"]]
    out += ["", "## 4/4 vs non-4/4 (downbeat stress test)", ""]
    for label, sub in (("4/4", ff), ("non-4/4", nff)):
        if not sub:
            continue
        out += [f"### {label} ({len(sub)} clips)", ""] + _table(sub, ["downbeat_f", "amlt", "tempo_err"]) + [""]

    out += ["## Per tempo band", ""]
    bands = sorted({r["band"] for r in rows})
    for band in bands:
        sub = [r for r in rows if r["band"] == band]
        out += [f"### {band} BPM ({len(sub)} clips)", ""] + _table(sub, ["downbeat_f", "amlt", "tempo_err"]) + [""]

    out += ["## Verdict", "",
            "_Fill from the tables above: if downbeat_f & amlt are within paired noise "
            "(wilcoxon p>0.05), the cleanup argument decides → drop Beat Transformer "
            "(bigger code win; madmom owns the shared DBN regardless)._", ""]
    (out_dir / "summary.md").write_text("\n".join(out))


# ---------- cli ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="madmom vs Beat Transformer beat-tracker A/B")
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
