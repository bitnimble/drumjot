"""Benchmark CLI: iterate a dataset, transcribe each track, score against ground truth.

Run with:
    cd transcriber
    python -m benchmarks.run_benchmark --dataset e-gmd --limit 10

Requires a running transcriber service (see `benchmarks/README.md`).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .core.midi_events import midi_bytes_to_events
from .core.score import (
    DEFAULT_TOLERANCE_SECONDS,
    DatasetSummary,
    TrackScore,
    score_track,
    summarise,
)
from .core.transcribe_client import (
    TranscribeOptions,
    fetch_prediction_midi,
    transcribe_file,
    wait_for_service,
)
from .loaders import LoadedTrack, get_loader

log = logging.getLogger("drumjot.benchmark")

DATASETS_ROOT = Path(__file__).resolve().parent / "datasets"


# ---------------------------------------------------------------------------
# CLI parsing


@dataclass
class CliArgs:
    dataset: str
    limit: int | None
    sample_ratio: float
    seed: int
    beat_input: str
    service_url: str
    split: str
    tolerance: float
    output_dir: Path
    resume: bool


def parse_args(argv: list[str] | None = None) -> CliArgs:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.run_benchmark",
        description="Score the running Drumjot transcriber against an ADT dataset.",
    )
    p.add_argument("--dataset", required=True, choices=["e-gmd", "mdb-drums", "idmt-smt-drums"])
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Hard cap on tracks (applied AFTER --sample-ratio). Default: no cap.",
    )
    p.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of the test split (0.0-1.0). Default: 1.0.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --sample-ratio sampling.")
    p.add_argument(
        "--beat-input",
        choices=["full_mix", "drum_stem"],
        default="full_mix",
        help=(
            "Which audio feeds the beat tracker. `full_mix` (default) is "
            "Beat This!'s training distribution; `drum_stem` can help on tracks "
            "with heavy non-drum syncopation."
        ),
    )
    p.add_argument(
        "--service-url",
        default="http://localhost:8001",
        help="Transcriber base URL. Default: http://localhost:8001.",
    )
    p.add_argument(
        "--split",
        default="test",
        help="E-GMD only: which CSV split to evaluate. Default: test.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help="mir_eval onset match window in seconds. Default: 0.05 (N2N).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write per_track.jsonl + summary.json. Default: benchmarks/results/<ds>/<ts>.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip tracks that already have a result in --output-dir/per_track.jsonl.",
    )

    ns = p.parse_args(argv)

    if not (0.0 < ns.sample_ratio <= 1.0):
        p.error("--sample-ratio must be in (0.0, 1.0]")
    if ns.limit is not None and ns.limit <= 0:
        p.error("--limit must be positive")

    output_dir = ns.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "results" / ns.dataset / ts

    return CliArgs(
        dataset=ns.dataset,
        limit=ns.limit,
        sample_ratio=ns.sample_ratio,
        seed=ns.seed,
        beat_input=ns.beat_input,
        service_url=ns.service_url,
        split=ns.split,
        tolerance=ns.tolerance,
        output_dir=output_dir,
        resume=ns.resume,
    )


# ---------------------------------------------------------------------------
# Track selection (sample-ratio + limit)


def _apply_sampling(
    tracks: list[LoadedTrack],
    sample_ratio: float,
    limit: int | None,
    seed: int,
) -> list[LoadedTrack]:
    """Deterministically subsample then cap.

    sample-ratio is applied first as a target fraction (count =
    floor(N * ratio), with a floor of 1 if ratio > 0 and N > 0), then
    --limit caps. Using `random.sample` with a fixed seed makes the
    selection reproducible for a given (N, ratio, seed).
    """
    if sample_ratio < 1.0 and tracks:
        target = max(1, int(len(tracks) * sample_ratio))
        rng = random.Random(seed)
        tracks = rng.sample(tracks, target)
        tracks.sort(key=lambda t: t.track_id)
    if limit is not None:
        tracks = tracks[:limit]
    return tracks


# ---------------------------------------------------------------------------
# Per-run state (resume + streaming output)


def _load_completed_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("track_id")
            if isinstance(tid, str):
                done.add(tid)
    return done


def _iter_scoreable_tracks(
    loader, dataset_root: Path, args: CliArgs
) -> Iterator[LoadedTrack]:
    """Build the full track list, log selection stats, then yield."""
    all_tracks = list(loader.iter_tracks(dataset_root))
    log.info(
        "Discovered %d candidate tracks under %s (split=%s).",
        len(all_tracks),
        dataset_root,
        args.split if args.dataset == "e-gmd" else "n/a",
    )
    if not all_tracks:
        return

    selected = _apply_sampling(all_tracks, args.sample_ratio, args.limit, args.seed)
    log.info(
        "Selected %d / %d tracks (sample_ratio=%.3f, limit=%s, seed=%d).",
        len(selected),
        len(all_tracks),
        args.sample_ratio,
        args.limit,
        args.seed,
    )
    yield from selected


# ---------------------------------------------------------------------------
# Main loop


def run(args: CliArgs) -> DatasetSummary:
    dataset_root = DATASETS_ROOT / args.dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"{dataset_root} not found. Paste the dataset there per the folder's README."
        )

    loader = get_loader(args.dataset)
    if args.dataset == "e-gmd":
        # Mutate the loader's split — only E-GMD has a `split` knob.
        loader.split = args.split  # type: ignore[attr-defined]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_track_path = args.output_dir / "per_track.jsonl"
    summary_path = args.output_dir / "summary.json"

    completed = _load_completed_ids(per_track_path) if args.resume else set()
    if completed:
        log.info("Resuming: %d tracks already scored, will skip those.", len(completed))

    log.info("Sanity-checking service at %s ...", args.service_url)
    wait_for_service(args.service_url)
    log.info("Service ready. Starting benchmark run -> %s", args.output_dir)

    options = TranscribeOptions(beat_input=args.beat_input)

    track_scores: list[TrackScore] = []
    # If resuming, re-load previously-scored tracks so the summary
    # reflects every scored item, not just the ones from this resumption.
    if completed:
        track_scores.extend(_load_existing_track_scores(per_track_path))

    n_run = 0
    n_failed = 0
    started = time.perf_counter()
    with per_track_path.open("a") as out_fh:
        for track in _iter_scoreable_tracks(loader, dataset_root, args):
            if track.track_id in completed:
                continue
            t0 = time.perf_counter()
            try:
                result = transcribe_file(
                    track.audio_path,
                    options=options,
                    service_url=args.service_url,
                )
            except Exception as exc:
                n_failed += 1
                log.warning("transcribe failed for %s: %s", track.track_id, exc)
                _write_failure(out_fh, track.track_id, f"transcribe: {exc}")
                continue

            try:
                if not result.prediction_midi_url:
                    raise ValueError("service returned no prediction_midi_url")
                midi_bytes = fetch_prediction_midi(
                    args.service_url, result.prediction_midi_url
                )
                predicted = midi_bytes_to_events(midi_bytes)
            except Exception as exc:
                n_failed += 1
                log.warning(
                    "prediction decode failed for %s: %s",
                    track.track_id, exc,
                )
                _write_failure(out_fh, track.track_id, f"decode: {exc}")
                continue

            score = score_track(
                track.track_id, track.reference, predicted, tolerance=args.tolerance
            )
            track_scores.append(score)
            n_run += 1

            record = score.as_jsonable()
            record["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
            record["n_reference"] = len(track.reference)
            record["n_estimated"] = len(predicted)
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

            log.info(
                "[%d] %s  F1=%.3f  KD=%.2f SD=%.2f HH=%.2f  ref=%d est=%d  %.1fs",
                n_run,
                track.track_id,
                score.f1_macro,
                _safe_f1(score, "KD"),
                _safe_f1(score, "SD"),
                _safe_f1(score, "HH"),
                len(track.reference),
                len(predicted),
                record["elapsed_seconds"],
            )

    elapsed = time.perf_counter() - started
    summary = summarise(args.dataset, track_scores, tolerance=args.tolerance)

    summary_payload = summary.as_jsonable()
    summary_payload["wall_clock_seconds"] = round(elapsed, 1)
    summary_payload["n_failed"] = n_failed
    summary_payload["options"] = {
        "beat_input": args.beat_input,
        "sample_ratio": args.sample_ratio,
        "limit": args.limit,
        "seed": args.seed,
        "split": args.split,
        "tolerance_seconds": args.tolerance,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

    log.info("=" * 72)
    log.info(
        "Done. %d scored, %d failed in %.1fs. F1_macro_mean=%.3f  F1_weighted_mean=%.3f",
        summary.n_tracks,
        n_failed,
        elapsed,
        summary.f1_macro_mean,
        summary.f1_weighted_mean,
    )
    for cls_name, f1 in summary_payload["per_class_f1_mean"].items():
        n_ref = summary_payload["per_class_n_reference"].get(cls_name, 0)
        log.info("  %s: F1=%.3f over %d ref onsets", cls_name, f1, n_ref)
    log.info("Summary written to %s", summary_path)
    return summary


def _safe_f1(score: TrackScore, cls_name: str) -> float:
    for cls, s in score.per_class.items():
        if cls.value == cls_name:
            return s.f1
    return float("nan")


def _write_failure(fh, track_id: str, reason: str) -> None:
    fh.write(json.dumps({"track_id": track_id, "failed": True, "reason": reason}) + "\n")
    fh.flush()


def _load_existing_track_scores(per_track_path: Path) -> list[TrackScore]:
    """Hydrate per-track scores written by a prior run so the summary aggregates them too.

    Only the bits needed for `summarise()` are reconstructed — full
    fidelity isn't required since the JSONL itself is the source of truth.
    """
    from .core.classes import DrumClass
    from .core.score import ClassScore

    out: list[TrackScore] = []
    with per_track_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("failed"):
                continue
            per_class_raw = rec.get("per_class") or {}
            per_class: dict[DrumClass, ClassScore] = {}
            for cls_name, payload in per_class_raw.items():
                try:
                    cls = DrumClass(cls_name)
                except ValueError:
                    continue
                per_class[cls] = ClassScore(
                    drum_class=cls,
                    precision=float(payload.get("precision", 0.0)),
                    recall=float(payload.get("recall", 0.0)),
                    f1=float(payload.get("f1", 0.0)),
                    n_reference=int(payload.get("n_reference", 0)),
                    n_estimated=int(payload.get("n_estimated", 0)),
                )
            out.append(
                TrackScore(
                    track_id=str(rec["track_id"]),
                    per_class=per_class,
                    f1_macro=float(rec.get("f1_macro", 0.0)),
                    f1_weighted=float(rec.get("f1_weighted", 0.0)),
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    try:
        run(args)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("Interrupted. Partial results retained in %s", args.output_dir)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
