# Benchmarks

Offline test harness that scores the running transcriber against the
three datasets used by the Noise-to-Notes paper (arXiv 2509.21739):

| Dataset | N2N-reported F1 | Folder to paste into |
|---|---|---|
| E-GMD | 0.897 | `datasets/e-gmd/` |
| MDB Drums | 0.879 | `datasets/mdb-drums/` |
| IDMT-SMT-Drums | 0.949 | `datasets/idmt-smt-drums/` |

The harness does **not** download datasets — see each dataset folder's
`README.md` for what to drop in and the expected layout.

## Evaluation protocol

Mirrors the standard ADT 3-class evaluation that N2N (and most of the
field) report:

- Drum classes: **KD** (kick), **SD** (snare), **HH** (hi-hat, both
  closed and open).
- Metric: `mir_eval.onset.f_measure` per class with a **50 ms** match
  tolerance (matches N2N).
- Per-track score = mean of per-class F1 (only classes with at least
  one reference or predicted onset count).
- Dataset score = mean of per-track scores (macro-averaged).

These three classes are the cross-dataset comparable subset — MDB Drums
and IDMT-SMT-Drums also include other classes (TT/CY/OH/...) but the
field convention is to report 3-class numbers across all three.

## Running

The harness is baked into the transcriber image — there is no separate
container or host install. It POSTs audio to the same service it ships
with (over localhost inside the container) and writes results to a
host-mounted folder, so a typical workflow is:

```bash
# 1. Drop dataset fixtures into the per-dataset folders on the host:
#    transcriber/benchmarks/datasets/e-gmd/
#    transcriber/benchmarks/datasets/mdb-drums/
#    transcriber/benchmarks/datasets/idmt-smt-drums/
# (See each folder's README for the expected layout.)

# 2. Bring the transcriber service up (rebuild if benchmark code changed).
cd transcriber
docker compose up --build -d
# wait until `docker compose logs transcriber` shows "service is ready"

# 3. Run the benchmark inside the running container.
docker compose exec transcriber \
  python3.11 -m benchmarks.run_benchmark --dataset e-gmd --limit 10

# 4. Results land on the host at transcriber/benchmarks/results/<dataset>/<ts>/.
ls benchmarks/results/e-gmd/
```

`per_track.jsonl` is appended to as each track scores, so you can
`tail -f` it during a long run. `summary.json` is written at the end.

Useful knobs:

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | required | One of `e-gmd`, `mdb-drums`, `idmt-smt-drums`. |
| `--limit N` | _(no cap)_ | Hard cap on number of tracks. Applied **after** `--sample-ratio`. |
| `--sample-ratio R` | `1.0` | Randomly keep this fraction of the test split (0.0–1.0). |
| `--seed N` | `0` | RNG seed for `--sample-ratio` sampling. |
| `--beat-input` | `full_mix` | Which audio feeds the beat tracker (`full_mix` or `drum_stem`). |
| `--service-url URL` | `http://localhost:8001` | Transcriber base URL. |
| `--split` | `test` | E-GMD only — which CSV split to evaluate. |
| `--tolerance SEC` | `0.05` | mir_eval onset match window in seconds (N2N uses 0.05). |
| `--output-dir DIR` | `benchmarks/results/<dataset>/<timestamp>` | Per-run output folder. |
| `--resume` | off | Skip tracks that already have a result in `--output-dir`. |

Per-track results are streamed to `per_track.jsonl` as the run
progresses; the aggregate `summary.json` is written at the end. Both
files are safe to inspect mid-run.

Cost note: each track is one /transcribe call. The pipeline makes one
filter LLM call per drum pitch (parallel), so cost scales with how many
instruments are present rather than with knobs at the CLI.

## Meter backtest (`meter_backtest.py`)

A **separate, standalone** harness (no transcriber service, no Docker) that
checks *beats-per-bar* accuracy rather than onset F1. It runs Beat This!
directly on E-GMD audio and compares the modal detected bar length to E-GMD's
ground-truth `time_signature`, with the meter-recovery pass
(`beats._recover_bar_length_if_incoherent`) OFF vs ON. It's the regression
guard for that pass, it must not regress the common meters (4/4, 3/4) while it
rescues the odd meters (5/4, 7/4, 7/8) Beat This!'s DBN-free downbeat head can't
group.

```bash
# Points at the raw E-GMD dataset root (holds e-gmd-v1.0.0.csv + drummerN/),
# NOT the paste-in datasets/e-gmd/ fixture folder above.
PYTHONPATH=transcriber transcriber/.venv/bin/python3 -m benchmarks.meter_backtest \
    --root /codebox-workspace/datasets/e-gmd-v1.0.0 --workers 4
```

Prints per-meter OLD vs NEW accuracy and the detected-bar-length distributions.
Beat This! runs once per song (CPU is fine, ~3-5 s/clip); OLD/NEW diverge only
at the downbeat-grouping stage. `--root` defaults to `$DRUMJOT_EGMD_RAW`.
