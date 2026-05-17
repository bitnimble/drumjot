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
| `--refine` / `--no-refine` | `--refine` | Toggle the F1-gated refinement loop (matches the web UI checkbox). |
| `--lint` / `--no-lint` | `--lint` | Toggle the deterministic Jot lint pass (matches the web UI checkbox). Independent of `--refine`. |
| `--best-of-k N` | `1` | Number of initial transcription candidates (matches the web UI dropdown). |
| `--service-url URL` | `http://localhost:8001` | Transcriber base URL. |
| `--split` | `test` | E-GMD only — which CSV split to evaluate. |
| `--tolerance SEC` | `0.05` | mir_eval onset match window in seconds (N2N uses 0.05). |
| `--output-dir DIR` | `benchmarks/results/<dataset>/<timestamp>` | Per-run output folder. |
| `--resume` | off | Skip tracks that already have a result in `--output-dir`. |

Per-track results are streamed to `per_track.jsonl` as the run
progresses; the aggregate `summary.json` is written at the end. Both
files are safe to inspect mid-run.

Cost note: each track is one /transcribe call. With `--refine` and
`--lint` on and `--best-of-k 3` that's three LLM calls + a lint pass
(N small per-segment calls) + a refinement loop per track. For a
sanity check on a fresh setup, start with
`--limit 20 --no-refine --no-lint --best-of-k 1`.
