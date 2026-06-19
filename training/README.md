# Drumjot drum-onset model training

Trains a drum **onset-detection** model: a **frozen music-SSL encoder**
(MERT / MusicFM) + **separate small per-lane heads**. Feasibility-first, prove a frozen encoder + tiny head learns drum onsets on clean
open-licensed data before any scope commitment.

**Design spec:**
[`docs/superpowers/specs/2026-06-07-drum-onset-frozen-ssl-design.md`](../docs/superpowers/specs/2026-06-07-drum-onset-frozen-ssl-design.md)
(read it first, phasing, the data-cleaning stage, licensing, architecture
decisions).

## Why a separate top-level folder

Research/training code, not part of the transcriber runtime. It **reuses
the transcriber's uv venv** (`transcriber/.venv`) for deps, `torch`,
`transformers`, `librosa`, `mir_eval`, `mido`, `pretty_midi`, `scipy` are
all already there. No separate environment, no new core deps.

MERT load is sandbox-verified: `m-a-p/MERT-v1-330M`, sr 24000, 25 hidden
states, layer-10 features `(frames, 1024)` at ~75 fps. The model's
`nnAudio` warning is harmless (auxiliary CQT feature we don't use), so **no
extra dependency is needed**.

## Layout

```
training/
  drumjot_training/
    lanes.py        9-lane vocab + GM-note mapping
    config.py       Config (encoder, head dims, lr, batch_size, cache_dtype)
    targets.py      onset times -> per-frame Gaussian target curve
    metrics.py      peak-pick + onset-F1 (mir_eval, +/-50 ms)
    model.py        OnsetHead + MultiLaneHeads (separate per-lane BiGRU)
    embeddings.py   frozen MERT features + .npy cache (fp16 default)
    runtime.py      TF32 + bf16 autocast, guarded for non-tensor-core GPUs
    train.py        training loop: padded batches, DataLoader streaming,
                    mixed precision, --resume warm-start, periodic checkpoints
    inference.py    load checkpoint -> windowed transcribe -> per-lane onsets;
                    lane->DSL-pitch map for the transcriber handoff
    checkpoint.py   save/load model.pt + meta.json (with tuned thresholds)
    postfilter.py   envelope support gate + peak alignment (deterministic filter)
    midi_labels.py  MIDI -> per-lane onset times (E-GMD)
    egmd.py / star.py   E-GMD CSV + STAR (.txt, 18->11 fold) loaders
    rlrr.py         ParaDB/Paradiddle .rlrr parser: per-lane onsets, adaptive
                    hat/cymbal fold, bimodal-velocity hats, perc-track gate
    forced_align.py per-note envelope snap + support gate
    dedup.py / clean.py   dedup keys + cleaning-stage dedup/support scoring
    paths.py        dataset paths via env var / data_paths.toml
  scripts/
    fetch_egmd.sh / fetch_star.sh   dataset download + unpack
    extract_star_mix.py     mix-only STAR extract from the zip parts (~39 GB)
    extract_star_subset.py  N-song subset from the parts
    extract_star_stems.py   mirror a subset with re-synth DRUM-STEM audio
    extract_star_balanced.py  class-balanced (rare-lane) selection from the parts
    eval_paradb.py          ParaDB test-set eval (separate -> per-instrument -> score)
    eval_filtered.py        raw vs +envelope-filter F1 for a checkpoint
  Dockerfile        self-contained GPU trainer image (MERT baked, runs as uid 1000)
  tests/            pytest (run in the CUDA sandbox; 111 tests)
```

## Lanes (9)

`k` kick · `s` snare · `ss` side-stick · `t` toms · `hc`/`hp`/`ho`
closed/pedal/open hi-hat · `rd` ride · `cr` crash.

Fold decisions (`lanes.py` / `star.py`): ride and crash are split; the three
hat articulations are separate (the HIHAT.md goal); ride bell folds into `rd`
(same physical cymbal). The `mc` (misc cymbals: splash/china/ride-bell) and
`mp` (misc percussion: cowbell/clap/tambourine) lanes were REMOVED (2026-06):
the separators don't isolate them and they scored ~noise; their source classes
now map to None (except ride bell -> `rd`). Side-stick is its own training lane
and emits to GM-37 on its own MIDI track; the frontend folds it onto the snare
track as a snare articulation at Jot-load (integration, not handled in
training). Rare lanes (`hp`, `ss`) are sparse, expect weak F1 until trained on
the full dataset, with per-lane `pos_weight`.

## Running

Direct `pytest` / `ruff` are denied by the permission config; go through
the scripts.

- **Tests:** run in the CUDA sandbox (it has torch + the stack; `uv pip
  install pytest ruff` once per container):
  `scripts/sandbox-run env PYTHONPATH=.../training python3 -m pytest training/tests -q`.
  Current: **111 passed**. (`scripts/test-py` in `transcriber/.venv` runs the
  torch-free subset on the host; the torch tests skip there.)
- **Anything needing torch / the GPU** (model, MERT, training, eval): the CUDA
  sandbox. Dataset-free self-test:
  `scripts/sandbox-run python3 -c "import sys; sys.path.insert(0,'/home/bitnimble/code/drumjot/training'); from drumjot_training.train import synthetic_smoke; synthetic_smoke(epochs=80)"`.

## Data

**E-GMD:**
1. `bash training/scripts/fetch_egmd.sh` (or let provisioning run it), downloads E-GMD to `/codebox-workspace/datasets` by default.
2. `cp training/data_paths.toml.example training/data_paths.toml` and set
   `egmd` to the extracted root (the dir with `e-gmd-v1.0.0.csv`), or
   `export DRUMJOT_EGMD=...`.

**STAR (mix-only, ~39 GB):** STAR ships as a ~181 GB zip split into
`STAR_Drums_full.zip.part-a?`, but the trainer only reads `audio/mix/*.flac`
+ `annotation/*.txt` (`star.index`), the per-instrument stem buckets are dead
weight. `scripts/extract_star_mix.py` reads the parts as one virtual file and
extracts **only** those members (no 181 GB reassembly), giving a ~39 GB tree
that `star.index` consumes unchanged:

```bash
# parts can stay on slow storage (NAS); reads only ~39 GB of mix bytes
python3 training/scripts/extract_star_mix.py /path/to/star_parts /mnt/ssd/star_mix
export DRUMJOT_STAR=/mnt/ssd/star_mix          # or set `star = ...` in data_paths.toml
```

(`scripts/extract_star_subset.py <parts> <out> <ntrain> <nval>` does the same
for a small N-song subset.) Either tree works with `--dataset star`; the
feature cache lands in `<DRUMJOT_STAR>/_cache_mert`.

## ParaDB test set & scoring (`scripts/eval_paradb.py`)

Real-song, hand-charted held-out test: a folder of Paradiddle `.rlrr` map zips
(`rlrr.py` parses them; see its module docstring for the class map). Per map:

1. **Reconstruct the song** from the chart's audio tracks (`build_mix`). Add the
   drum track only if the song tracks are drumless backing, decided by
   drum/song signal correlation, so a full-mix map's drums aren't double-counted.
2. **Our separation** (BS-Roformer `stems_all` → drum stem → MDX23C `stems_per`
   → kick/snare/hi-hat/cymbals/toms). We ignore the mapper's own drum split.
3. **Run the model on each ISOLATED stem**, keeping only the lanes that belong
   to that stem (`STEM_TO_LANES`) and counting cross-instrument firings as
   leakage (the model's hallucination rate). Optionally gate each kept lane
   through its own stem's onset-strength envelope (the deterministic filter).
4. **Onset-F1 vs the chart**, with per-map optimistic hat/cymbal folding
   (`rlrr.comparison_pairs`: split a group only when the chart distinguishes it;
   hi-hat open/closed also recovered from a strictly-bimodal velocity chart).

**Global offset / alignment check.** Each map's chart is checked against the
drum-stem onset envelope: `support@0` (fraction of chart onsets within ±window
of a real transient) flags corrupted/desynced charts, and a **global offset**
is estimated as the **median signed distance from each chart onset to its
nearest envelope peak**. The median is deliberately *not* the offset that
maximises support/overlap: on dense drum onsets that support landscape is
near-saturated and its argmax overshoots the true offset by tens of ms (chasing
a few straggler onsets), whereas the median reports the true systematic shift.
GT is shifted by it before scoring only when it's a real offset
(`|median| > --offset-correct-min`); well-aligned charts (median a few ms, the
envelope's own flux-lag) are left alone. NB the envelope is a slightly-*late*
reference, so for a strict per-map sync the GT-vs-model offset is fairer; here
the offsets are sub-tolerance so it's mostly a diagnostic.

```bash
# needs the transcriber app + audio-separator models + drumjot_training + a GPU
scripts/sandbox-run env PYTHONPATH=.../training MODELS_DIR=<models-cache> \
  python3 training/scripts/eval_paradb.py --maps-dir <folder-of-zips> \
    --checkpoint <ckpt-dir> --stems-cache <dir>   # stems cached -> fast re-runs
```

## Docker (train on another machine)

`training/Dockerfile` packages a **self-contained, GPU trainer**: the lean
dep stack (torch cu128 + transformers + librosa + mir_eval + mido), the
`drumjot_training` package, and the frozen **MERT-v1-330M weights baked in**.
Datasets are *not* baked in, mount them.

```bash
# build (context = training/)
docker build -f training/Dockerfile -t drumjot-trainer training/

# run on a CUDA box (e.g. RTX 3080; driver new enough for CUDA 12.8)
docker run --rm --gpus all \
    -v /mnt/ssd/e-gmd-v1.0.0:/data/e-gmd \
    -e DRUMJOT_EGMD=/data/e-gmd \
    drumjot-trainer --train-min 240 --val-min 30 --epochs 80
```

Notes:
- The MERT feature cache writes to `<dataset>/_cache_mert`, keep the
  mounted dataset on the **SSD** so caching + reads are fast; the mount must
  be writable. RTX 3080 (10 GB) easily fits frozen MERT-330M + the heads.
- Cache precision is **float16 by default** (~4.6 MB/clip, so the full STAR
  cache is ~28 GB and fits the OS page cache), which halves per-epoch read
  bandwidth at no real cost. Pass `--cache-dtype float32` for a
  full-precision cache (~56 GB). On WSL2, give the VM plenty of RAM (e.g.
  `.wslconfig` `memory=52GB`) so the cache stays resident after epoch 1.
- The baked image runs **HF-offline by default** (no runtime network).
  `--build-arg BAKE_MERT=0` skips baking (weights download at runtime to
  `HF_HOME`, mount a volume there to cache, and pass
  `-e TRANSFORMERS_OFFLINE=0 -e HF_HUB_OFFLINE=0`).
- `docker run drumjot-trainer --help` lists all flags (`--dataset egmd|star`,
  `--overfit-one`, `--train-min`/`--val-min` (E-GMD), `--train-clips`/
  `--val-clips` (STAR), `--max-seconds`, `--epochs`, `--batch-size`,
  `--num-workers`, `--layer`, `--pos-weight-cap`). No args prints help.
- **Memory:** features are encoded once into `<dataset>/_cache_mert` then
  **streamed from disk per batch**, so the dataset does *not* need to fit in
  RAM (only a few batches are resident). Keep the dataset on the **SSD**.
- **STAR instead of E-GMD:** mount the STAR root, a writable checkpoint dir,
  and pass `--dataset star`:
  ```bash
  docker run --rm --gpus all \
      -v /mnt/ssd/star_full:/data/star -e DRUMJOT_STAR=/data/star \
      -v /mnt/ssd/checkpoints:/out \
      drumjot-trainer --dataset star \
        --train-clips 6000 --val-clips 150 \
        --epochs 80 --batch-size 16 --out /out/star_full_v1
  ```
- **Prefetch workers (`--num-workers N`)** overlap SSD reads with GPU compute,
  but pass batches through `/dev/shm`, which Docker caps at **64 MB** by
  default → the DataLoader **hangs**. To use workers, add **`--shm-size=2g`**
  (or `--ipc=host`) to `docker run` *and* pass e.g. `--num-workers 8`. The
  default `--num-workers 0` still streams from the SSD (RAM stays bounded),
  just without the prefetch overlap, and never hangs.
- To move the built image without rebuilding: `docker save drumjot-trainer | zstd > trainer.tar.zst` on one box, `zstd -d | docker load` on the other (it's a multi-GB image; rebuilding on the target is usually easier).

## Status & findings

The frozen-MERT + per-lane-head approach **works**. Verified on real data:

- **Trained checkpoints** (STAR, sandbox 1660): on held-out STAR val, kick/snare
  reach ~0.99/0.97; mid lanes (toms, hats) 0.6-0.8.
- **Drum stems >> full mix:** training+evaluating on isolated drum stems beats
  the full mix on every lane (kick 0.92→1.0, snare 0.86→0.97). So the model
  should run on **separated drums**, not the mix.
- **Class-balanced data fixes rare lanes:** STAR's near-zero ride/cymbal data
  gave a useless ride threshold (0.30) + flooding; the balanced extract (real
  ride/cymbal coverage) lifts crash and re-tunes ride's threshold to 0.80.
- **ParaDB eval (real songs, hand-charted):** kick 0.90, snare 0.83, open-hat
  0.78 on real separated drums. The model is **promiscuous** (27-62% of onsets
  fired on a clean single-instrument stem land in the wrong lane: hi-hat→ride,
  everything→`mp`); per-instrument isolation discards those, but the deterministic
  envelope filter can't (the false onsets sit on real transients).

Open / next:

1. **Ride + cross-instrument promiscuity:** the worst gap. Likely a
   synthetic-stem → real-separated-stem domain gap; train on STAR's
   `original_drum` (real isolated drums) or on our own separator's output.
2. ~~**`mp`/`mc`:** effectively noise lanes here.~~ DONE (2026-06): both lanes
   removed (no per-stem target, scored ~noise); ride bell folded into `rd`.
3. **Encoder A/B:** MERT layer sweep; MusicFM as the clean-license alternative
   behind the same `encode` interface.
4. **Transcriber integration:** `app/pipeline/learned_onsets.py` is the
   spike that runs a checkpoint as a pipeline stage; wire it behind a config
   flag for production use.

## A/B testing traps (read before any loss/recipe comparison)

Hard-won, cost real false conclusions. When comparing two training recipes
(focal-vs-bce, oversample, a new feature, …):

1. **Always compare at per-lane TUNED thresholds, never a shared/fixed one.** The
   peak-pick height is per-lane and must be re-tuned per arm. Calibration-shifting
   losses (focal fires LOWER than BCE) look tied-or-worse at a fixed threshold and
   only reveal their win at their own optimum. Real misses this caused: focal
   *closed-hat* looked tied (0.68=0.68) at the during-training fixed threshold but
   was **+0.030** tuned (opt thr 0.45 vs bce 0.90); focal *ride* looked **worse**
   untuned (0.60 vs 0.62) but was **+0.065** tuned. The during-training keep_best
   val metric is at a FIXED threshold; it is NOT a valid cross-recipe comparison,
   only an each-arm convergence signal.
2. **Always report EVERY trained lane, not just the lanes of interest.** A cym-only
   table (`cymbal_recall_confusion.decompose` reports rd/cr) hid focal's closed-hat
   win entirely, hc was trained and tuned but never printed. `cymbal_loss_ab.py`
   now prints an all-lane tuned-F1 table (`_all_lane_f1` → `evaluate_clip` at tuned
   thresholds, per-stem isolation). Use it / copy it; don't trust a partial table.
3. **Score with per-stem isolation** (a lane scored only on clips that carry its
   onsets); counting a lane's firing on stems where it's absent is cross-stem
   leakage, a separate metric, and inflates false positives ~2× (skews the optimal
   threshold high). Matches `decompose`'s `if not len(ref): continue`.
4. **Per-lane loss is supported and is the right default** when a loss helps some
   lanes and hurts others: `train_loop(focal_lanes=[…])` trains those lanes with
   focal and the rest with BCE (heads are independent, so no cross-lane effect).
   Decide the map by re-scoring saved checkpoints per lane
   (`cymbal_lane_loss_pick.py`), not by guessing. Current data-driven map:
   focal on `hc,rd`; BCE on `hp,ho,cr,k,s,ss,t`.
