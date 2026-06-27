# Learned-onset checkpoint (the default onset backend)

The transcriber's default onset detector is the trained frozen-MERT +
per-lane-heads model (`training/`, run per stem by
`app/pipeline/learned_onsets.py`). It loads a **run directory** containing:

- `model.pt`, the trained weights.
- `meta.json`, encoder name/layer, the lane list, and the **tuned per-lane
  thresholds** (e.g. the patched `cr` operating point). The thresholds live
  here, separate from the weights, so retuning a lane is a one-number edit.

## Baking it into the Docker image

`docker/Dockerfile` copies this whole `checkpoints/` directory into the image
and sets `LEARNED_ONSETS_CHECKPOINT=/app/checkpoints/learned_onsets`, so just
drop the run dir's files here **before building**:

```sh
# from the repo root, using the current full-kit checkpoint:
cp /codebox-workspace/datasets/ab3_prev/model.pt  transcriber/checkpoints/learned_onsets/
cp /codebox-workspace/datasets/ab3_prev/meta.json transcriber/checkpoints/learned_onsets/
```

`model.pt` (≈85 MB) and `meta.json` are git-ignored, they're a per-deploy
artifact, not source.

## If you don't bake a model

The image still builds with this directory empty. At runtime the learned stage
detects the missing checkpoint and **falls back to the ADTOF backend** (whose
weights ship inside the `adtof_pytorch` wheel), logging a warning. To force
ADTOF explicitly instead, set `USE_LEARNED_ONSETS=false` (or pass
`onset_backend=adtof` per request).

## Overriding the path

Point `LEARNED_ONSETS_CHECKPOINT` (env var / `transcriber/.env`) at any other
run dir to use a different model without rebuilding, e.g. bind-mount a
checkpoint and set the env to its in-container path.

## MERT weights

The frozen MERT-v1-330M encoder is **not** baked; it downloads from the
Hugging Face hub on first use into `HF_HOME` (`/models/huggingface`, a
persisted volume), the same way the separation models provision. The first
learned transcribe after a fresh `/models` volume therefore pulls ~1.3 GB once.
