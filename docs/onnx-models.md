# ONNX models: format, conversion, adding + A/B-testing new ones

Every ML model the transcriber runs at inference is **torch-free ONNX on
onnxruntime** (separation, learned onsets, ADTOF, beats, lyrics). Torch is used
ONLY to export the `.onnx` once (build time) and for the `DRUMJOT_*_ONNX=0`
opt-out fallbacks. This doc is what you need to convert a new model to our format,
ship it, and A/B it against an incumbent.

Related: [[onnx-fp16-shipping]] memory (the short version), `config.py` "Model
asset sources", `provision.py`, [docs/transcriber-pipeline.md](transcriber-pipeline.md).

## The shipping format

- **fp16 ONNX.** fp16 halves the download and unlocks GPU tensor-core / NPU fp16
  execution. It is the format on HF `bitnimble/drumjot-onnx` and what the app
  downloads.
- **GPU EPs only.** ORT's CPU EP has no working fp16 GRU kernel, the onset heads
  + ADTOF **segfault** on the CPU EP at fp16. A CPU-only deployment needs the
  fp32 set (we don't ship one; we don't support the CPU EP). See the CUDA-lib
  preload gotcha below, without it, onnxruntime-gpu silently falls back to CPU
  and you hit exactly this crash.
- **Not fp8 / int8.** fp8 needs Ada/Hopper tensor cores (useless on our Ampere
  3080); int8 is ~2× smaller but risks accuracy on the GRU/CTC parts and needs
  the TensorRT EP, not worth it vs fp16.

## Anatomy of a shipped model

Each model is THREE things, not just a `.onnx`:

1. **The ONNX body** (`{name}.fp16.onnx`), the neural net, exported from torch.
2. **A numpy inference frontend**, the model's pre/post-processing reproduced in
   **pure numpy** so inference needs no torch. This is the hard part and the
   thing to get bit-right (see validation). Examples:
   - separation: `separation/np_stft.py` (STFT/iSTFT bit-compatible with
     `torch.stft`), `separation/np_inference.py::NumpySeparator`.
   - beats: `beat_onnx.py` (numpy log-mel matching torchaudio + Beat This!
     chunking/peak-picking).
   - onsets: `onset_onnx/np_onsets.py` (MERT truncation + per-lane peak-pick).
   - lyrics: `lyrics_onnx.py` (`generate_emissions_np`, vendored CTC align).
3. **Sidecar files** the frontend reads at load, shipped ALONGSIDE the onnx:
   - separation: the architecture **yaml** (`config_*.yaml`), STFT params
     (`n_fft`/`hop`/`dim_f`/`dim_t`), instrument names, roformer band splits. The
     onnx body can't run without it.
   - learned onsets: **`onset_meta.json`**; lane vocab, tuned per-lane
     thresholds, fps, `encoder`/`encoder_layer`/`in_dim`.

## Converting a model to our format

The export lives per-model under `transcriber/app/pipeline/**/export*.py` and is
driven by `scripts/export_onnx_models.py`. The recipe:

```python
# 1. torch.onnx.export the inference body (a thin nn.Module wrapping the model).
torch.onnx.export(
    body, (dummy,), str(out_path),
    input_names=[...], output_names=[...],
    dynamic_axes={"input": {1: "samples"}, "output": {1: "frames"}},  # per-axis names
    opset_version=17, do_constant_folding=True,
    dynamo=False,   # REQUIRED: torch 2.11 defaults to the dynamo exporter, which
                    # pulls onnxscript and changes the graph; we use the legacy one.
)
# 2. fp16-convert in place (see onnx_fp16.to_fp16):
from onnxruntime.transformers.float16 import convert_float_to_float16
onnx.save(convert_float_to_float16(onnx.load(p), keep_io_types=True), p)
```

Non-obvious bits:

- **`keep_io_types=True`** leaves the graph's inputs/outputs **fp32** (Cast nodes
  at the boundary), so the numpy frontend feeds/reads fp32 unchanged; only
  internal weights/compute go fp16. Don't drop it, otherwise every numpy caller
  has to fp16-cast at the seam.
- **`dynamo=False`**, see above; the legacy exporter is what our graphs + the
  fp16 converter expect.
- **opset 17**, `do_constant_folding=True` across the board.
- **Export the *inference* subgraph, not the training model.** The `MertBody` /
  `HeadsBody` wrappers exist to expose exactly the tensor the numpy path needs
  (e.g. MERT truncated to `L` blocks with the final layer-norm neutralised so the
  output equals `hidden_states[L]` bit-exact, one clean output, ~L/24 the
  compute). Match batch=1 / no-mask / no-pack to whatever the numpy frontend does.
- Export runs on **CPU**, sequential with `gc.collect()` so peak RAM ≈ one model.

### Validating a conversion

The bar is **corr ≥ 0.99998 vs fp32 on CUDA** for the full path (numpy frontend +
onnx body) against the original torch model on the same input. Two things can
drift: the fp16 rounding (tiny) and the numpy frontend not matching torch's
prep (can be large; this is where bugs hide). Validate BOTH the fp32 onnx (numpy
frontend correctness) and the fp16 onnx (rounding) before shipping. The
`DRUMJOT_*_ONNX=0` torch path is the reference to diff against.

## Adding a new model, checklist

1. **Export fn** under the model's package (`export_*`, `fp16: bool` param calling
   `onnx_fp16.to_fp16`), and wire it into `scripts/export_onnx_models.py` under a
   `--only` tag.
2. **Numpy frontend + loader.** Reproduce prep in numpy; the loader builds an ORT
   session and **prefers the provisioned fp16** via `provision.shipped_onnx(name)`
   (→ `{models_dir}/{name}.fp16.onnx`), falling back to a local export only in a
   dev checkout. Gate it behind a `DRUMJOT_<X>_ONNX` env (default ON; `0` = torch
   fallback). Providers come from `_onnx_providers()` (CPU-pinned only when
   `settings.device` is cpu/mps, else the default GPU set).
3. **Provision under ONE capability.** Add the asset to `provision._<cap>_assets`
   for the single capability that uses it, NEVER a global "fetch all"
   ([[capability-scoped-downloads]]). `transcription` and `lyrics` compose
   `separation`.
4. **Asset source in `config.py`**, not hardcoded. URLs/HF ids are `settings.*`
   build fields (`onnx_repo`, `lyrics_align_model_*`, …) so a build can repoint
   them. No naked URLs in loaders.
5. **Upload** `{name}.fp16.onnx` (+ any sidecar) to HF `bitnimble/drumjot-onnx`
   and add its attribution to that repo's `LICENSE.md`. Mind licenses, MERT /
   ADTOF / MMS are **non-commercial**; a new model's license propagates to the
   feature that uses it.
6. **Validate** (above), record any GPU eval in `training/RESULTS.md`.

## A/B testing models

The design is built for this; the torch path is literally kept as "the A/B
reference":

- **Swap the onnx, keep the pipeline.** Loaders resolve `shipped_onnx(name)` from
  `settings.models_dir`, so dropping a candidate `{name}.fp16.onnx` into the
  models dir (or provisioning it) makes the loader pick it up with no code change.
  Export both, put them under distinct names, point the loader/env at each.
- **Reference = the torch path.** `DRUMJOT_<X>_ONNX=0` runs the original torch
  model; diff a candidate's output against it (and against the incumbent onnx) on
  the same input. The onset backend is also swappable end-to-end (`learned` vs
  `adtof`), the onset backend only changes the model + its post-processing, the
  rest of the pipeline is backend-agnostic.
- **Keep prep identical across arms**, or you're A/B-ing the frontend, not the
  model. If a candidate needs different prep (different sample rate, mel bins,
  STFT), that's a new numpy frontend, not a drop-in.
- Score real transcriptions, not just tensor corr, see the beat-tracker A/B
  harness (`transcriber/benchmarks/beat_ab.py`) and the ParaDB/MDB onset evals
  for the shape of a proper A/B (per-lane F1 at tuned thresholds, cross-checked on
  a second corpus so you don't overfit a tiny eval set).

## Gotchas / invariants

- **fp16 is GPU-only** (CPU EP segfaults on fp16 GRU). If a model has GRUs/LSTMs
  and must run on CPU somewhere, it needs an fp32 variant.
- **CUDA-lib preload is load-bearing.** The torch-free runtime never loads torch's
  CUDA libs, so onnxruntime-gpu can't `dlopen` its CUDA provider
  (`libcublasLt.so.12 not found`) and SILENTLY falls back to CPU → the fp16 GRU
  segfault. `onnx_cuda.preload_cuda_libs()` (called at `app/sidecar.py::main`)
  RTLD_GLOBAL-preloads the venv's `nvidia/*/lib` so the provider resolves, no
  `LD_LIBRARY_PATH` needed. This is Linux-only today; **Windows + NVIDIA has the
  same bug uncovered** (needs `os.add_dll_directory` over `nvidia/*/bin`), macOS
  uses CoreML/MPS (no CUDA), Android has no sidecar (HTTP backend).
- **Ship the sidecar files.** A separation onnx without its yaml, or onset heads
  without `onset_meta.json`, loads but can't run / mislabels lanes.
- **The numpy frontend is the contract**, not the onnx. Bit-compatibility with
  the model's original prep (STFT windowing, mel filterbank, normalization,
  chunking) is what keeps fp32-onnx corr at rounding level; get it wrong and the
  fp16 number hides a real bug.

## Reference

| Model | onnx name | sidecar | loader | env flag |
|---|---|---|---|---|
| separation (BS-RoFormer) | `model_bs_roformer_sw.fp16.onnx` | `config_bs_roformer_sw.yaml` | `NumpySeparator` | `DRUMJOT_SEP_ONNX` |
| separation (MDX23C 5-stem) | `drumsep_5stems_mdx23c_jarredou.fp16.onnx` | `config_drumsep_5stems_mdx23c.yaml` | `NumpySeparator` | `DRUMJOT_SEP_ONNX` |
| learned onsets (encoder) | `mert_L{layer}.fp16.onnx` | `onset_meta.json` | `np_onsets.load_onnx_onset` | `DRUMJOT_ONSET_ONNX` |
| learned onsets (heads) | `onset_heads.fp16.onnx` | `onset_meta.json` | `np_onsets.load_onnx_onset` | `DRUMJOT_ONSET_ONNX` |
| ADTOF onsets | `adtof_frame_rnn.fp16.onnx` |, | `adtof_onnx.load_adtof_session` | `DRUMJOT_ONSET_ONNX` |
| beats (Beat This!) | `beat_this.fp16.onnx` |, | `beat_onnx.load_beat_session` | `DRUMJOT_BEAT_ONNX` |
| lyrics (CTC align) | `ctc_align__{model}.fp16.onnx` |, | `lyrics_onnx.load_onnx_aligner` | `DRUMJOT_LYRICS_ONNX` |

- Export all: `transcriber/.venv/bin/python3 scripts/export_onnx_models.py [OUT_DIR] [--fp32] [--only sep,onset,adtof,beat,lyrics]`
- fp16 converter: `transcriber/app/pipeline/onnx_fp16.py::to_fp16`
- Provisioning (capability-scoped): `transcriber/app/pipeline/provision.py`
- Asset sources (build settings): `transcriber/app/config.py` "Model asset sources"
- Shipped weights + attributions: HF `bitnimble/drumjot-onnx` (`LICENSE.md`)
