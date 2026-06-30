# Stem-separation models: research summary & options

Handoff doc for work on **upgrading / retraining the source-separation
stages** (STEMS_ALL and STEMS_PER). Date: 2026-06-30. This is research +
options, not a committed plan. Pairs with [DATASETS.md](DATASETS.md).

## Current setup (two PyTorch models, in-graph STFT)

The transcriber's two separation stages (`transcriber/app/pipeline/separate.py`,
invoked via the `audio-separator` package):

1. **STEMS_ALL**, full mix → drum stem. Model: **BS-Roformer "SW"**
   (6-stem instrument splitter: bass/drums/other/vocals/guitar/piano; we
   keep only `drums`). Config `config_bs_roformer_sw.yaml`: dim 256, depth
   12, heads 8, 62-band split, flash_attn. **STFT = n_fft 2048, hop 512,
   win 2048** (read from `model.stft_*`; the `audio.hop_length: 441` field
   is a decoy annotated "don't work (use in model)"). Weights
   `model_bs_roformer_sw.ckpt` (668 MB).
2. **STEMS_PER**, drum stem → per-instrument. Model: **MDX23C DrumSep
   (jarredou 5-stem)**: kick/snare/toms/hh/**cymbals** (ride+crash merged).
   Config: TFC-TDF, n_fft 2048, hop 512, dim_f 1024, dim_t 1024,
   InstanceNorm, num_scales 5. Weights `drumsep_5stems_mdx23c_jarredou.ckpt`
   (418 MB). Also in-graph STFT.

Both on disk at `/codebox-workspace/drumjot/models-cache/`. **Upstream is
gone**, jarredou's GitHub was deleted; HF mirrors (Politrees, Sucial,
lainlives for the YAML) + our local copies are the only source. **Vendor
these to our own storage now.**

## Key finding: STEMS_ALL is already SOTA-open, leave it

mvsep multisong leaderboard, drums SDR (open single models):
**BS-Roformer SW = 14.11 SDR, #1 open.** Variants (RA/RAv2/Logic/OA-Residual)
are 14.01–14.08 (noise). Only proprietary 3-model ensembles beat it
(~14.33–14.35, +0.2), and they *include* SW as a component. → **No open
upgrade worth taking for STEMS_ALL.** Full-mix→drums is saturated at ~14 SDR.

## The real opportunity: STEMS_PER (per-instrument), esp. cymbals/hats

Per-instrument SDR is where quality is lost. mvsep DrumSep leaderboard:

| Model (6-stem) | kick | snare | toms | cymbals | hh | ride/crash | Open? |
|---|---|---|---|---|---|---|---|
| MDX23C (ours is the 5-stem sibling) | 18.3 | 13.6 | 13.3 | 6.7 | 5.4 | 7.6 | **Yes** |
| SCNet-XL | 20.2 | 14.8 | 15.9 | 6.7 | 5.0 | 7.6 | Proprietary |
| **Mel-Band Roformer** | 20.2 | 15.3 | 15.5 | **8.8** | **7.0** | **8.8** | **Proprietary** |
| Ensemble | 20.6 | 15.1 | 16.4 | 7.2 | 5.6 | 7.9 | Proprietary |

The **Mel-Band Roformer DrumSep is the best for our hard lanes (hh, ride,
crash) but is mvsep-proprietary** (absent from ZFTurbo's open
`pretrained_models.md`; present only on the mvsep service). hh/cymbal
separation is the universal weak point.

## Options for STEMS_PER

### A. Keep open MDX23C 5-stem + lean on downstream models
What we do today. Our learned onset model + hi-hat articulation model
already do per-lane work; ride/crash disambiguation happens downstream.
Cheapest. The cymbal/hat SDR gap is the ceiling.

### B. Train our own Mel-Band Roformer (or BS-Roformer) DrumSep
The **architecture is fully open** (lucidrains MelBand/BS-Roformer +
**ZFTurbo's `Music-Source-Separation-Training`** is the actual trainer);
only mvsep's *trained weights* are withheld. This is the path to
mvsep-grade cymbal/hat separation that we'd **own** (and can ONNX-export,
and emit our exact lane split incl. separate ride/crash).

- **Architecture lineage**: MDX23C (TFC-TDF-UNet v3, conv, SDC-2023 winner)
  is prior-gen; BS-Roformer (Sept 2023) and **Mel-Band Roformer** (Oct
  2023, ByteDance) are the newer band-split + RoPE-transformer SOTA.
  MelBand > BS-Roformer for drums.
- **Fine-tuning vs scratch**: ZFTurbo supports `start_check_point`. There
  is **no open *drum* Roformer to fine-tune**, you'd fine-tune from a
  general/vocal/4-stem MelBand checkpoint and **reinit the output head**
  to N drum stems (transfer learning), or train from scratch. Could also
  reinit-head from our own BS-Roformer-SW (its trunk already models drums).
- **Compute / VRAM**: original MelBand trained on 16×V100-32GB. ZFTurbo
  configs target a single A6000 48GB but say reduce batch + raise
  grad-accum for less. **A 5090 (32GB) is sufficient for fine-tuning**
  (small batch + accumulation + bf16; LoRA option exists); 32GB is the
  same per-GPU class the authors used. From-scratch large-config is more
  comfortable at 48GB but works on 32GB (slower). 5090's compute >> A6000.

### C. mvsep paid API
Best quality without training, but breaks local/offline/multi-backend, non-starter for the desktop app.

## DATA, the gating constraint (read carefully)

Source-separation training needs **ground-truth (mixture → isolated-stem)
audio pairs**, NOT onset/transcription labels.

**We do NOT currently have ground-truth per-instrument stems on disk.** The
`*_sep` datasets under `/codebox-workspace/datasets/` (egmd_sep, enst-sep,
star_*_sep, paradb-sep, …) are the **OUTPUTS of the current
BS-Roformer→MDX23C separators**, distillation targets for the onset model,
**circular** for separator training (you'd cap at current quality).

Candidate ground-truth sources:
- **OPEN QUESTION (resolve first)**: is **STAR** (synthetic) rendered with
  ground-truth per-instrument stems, or are `star_*_sep` also separator
  outputs? If STAR emits rendered-truth stems, that's StemGMD-style DrumSep
  training data **already on disk (21–169 GB)**, the cheapest path. Check
  the STAR synthesis pipeline.
- **StemGMD** (Zenodo, open, **not on disk**): 1,224 h, GMD MIDI rendered
  through 10 Logic acoustic kits, isolated 9-piece stems (→ supports
  separate hh/ride/crash). ~1.13 TB unzipped (collides with our SSD limit +
  NFS-HDD stall issues). **Synthesis realism vs E-GMD**: comparable tier
  (both sample-playback of the same GMD performances; StemGMD's acoustic
  timbres arguably richer, E-GMD has more kit variety via TD-17). The
  decisive gap for *separation* is that StemGMD stems are **bleed-free** and
  its mixtures are clean digital sums; real drum buses have heavy mic
  bleed. Plan to **synthesize bleed** (room IRs, inter-mic crosstalk, bus
  processing) and/or blend real multitrack, or the separator overfits to
  unrealistically clean mixes.
- **Real multitrack drum recordings** (best domain match, scarce). What we
  have that's real: ENST, A2MD, ParaDB, MDB; but these are mixes + labels,
  not isolated per-drum-instrument stems.
- For STEMS_ALL (moot, but for reference): full-song multi-instrument stems
  = **MUSDB18HQ** (not on disk).

## Recommended next steps for the separation work

1. **Vendor the 4 model files** (both ckpt + yaml) to our own HF/storage;
   jarredou is gone.
2. **Resolve the STAR-stems question**, determines whether DrumSep
   training data already exists locally.
3. If pursuing option B: scout an open MelBand 4-stem base checkpoint,
   define the head-reinit fine-tune, and a bleed-augmentation recipe for
   any synthetic data.
4. STEMS_ALL: no action beyond vendoring.

## Pointers
- Trainer: `github.com/ZFTurbo/Music-Source-Separation-Training` (configs,
  open `docs/pretrained_models.md`).
- Arch: `github.com/lucidrains/BS-RoFormer`; papers arXiv 2309.02612
  (BS-Roformer), 2310.01809 (Mel-Band Roformer).
- Drum-sep benchmarks: arXiv 2509.24853, IEEE 10704147; StemGMD/LarsNet
  arXiv 2312.09663 (Zenodo record 7860223).
- Leaderboards: mvsep.com/quality_checker/multisong_leaderboard?sort=drums,
  mvsep.com/algorithms/29 (DrumSep).
- On-disk data map: see `AGENTS.md` → "Data & models on disk".
