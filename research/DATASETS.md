# Public ADT datasets, catalog & "where to get more data"

Reference for when the drum-onset model needs more training data. Captures the
public Automatic Drum Transcription (ADT) datasets, their size/type, whether we
already use them, and, most importantly, **which are worth pulling in next** (it's
lopsided; don't just "add everything").

## What we train on now

The pooled per-stem training set draws from **STAR + ENST + E-GMD**, all as
**separation-aware per-stem extracts** (`star_balanced_sep`, `enst-sep`, `egmd_sep`;
built by `training/scripts/separate_*_dataset.py`). Current cap-3000 mix (by clips):
**STAR ~71% / ENST ~19% / E-GMD ~9%**. The full aligned pool is **9,420 stems /
15,952 windows**; cap-3000 = 6,559 windows = **~41% of that**.

But that pool is itself a *tiny extract* of the datasets below, by audio-hours,
cap-3000 is only **~1–4% of all public ADT training data** (~1,300–1,500+ h). There
is ~50–100× headroom, but most of it is synthetic (low value, see below).

## The universe (public ADT datasets)

| dataset | ~hours | type | we use? | access |
|---|---|---|---|---|
| **SDDS** (Cartwright & Bello 2018) | 467 | synthetic | ✗ | DAFx-18 release |
| **E-GMD** (Magenta) | ~444 | real e-kit recordings | tiny subset (196 clips) | Magenta, open |
| **TMIDT** (Vogl 2018) | 259 | synthetic | ✗ | on request |
| **Slakh2100** | ~145 | synthetic multitrack | ✗ | open (Zenodo) |
| **ADTOF** (Zehren 2021/23) | 114 | **real**, crowdsourced charts | ✗ | open (github.com/MZehren/ADTOF) |
| **STAR** (Weber, ISMIR) | large (synthetic) | balanced extract only | partial | open |
| **A2MD** | tens | crowdsourced real | ✗ | github (Sma1033/adt_with_a2md) |
| **RBMA-13** | 1.72 | real (EDM/jazz) | ✗ | paid (Bandcamp), skip |
| **ENST-drums** | 1.02 | real (3 drummers) | ~full | open |
| **MDB-Drums** | 0.35 | real (MedleyDB subset) | **eval only** | open (github.com/CarlSouthall/MDBDrums) |

(ENST/MDB/RBMA/SDDS/TMIDT/ADTOF hours: ADTOF paper Table 1. E-GMD/Slakh: their
releases. STAR/A2MD: approximate.)

## Which to pull when we need more, priority order

The headroom is lopsided. **More data ≠ better blindly**: our own RESULTS showed
synthetic STAR *plateaus on cymbals* (the synthetic→real gap), and crash is
**data-bound** (more REAL data lifts it). So:

1. **ADTOF (114 h, REAL), the prime target.** It's the current SOTA training set,
   real audio, and uses the *same crowdsourced-chart approach as our ParaDB pipeline*
   (`rlrr.py`), so parsing it is the shape we already handle. Best bang for the buck
   on the weak hat/cymbal lanes. Open on GitHub. **CAVEAT (cymbal taxonomy):** the
   DEFAULT ADTOF build is **5-class** and LUMPS ride + crash into ONE cymbal class
   (`LABELS_5TXT = [BD, SD, TT, HH, "CY+RD"]`, `config.py`), so it adds NO ride-vs-crash
   signal by default; it lifts the *crash* lane (we map the merged class to `cr`). To
   get real ride/crash separation you must rebuild with `automaticGrooming.py -t 7`
   (task=7 splits CY/RD); the loader then routes ride->rd, crash->cr automatically
   (its pitch map is a task-agnostic superset). See "Acquiring ADTOF audio" below +
   the `adtof.py` module docstring (loader + mapping table already wired).
2. **More of E-GMD / full STAR**, we sample <1% of E-GMD. Cheap to scale (already
   wired: `egmd_sep`/`star_balanced_sep`). Caveat: E-GMD is electronic-kit
   (domain-shifted from acoustic cymbals); good for kick/snare/hat *density*, weaker
   for the acoustic cymbal problem.
3. **A2MD**, crowdsourced real, smaller; similar chart-derived approach.
4. **Synthetic megasets (SDDS 467 h + TMIDT 259 h + Slakh 145 h ≈ 870 h), low
   priority.** Big, but synthetic; expected to plateau the hard lanes like STAR did.
   Only worth it if a specific lane is *quantity*-starved in a way real data can't fix.
5. **RBMA, skip** (paid-only; the one dataset without a free source).

**MDB-Drums** stays an **eval** set (pristine, never in train/val), don't move it to
training or we lose the clean SOTA-comparison benchmark.

## How to add one (the pattern)

Each new mix dataset becomes a per-stem tree like `enst-sep`/`mdb-sep`:
1. Loader in `training/drumjot_training/<name>.py` (annotation → 9-lane onsets;
   `mdb.py` is the template; native parser, no `mirdata` needed).
2. `training/scripts/separate_<name>_dataset.py` (full_mix → BS-Roformer → MDX23C
   per-stem; `separate_mdb_dataset.py` is the template; needs `MODELS_DIR`).
3. Wire into the training pool (`build_specs`) and/or `sota_eval.py` + the GT
   cleanliness probe (run that FIRST on any new set, see `eval_gt_cleanliness.py`).

## Acquiring ADTOF audio (PARKED, needs the user)

The code side of ADTOF is **done and validated** (CPU, no GPU): loader
`training/drumjot_training/adtof.py`, separator
`training/scripts/separate_adtof_dataset.py`, and the `adtof` source wired into
`train.py`'s `--pool-sources`. What's **parked** is the data pull, ADTOF audio is
**NOT redistributed** (the songs are copyrighted rhythm-game charts). Two routes:

### Route A, request the prebuilt dataset (mel-spectrograms only)
The ADTOF authors share the built datasets (359 h) **on request** via Zenodo
(zenodo.org/doi/10.5281/zenodo.10084510). **BUT it ships as mel-scale
spectrograms, not audio.** Our pipeline needs raw audio (MERT encodes waveforms,
and we run our own BS-Roformer/MDX23C separation), so the Zenodo spectrograms are
**not directly usable**, this route is a dead end for us unless they also provide
audio. Mentioned only so it isn't re-investigated.

### Route B, build it yourself from charts (the real route)
Reproduces the dataset from crowdsourced charts you supply (this is what the
ADTOF repo's cleansing pipeline is for). Steps (from the ADTOF README §2-4):

1. **Get charts.** Download drum charts from a site like Rhythm Gaming World
   (`rhythmgamingworld.com`). These are copyrighted; sourcing is the user's call.
2. **Convert to PhaseShift format** (a folder per song with `song.ogg` +
   `notes.mid`) using **C3 CON Tools** "Phase Shift Converter" (Windows GUI).
3. **Build (the cleansing pipeline).** In a clone of `github.com/MZehren/ADTOF`
   (Python 3.10), `pip3 install .`, then:
   ```
   # default 5-class build (ride+crash MERGED -> our `cr` lane):
   python bin/automaticGrooming.py -p <phaseshift_charts_dir> <adtof_built>
   # OR a 7-class build for REAL ride/crash separation (recommended for us):
   python bin/automaticGrooming.py -p -t 7 <phaseshift_charts_dir> <adtof_built>
   ```
   Output tree (what our loader reads): `<adtof_built>/audio/audio/<track>.ogg` +
   `<adtof_built>/annotations/aligned_drum/<track>.txt`.

### Then (our side, GPU): separate + train
Once `<adtof_built>` exists, run the same per-stem flow as the other sets:

```
# 1. separate full song -> per-stem tree (BS-Roformer -> MDX23C). Needs MODELS_DIR.
MODELS_DIR=/codebox-workspace/drumjot/models-cache \
scripts/sandbox-run env PYTHONPATH=transcriber:training:dsp \
  python3 training/scripts/separate_adtof_dataset.py \
    /codebox-workspace/datasets/adtof_built /codebox-workspace/datasets/adtof-sep

# 2. (recommended) GT cleanliness probe on the new set BEFORE training:
scripts/sandbox-run env PYTHONPATH=transcriber:training:dsp \
  python3 training/scripts/eval_gt_cleanliness.py ...   # adapt to adtof-sep

# 3. point the env var at the sep tree and add `adtof` to --pool-sources. The
#    loader is task-agnostic: a task=7 tree routes ride(51)->rd, crash(49)->cr;
#    a task=5 tree has only the merged 49->cr (no rd signal). No flag needed.
DRUMJOT_ADTOF=/codebox-workspace/datasets/adtof-sep \
... python3 -m drumjot_training.train --dataset pooled \
    --pool-sources star,enst,egmd,adtof --pool-cap <N> ...
```

`adtof.py` maps the reduced-MIDI pitches in `aligned_drum/*.txt` to lanes:
BD 35->k, SD 38->s, TT 47->t, HH 42->hc, and the cymbal class 49->cr (task=5
merged; task=7 crash). task=7 also gives OH 46->ho and RD 51->rd. No `ss` lane
from ADTOF (it folds side stick into snare upstream).

## Sources

- ADTOF dataset table (sizes): [ISMIR 2021](https://archives.ismir.net/ismir2021/paper/000102.pdf)
- MDB-Drums: github.com/CarlSouthall/MDBDrums · ADTOF: github.com/MZehren/ADTOF
- A2MD: github.com/Sma1033/adt_with_a2md
