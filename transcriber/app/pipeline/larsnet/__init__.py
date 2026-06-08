"""Vendored LarsNet drum-piece separator (polimi-ispl/larsnet).

LarsNet is five parallel U-Nets that split a drum stem into
kick / snare / toms / hihat / cymbals via spectro-temporal soft masking
(Mezza et al., "Toward Deep Drum Source Separation", arXiv:2312.09663).
It is an opt-in alternative Stage-2 separator to the default jarredou
MDX23C model, same five output classes (ride+crash merged into
`cymbals`), so the rest of the pipeline is unchanged. It runs ~20-40x
faster than MDX23C but separates a bit bleedier (validated: comparable
onset counts through our ADTOF detector, modestly lower cymbal/snare
recall).

The model architecture (`unet.py`) is vendored byte-for-byte from
upstream (no pip package exists). This module holds the Drumjot glue:
config-free model construction and a config-free `separate()`.

**LICENSING**: the upstream *code* carries no explicit license, but the
PRETRAINED WEIGHTS are **CC BY-NC 4.0 (non-commercial)**. `provision.py`
fetches them from the `JosefKuchar/LarsNet` HF mirror (which states the
CC-BY-NC-4.0 license) into `<models_dir>/larsnet/<stem>/`. Any
deployment that ships or serves these weights inherits the
non-commercial constraint, see docs/transcriber-pipeline.md.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .unet import UNetWaveform

log = logging.getLogger(__name__)

# torch / the vendored unet are imported lazily inside `load_models` /
# `separate` (never at module top), mirroring `adtof_onsets.py`. This
# keeps importing `app.pipeline.larsnet` - and so test collection and the
# runner's import graph - free of the heavy torch import (and of any host
# CUDA-lib load failure), exactly as the rest of the pipeline does.

# Output stem name -> Drumjot DSL pitch letter. Aligned with the MDX23C
# lane mapping in `separate.STEM_NAME_TO_PITCH` so LarsNet is a drop-in
# Stage-2 replacement: ride+crash arrive merged as `cymbals` -> `c`, and
# `cymbal_split.py` splits that lane into ride (`d`) / crash (`c`)
# downstream exactly as it does for MDX23C.
STEM_TO_PITCH: dict[str, str] = {
    "kick": "k",
    "snare": "s",
    "toms": "t",
    "hihat": "h",
    "cymbals": "c",
}

# Inference order = the five LarsNet checkpoints. All five share the same
# UNet input size (F=2048 magnitude bins, T=512 frames) and 44.1 kHz rate
# per upstream `config.yaml`; there is no per-stem architecture variation,
# so we hardcode it here rather than vendoring the yaml.
STEMS: tuple[str, ...] = ("kick", "snare", "toms", "hihat", "cymbals")
_INPUT_F = 2048
_INPUT_T = 512
SAMPLE_RATE = 44100


def checkpoint_path(models_dir: Path, stem: str) -> Path:
    """Local path of one stem's checkpoint under the models dir.

    Mirrors the HF mirror's layout (`<stem>/pretrained_<stem>_unet.pth`)
    rooted at `<models_dir>/larsnet/`, see `provision._provision_larsnet`.
    """
    return models_dir / "larsnet" / stem / f"pretrained_{stem}_unet.pth"


def load_models(models_dir: Path, device: str) -> dict[str, UNetWaveform]:
    """Build + load the five LarsNet U-Nets onto `device`.

    Raises `FileNotFoundError` (caught upstream -> StageError -> HTTP 500)
    when a checkpoint is missing; provisioning runs at startup, so this
    only fires on a misconfigured/offline deploy.

    `weights_only=False` is required: the checkpoints are pickled dicts
    (`{"model_state_dict": ...}`) and torch>=2.6 defaults `weights_only`
    to True, which rejects the load. The weights come from a pinned HF
    mirror provisioned by us, so unpickling them is no riskier than the
    other model weights this service loads.
    """
    # Validate every checkpoint exists BEFORE importing torch, so a
    # misprovisioned deploy fails loud without paying the torch import.
    paths: dict[str, Path] = {}
    for stem in STEMS:
        cp = checkpoint_path(models_dir, stem)
        if not cp.exists():
            raise FileNotFoundError(
                f"LarsNet checkpoint missing for '{stem}': {cp}. "
                "Run startup provisioning (settings.provision_larsnet) or "
                "place the CC-BY-NC weights there manually."
            )
        paths[stem] = cp

    import torch

    from .unet import UNetWaveform

    models: dict[str, UNetWaveform] = {}
    for stem, cp in paths.items():
        model = UNetWaveform(input_size=(2, _INPUT_F, _INPUT_T), device=device)
        checkpoint = torch.load(str(cp), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[stem] = model
        log.info("LarsNet: loaded %s U-Net from %s", stem, cp.name)
    return models


def separate(
    models: dict[str, UNetWaveform],
    drum_stem_path: Path,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run the five U-Nets on a drum stem; return per-stem waveforms.

    Loads + resamples the drum stem to 44.1 kHz, runs each stem's U-Net
    over the whole signal (the model folds the spectrogram into fixed-T
    chunks internally, so arbitrary length is fine), and returns a
    `{stem_name: waveform[channels, samples] on CPU}` dict. Default mask
    mode (no alpha-Wiener post-filter, our A/B showed Wiener didn't
    improve separation and slightly cut cymbal recall).
    """
    import torch
    import torchaudio as ta

    x, sr = ta.load(str(drum_stem_path))
    if sr != SAMPLE_RATE:
        x = ta.functional.resample(x, sr, SAMPLE_RATE)
    x = x.to(device)
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for stem, model in models.items():
            y, _mask = model(x)
            out[stem] = y.squeeze(0).detach().cpu()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
