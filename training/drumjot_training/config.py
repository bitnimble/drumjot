"""Training configuration for the drum-onset model (design spec §4)."""
from __future__ import annotations

from dataclasses import dataclass

from drumjot_training.lanes import LANES


@dataclass(frozen=True)
class Config:
    # Frozen SSL encoder. MERT-v1-330M is CC-BY-NC (fine per the data
    # owner); MusicFM (MIT/Apache) is the clean alternative. See spec §4.
    encoder: str = "m-a-p/MERT-v1-330M"
    encoder_layer: int = 10  # N2N's pick; tunable
    encoder_fps: float = 75.0  # MERT feature rate (~13 ms); verify per encoder
    # On-disk feature cache precision. float16 halves cache size + per-epoch
    # read bandwidth (so it fits the OS page cache) at no real cost; fp32 path
    # autocasts to bf16 anyway. See embeddings.embed_clip.
    cache_dtype: str = "float16"

    # Targets (Gaussian onset bumps).
    lanes: tuple[str, ...] = LANES
    sigma_frames: float = 1.5

    # Peak-pick + eval.
    peak_threshold: float = 0.5
    peak_min_distance_s: float = 0.03
    onset_tolerance_s: float = 0.05  # +/-50 ms, standard ADT tolerance

    # Per-lane head.
    head_hidden: int = 128
    head_layers: int = 2

    # Optimisation.
    lr: float = 1e-3
    # Padded mini-batch of full-length (~30s/2250-frame) clips. Measured peak
    # ~520 MiB/clip FP32 + ~226 MiB base, so 16 fits a 10GB card (~8.5GB) with
    # headroom; the 3080's bf16 path uses ~half, leaving room to push higher
    # via --batch-size. Safe in either precision.
    batch_size: int = 16
    max_epochs: int = 50
