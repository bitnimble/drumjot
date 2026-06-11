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
    # Append the 6-20 kHz high-band block (embeddings.highband_features) to the
    # MERT features (model input width = FEAT_DIM 1040 vs MERT_DIM 1024). On by
    # default; --no-high-band trains on raw MERT only, for the high-band ablation.
    # Part of the feature-cache key (variant), so on/off caches don't collide.
    high_band: bool = True
    # Append the sub-6 kHz ride/crash/hi-hat timbre block (embeddings.cym_features,
    # +CYM_BANDS dims). Off by default; --cym enables it for the cymbal ablation.
    # Also part of the feature-cache-key variant.
    cym: bool = False

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

    # Optimisation. AdamW (decoupled weight decay) + a warmup->cosine LR
    # schedule; both are strict improvements over plain Adam/constant-LR for a
    # fixed-epoch run (the cosine decay also lands the final, threshold-tuned
    # epoch at the LR minimum). See train.train_loop.
    lr: float = 1e-3
    weight_decay: float = 0.01
    # Sibling-aware loss weighting (lanes.CONFUSABLE): frames where a confusable
    # sibling lane is active get their loss scaled, hard NEGATIVES (sibling hit,
    # this lane silent -> punish false triggers on bleed) by `sib_neg_weight`,
    # and co-occurring POSITIVES (genuinely simultaneous hits, the harder
    # detection) by `sib_pos_weight`. 1.0 disables either term. On by default;
    # values are starting guesses, not tuned.
    sib_neg_weight: float = 8.0
    sib_pos_weight: float = 3.0
    # Auxiliary ring-activity objective (targets.SUSTAINED_LANES): joint BCE on
    # "is this instrument still ringing" frames, weighted by this factor. The
    # open-hat / cymbal tail is what defines those classes; the pure onset
    # target never shows it to the head.
    aux_act_weight: float = 0.5
    # Threshold tuning: lanes with fewer than `rare_lane_min_onsets` val onsets
    # get their tuned peak threshold floored at `rare_thr_floor` (a 4-clip val
    # lane once tuned ride to 0.10 and flooded real audio; see RESULTS.md).
    rare_lane_min_onsets: int = 50
    rare_thr_floor: float = 0.3
    # Padded mini-batch of full-length (~30s/2250-frame) clips. Measured peak
    # ~520 MiB/clip FP32 + ~226 MiB base, so 16 fits a 10GB card (~8.5GB) with
    # headroom; the 3080's bf16 path uses ~half, leaving room to push higher
    # via --batch-size. Safe in either precision.
    batch_size: int = 16
    max_epochs: int = 50
