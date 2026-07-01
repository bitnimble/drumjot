"""Training configuration for the drum-onset model (design spec §4)."""
from __future__ import annotations

from dataclasses import dataclass

from drumjot_training.lanes import LANES


@dataclass(frozen=True)
class Config:
    # Frozen SSL encoder. MERT-v1-330M is CC-BY-NC (fine per the data
    # owner); MusicFM (MIT/Apache) is the clean alternative. See spec §4.
    encoder: str = "m-a-p/MERT-v1-330M"
    encoder_layer: int = 10  # N2N's pick; tunable. Also the per-lane FALLBACK layer.
    # Per-lane MERT-layer routing. None => every lane reads `encoder_layer` (the
    # single-layer path, byte-identical to before). A tuple of (lane, layer) pairs
    # routes those lanes to their OWN MERT hidden layer -- the per-lane layer sweep
    # finds e.g. cymbals peak at a later layer than kick, and a single shared layer
    # leaves that on the table. Lanes absent from the tuple fall back to
    # `encoder_layer`. Stored as a tuple (not a dict) so the frozen Config stays
    # hashable. Needs no extra encode: the on-disk cache is already keyed per layer
    # (embeddings.cache_key), so each head just reads its layer's existing .npy.
    lane_layers: tuple[tuple[str, int], ...] | None = None
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
    # Per-clip robust peak-normalise the WAVEFORM (embeddings.robust_peak_normalize)
    # before feature extraction, so MERT (do_normalize=false) + the high-band block
    # see a level-consistent input across songs. Off by default; --input-norm enables
    # it. Part of the feature-cache key (variant "_pn"), so on/off caches don't collide.
    input_norm: bool = False

    # Targets (Gaussian onset bumps).
    lanes: tuple[str, ...] = LANES
    sigma_frames: float = 1.5

    # Peak-pick + eval.
    peak_threshold: float = 0.5
    peak_min_distance_s: float = 0.03
    onset_tolerance_s: float = 0.05  # +/-50 ms, standard ADT tolerance

    # Label-support GATE (per stem-window, against the stem's onset-strength
    # envelope): DROP the whole stem-window if ANY lane's support (fraction of onsets
    # on a transient within +/-window) is below `label_min_support`. **OFF by default
    # (0.0):** an A/B (RESULTS) found the gate HURTS every lane -- its relative
    # support floor over-drops sparse cymbal windows -- and snap-only onsets WITHOUT
    # the gate won. Set --label-min-support >0 to re-enable. See clean.filter_lanes_by_support.
    label_min_support: float = 0.0
    label_support_window_s: float = 0.04
    label_support_percentile: float = 60.0

    # Per-lane head.
    head_hidden: int = 128
    head_layers: int = 2
    # Lanes that get the auxiliary ring-activity objective. None ->
    # targets.SUSTAINED_LANES (ho,rd,cr). --aux-lanes overrides, e.g. `ho,rd` drops
    # crash from the aux objective: the ring/wash objective may be teaching cr to
    # follow the ride wash (false positives), so dropping it is the ride-crash
    # discrimination A/B. Only changes which lanes the aux LOSS supervises; the aux
    # targets themselves are still built for SUSTAINED_LANES (unused rows ignored).
    aux_lanes: tuple[str, ...] | None = None
    # Per-clip onset AUTO-CALIBRATION (model.OnsetHead.calib): a learned per-clip
    # operating-point shift on the onset logit. On by default; --no-auto-calibrate
    # bypasses it (the calib weights are still CONSTRUCTED so the RNG stream --
    # and thus the GRU init -- is identical to a calibrated run, for a clean A/B).
    auto_calibrate: bool = True
    # Joint ride/crash discrimination (experiment): train the cymbal lanes (rd, cr)
    # with a 3-way softmax {none, ride, crash} -- `none` is the fixed-0 reference, so
    # the rd/cr logits are unchanged in shape -- instead of two independent BCE
    # heads. Forces the model to COMMIT to one cymbal type per onset (attacks the
    # ride<->crash both-fire confusion on the merged MDX23C cymbals stem). The SAME
    # softmax is applied to the rd/cr rows at val/threshold-tune/eval (meta flag
    # `cymbal_softmax`). Off => independent sigmoid+BCE heads (the baseline).
    cymbal_softmax: bool = False
    cymbal_ce_weight: float = 1.0  # scale of the cymbal CE term vs the per-lane BCE

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

    def lane_layer_map(self) -> dict[str, int]:
        """{lane: MERT layer} for every lane in `self.lanes`. A lane listed in
        `lane_layers` uses that layer; the rest fall back to `encoder_layer`. The
        single source of truth for which layer each head reads."""
        overrides = dict(self.lane_layers or ())
        return {lane: overrides.get(lane, self.encoder_layer) for lane in self.lanes}

    def distinct_layers(self) -> list[int]:
        """Sorted unique MERT layers the heads need. One element => the single-layer
        path (no per-lane routing); >1 => per-lane-layer routing is engaged."""
        return sorted(set(self.lane_layer_map().values()))

    def is_multilayer(self) -> bool:
        """True when the heads span more than one MERT layer."""
        return len(self.distinct_layers()) > 1
