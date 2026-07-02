"""Per-frame masked loss terms for the onset heads.

Each averages over valid (unpadded, masked) frames; `frame_weight` (e.g.
`sibling_weight`) scales per-frame. Pure functions over torch tensors -- no
model/config state -- so the training loop and its tests share one definition.
"""
from __future__ import annotations


def sibling_weight(targets, sib_act, pos_w: float, neg_w: float):
    """Per-frame loss multiplier from sibling activity (lanes.CONFUSABLE).

    `sib_act` (B, n_lanes, T) is each lane's max confusable-sibling target.
    Frames where a sibling fires are the discriminative ones, so their loss is
    scaled up: negatives (this lane silent under sibling noise -> punish bleed
    triggers) toward `neg_w`, positives (true co-occurrence, the harder
    detection) toward `pos_w`. Smooth in both targets; 1 where no sibling is
    active or both weights are 1."""
    return 1.0 + sib_act * ((pos_w - 1.0) * targets + (neg_w - 1.0) * (1.0 - targets))


def masked_bce(logits, targets, mask, pos_weight, frame_weight=None):
    """Per-frame BCE averaged over valid (unpadded) frames and lanes.

    `pos_weight` is (n_lanes, 1), broadcasting over (B, n_lanes, T); `mask` is
    (B, T) and is broadcast across lanes so padded frames contribute nothing.
    `frame_weight` (B, n_lanes, T), e.g. `sibling_weight`, scales per-frame."""
    from torch.nn import functional as F

    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )  # (B, n_lanes, T)
    if frame_weight is not None:
        loss = loss * frame_weight
    m = mask.unsqueeze(1)  # (B, 1, T)
    denom = (m.sum() * logits.shape[1]).clamp_min(1.0)
    return (loss * m).sum() / denom


def masked_focal(logits, targets, mask, alpha: float = 2.0, beta: float = 4.0, frame_weight=None):
    """CenterNet-style penalty-reduced focal loss for soft Gaussian onset
    targets (peak 1.0), averaged over valid frames' positives.

    Positives are the exact peak frames (target == 1); elsewhere the negative
    penalty is reduced by `(1 - target) ** beta` so the Gaussian skirt around a
    peak is barely penalised, and `(.) ** alpha` focuses on hard frames. This
    replaces `pos_weight` reweighting (it targets the rare/hard frames directly)
    so no `pos_weight` is needed. Normalised by the positive count (CenterNet
    convention). `mask` is (B, T), broadcast across lanes; `frame_weight`
    (e.g. `sibling_weight`) scales per-frame."""
    import torch

    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    pos = (targets >= 1.0).float()
    pos_loss = -((1.0 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1.0 - targets) ** beta) * (p**alpha) * torch.log(1.0 - p) * (1.0 - pos)
    loss = pos_loss + neg_loss
    if frame_weight is not None:
        loss = loss * frame_weight
    m = mask.unsqueeze(1)  # (B, 1, T)
    npos = (pos * m).sum().clamp_min(1.0)
    return (loss * m).sum() / npos


def masked_cymbal_ce(rc_logits, rc_targets, mask, pos_weight=1.0):
    """Joint 3-way soft cross-entropy {none, ride, crash} for the cymbal lanes, with
    `none` the fixed-0 reference class. `rc_logits` / `rc_targets` are (B, 2, T) for
    (ride, crash) -- the rd, cr rows. A softmax over [0, ride, crash] forces the
    model to COMMIT to ONE cymbal type per onset (kills the ride<->crash both-fire
    confusion on the merged cymbals stem). Soft target = normalized [(1-r)(1-c), r, c]
    from the per-lane Gaussians. Onset frames are upweighted by `pos_weight` (the
    rare cymbal onsets, mirroring the per-lane BCE pos_weight); mean over valid
    frames so the magnitude sits alongside the BCE terms."""
    import torch
    from torch.nn import functional as F

    z = torch.zeros_like(rc_logits[:, :1])  # `none` logit = 0 (fixed reference)
    logp = F.log_softmax(torch.cat([z, rc_logits], dim=1), dim=1)  # (B, 3, T)
    rt, ct = rc_targets[:, 0], rc_targets[:, 1]  # (B, T) ride/crash Gaussians
    tgt = torch.stack([(1.0 - rt) * (1.0 - ct), rt, ct], dim=1)  # (B, 3, T)
    tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp_min(1e-6)
    ce = -(tgt * logp).sum(dim=1)  # (B, T)
    w = 1.0 + (pos_weight - 1.0) * torch.clamp(rt + ct, max=1.0)  # upweight onset frames
    return (ce * w * mask).sum() / mask.sum().clamp_min(1.0)
