"""Per-lane onset heads over frozen SSL-encoder features.

Decision (design spec §4): a SEPARATE small head per lane, not one
multi-output head. Separate heads avoid inter-lane negative transfer on
the overlapping HF lanes (open-hat sizzle vs crash) while the shared
frozen features still carry full-kit context. Each head is a small BiGRU +
linear emitting one onset logit per frame.
"""
from __future__ import annotations

import torch
from torch import nn

from drumjot_training.lanes import LANES


class OnsetHead(nn.Module):
    """BiGRU + linear over (B, T, in_dim) frozen features -> (B, T) logits.

    A second linear (`act`) emits an auxiliary frame-ACTIVITY logit ("is this
    instrument still ringing?") off the same GRU states. Only the sustained
    lanes' activity is supervised (Onsets-and-Frames-style dual objective; see
    `targets.ring_spans`): open hi-hat / cymbals are *defined* by their tails,
    which a pure onset target never shows the head. Inference uses only the
    onset logits, so old checkpoints (no `act` weights) stay loadable."""

    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 2,
                 calibrate: bool = True):
        super().__init__()
        self.calibrate = calibrate
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * hidden, 1)
        self.act = nn.Linear(2 * hidden, 1)
        # Per-clip calibration: a (mean+max) pooled summary of the GRU states ->
        # (bias, log-scale), applied to the onset logit as `exp(scale)*logit - bias`.
        # This lets the operating point adapt per clip (a learned per-clip threshold,
        # the differentiable part of the oracle gap) and trains end-to-end on the
        # onset BCE. Zero-init => identity at start (bias 0, scale 1), so it's a no-op
        # until learned -- and old checkpoints without it load as plain (no-cal) heads.
        self.calib = nn.Linear(4 * hidden, 2)
        nn.init.zeros_(self.calib.weight)
        nn.init.zeros_(self.calib.bias)

    @staticmethod
    def _pool(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """(B, T, 2h) -> (B, 4h) mean+max pool over real frames (mask True=keep)."""
        if mask is None:
            return torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        mb = mask.bool().unsqueeze(-1)  # (B, T, 1)
        cnt = mb.sum(dim=1).clamp(min=1).to(h.dtype)
        mean = (h * mb.to(h.dtype)).sum(dim=1) / cnt
        mx = h.masked_fill(~mb, float("-inf")).amax(dim=1)
        return torch.cat([mean, mx], dim=-1)

    def _calibrate(self, onset: torch.Tensor, h: torch.Tensor,
                   mask: torch.Tensor | None) -> torch.Tensor:
        if not self.calibrate:  # bypass (calib stays at zero-init => never used)
            return onset
        # Detach the pool input: the calib head adapts to the GRU representation but
        # cannot reshape it (no grad flows GRU->pool->calib). This stops the
        # co-adaptation that dragged the oracle/ranking down when calib was trained
        # jointly; the GRU still learns onset ranking via proj (scaled by exp(scale)).
        bg = self.calib(self._pool(h.detach(), mask))         # (B, 2)
        bias, log_scale = bg[:, 0:1], bg[:, 1:2]
        return torch.exp(log_scale.clamp(-3.0, 3.0)) * onset - bias  # (B,1) bcasts over (B,T)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.gru(x)
        return self._calibrate(self.proj(h).squeeze(-1), h, mask)

    def forward_all(self, x: torch.Tensor, mask: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU pass -> (calibrated onset logits (B, T), activity logits (B, T)).

        `mask` (B, T; True = real frame) bounds the calibration pool to non-pad
        frames; None pools over all frames (single clip / no-pad batch)."""
        h, _ = self.gru(x)
        onset = self._calibrate(self.proj(h).squeeze(-1), h, mask)
        return onset, self.act(h).squeeze(-1)


class MultiLaneHeads(nn.Module):
    """One `OnsetHead` per lane; forward -> (B, n_lanes, T) logits in `lane_names` order."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_layers: int = 2,
        lane_names: tuple[str, ...] = LANES,
        calibrate: bool = True,
    ):
        super().__init__()
        self.lane_names = tuple(lane_names)
        self.heads = nn.ModuleDict(
            {lane: OnsetHead(in_dim, hidden, num_layers, calibrate) for lane in self.lane_names}
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        outs = [self.heads[lane](x, mask) for lane in self.lane_names]
        return torch.stack(outs, dim=1)

    def forward_all(self, x: torch.Tensor, mask: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(onset logits (B, n_lanes, T), activity logits (B, n_lanes, T)). `mask`
        (B, T; True = real frame) bounds each head's per-clip calibration pool."""
        pairs = [self.heads[lane].forward_all(x, mask) for lane in self.lane_names]
        return torch.stack([p[0] for p in pairs], dim=1), torch.stack([p[1] for p in pairs], dim=1)
