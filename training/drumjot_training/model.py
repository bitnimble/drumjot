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

    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * hidden, 1)
        self.act = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.proj(h).squeeze(-1)

    def forward_all(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU pass -> (onset logits (B, T), activity logits (B, T))."""
        h, _ = self.gru(x)
        return self.proj(h).squeeze(-1), self.act(h).squeeze(-1)


class MultiLaneHeads(nn.Module):
    """One `OnsetHead` per lane; forward -> (B, n_lanes, T) logits in `lane_names` order."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_layers: int = 2,
        lane_names: tuple[str, ...] = LANES,
    ):
        super().__init__()
        self.lane_names = tuple(lane_names)
        self.heads = nn.ModuleDict(
            {lane: OnsetHead(in_dim, hidden, num_layers) for lane in self.lane_names}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [self.heads[lane](x) for lane in self.lane_names]
        return torch.stack(outs, dim=1)

    def forward_all(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(onset logits (B, n_lanes, T), activity logits (B, n_lanes, T))."""
        pairs = [self.heads[lane].forward_all(x) for lane in self.lane_names]
        return torch.stack([p[0] for p in pairs], dim=1), torch.stack([p[1] for p in pairs], dim=1)
