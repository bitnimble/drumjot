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
    """BiGRU + linear over (B, T, in_dim) frozen features -> (B, T) logits."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.proj(h).squeeze(-1)


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
