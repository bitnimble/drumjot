"""Demixed_DilatedTransformerModel — vendored from Beat-Transformer/code/DilatedTransformer.py.

Only modifications:
- `from DilatedTransformerLayer import ...` -> `from .layer import ...`
- removed the `__main__` smoke-test block

Architecture is byte-identical to upstream; the released fold_N
checkpoints were trained against this exact graph. Constructor defaults
intentionally left at the upstream-published demo values (dmodel=128
etc.); the released checkpoints actually use dmodel=256, nhead=8,
d_hid=1024 — those are passed explicitly by the wrapper that loads
the checkpoint, see `pipeline/beat_transformer.py`.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer as torchTransformerEncoderLayer

from .layer import DilatedTransformerLayer


class Demixed_DilatedTransformerModel(nn.Module):
    def __init__(
        self,
        attn_len=5,
        instr=5,
        ntoken=2,
        dmodel=128,
        nhead=2,
        d_hid=512,
        nlayers=9,
        norm_first=True,
        dropout=0.1,
    ):
        super().__init__()
        self.nhead = nhead
        self.nlayers = nlayers
        self.attn_len = attn_len
        self.head_dim = dmodel // nhead
        self.dmodel = dmodel
        assert self.head_dim * nhead == dmodel, "embed_dim must be divisible by num_heads"

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(5, 3), stride=1, padding=(2, 0)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=dmodel, kernel_size=(3, 6), stride=1, padding=(1, 0)
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(p=dropout)

        self.Transformer_layers = nn.ModuleDict({})
        for idx in range(nlayers):
            self.Transformer_layers[f"time_attention_{idx}"] = DilatedTransformerLayer(
                dmodel,
                nhead,
                d_hid,
                dropout,
                Er_provided=False,
                attn_len=attn_len,
                norm_first=norm_first,
            )
            if (idx >= 3) and (idx <= 5):
                self.Transformer_layers[f"instr_attention_{idx}"] = torchTransformerEncoderLayer(
                    dmodel, nhead, d_hid, dropout, batch_first=True, norm_first=norm_first
                )

        self.out_linear = nn.Linear(dmodel, ntoken)

        self.dropout_t = nn.Dropout(p=0.5)
        self.out_linear_t = nn.Linear(dmodel, 300)

    def forward(self, x):
        batch, instr, time, _ = x.shape
        x = x.reshape(-1, 1, time, x.shape[-1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = x.reshape(-1, self.dmodel, time).transpose(1, 2)
        t = []

        for layer in range(self.nlayers):
            x, skip = self.Transformer_layers[f"time_attention_{layer}"](x, layer=layer)
            skip = skip.reshape(batch, instr, time, self.dmodel)
            t.append(skip.mean(1))

            if (layer >= 3) and (layer <= 5):
                x = x.reshape(batch, instr, time, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, instr, self.dmodel)

                x = self.Transformer_layers[f"instr_attention_{layer}"](x)

                x = x.reshape(batch, time, instr, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, time, self.dmodel)

        x = torch.relu(x)
        x = x.reshape(batch, instr, time, self.dmodel)
        x = x.mean(1)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1)
        t = self.out_linear_t(t)

        return x, t
