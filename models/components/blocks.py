"""Shared 1D building blocks for denoiser variants."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStack1D(nn.Module):
    """Two-layer Conv-BN-GELU stack."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleAttentionGate(nn.Module):
    """Additive attention gate for skip filtering."""

    def __init__(self, skip_ch: int, gate_ch: int):
        super().__init__()
        self.skip_proj = nn.Conv1d(skip_ch, skip_ch, kernel_size=1)
        self.gate_proj = nn.Conv1d(gate_ch, skip_ch, kernel_size=1)
        self.score = nn.Conv1d(skip_ch, 1, kernel_size=1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-1] != skip.shape[-1]:
            gate = F.interpolate(gate, size=skip.shape[-1], mode="linear", align_corners=False)
        a = torch.tanh(self.skip_proj(skip) + self.gate_proj(gate))
        alpha = torch.sigmoid(self.score(a))
        return skip * alpha


class BiGRUBottleneck(nn.Module):
    """Temporal bottleneck with optional bidirectional GRU."""

    def __init__(self, channels: int, hidden: int = 128, layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.proj = nn.Conv1d(channels, hidden * (2 if bidirectional else 1), kernel_size=1)
        self.gru = nn.GRU(
            input_size=hidden * (2 if bidirectional else 1),
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.out = nn.Conv1d(hidden * (2 if bidirectional else 1), channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.proj(x).permute(0, 2, 1)
        seq, _ = self.gru(seq)
        return self.out(seq.permute(0, 2, 1))


class SwinLikeBottleneck(nn.Module):
    """Lightweight MHSA bottleneck as Swin-like proxy."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.permute(0, 2, 1)
        y = self.norm1(seq)
        y, _ = self.attn(y, y, y)
        seq = seq + y
        seq = seq + self.ffn(self.norm2(seq))
        return seq.permute(0, 2, 1)


class MaxPoolDown(nn.Module):
    """Downsample with maxpool."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class TransposeUp(nn.Module):
    """Upsample with transposed convolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        y = self.up(x)
        if target_len is not None and y.shape[-1] != target_len:
            y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
        return y
