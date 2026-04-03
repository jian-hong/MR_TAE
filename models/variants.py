"""Ablation and cross-combination denoiser variants."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AblationConfig, BaseDenoiser
from .components.blocks import (
    BiGRUBottleneck,
    ConvStack1D,
    MaxPoolDown,
    SimpleAttentionGate,
    SwinLikeBottleneck,
    TransposeUp,
)


class GenericUNetDenoiser(BaseDenoiser):
    """Configurable 1D U-Net style denoiser used by all variants."""

    MODEL_ID = "generic"

    def __init__(self, cfg: AblationConfig, base_channels: int = 32):
        super().__init__()
        self.cfg = cfg
        channels = [base_channels, base_channels * 2, base_channels * 4]

        self.enc1 = ConvStack1D(1, channels[0])
        self.enc2 = ConvStack1D(channels[0], channels[1])
        self.enc3 = ConvStack1D(channels[1], channels[2])
        self.down = MaxPoolDown()

        self.bottleneck = nn.Sequential()
        if cfg.use_bigru:
            self.bottleneck.append(BiGRUBottleneck(channels[2], hidden=channels[2] // 2, bidirectional=True))
        if cfg.use_swin:
            self.bottleneck.append(SwinLikeBottleneck(channels[2], heads=4))
        if len(self.bottleneck) == 0:
            self.bottleneck = ConvStack1D(channels[2], channels[2])

        self.attn2 = SimpleAttentionGate(channels[1], channels[2])
        self.attn1 = SimpleAttentionGate(channels[0], channels[1])

        self.up2 = TransposeUp(channels[2], channels[1])
        self.dec2 = ConvStack1D(channels[1] + channels[1], channels[1])
        self.up1 = TransposeUp(channels[1], channels[0])
        self.dec1 = ConvStack1D(channels[0] + channels[0], channels[0])

        self.out = nn.Conv1d(channels[0], 1, kernel_size=1)
        self.seg = nn.Conv1d(channels[0], 4, kernel_size=1) if cfg.use_mtl else nn.Identity()

    def _skip(self, skip: torch.Tensor, gate: torch.Tensor, attn: SimpleAttentionGate) -> torch.Tensor:
        if self.cfg.use_attention_gates:
            return attn(skip, gate)
        return skip

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))

        b = self.bottleneck(e3)

        u2 = self.up2(b, e2.shape[-1])
        s2 = self._skip(e2, b, self.attn2)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))

        u1 = self.up1(d2, e1.shape[-1])
        s1 = self._skip(e1, d2, self.attn1)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))

        den = self.out(d1)
        seg = self.seg(d1)
        if isinstance(self.seg, nn.Identity):
            seg = torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        return den, seg


class MRTAEFull(GenericUNetDenoiser):
    MODEL_ID = "MR-TAE-FULL"

    def __init__(self):
        super().__init__(AblationConfig(True, True, True, True, True))


class MRTAENoBiGRU(GenericUNetDenoiser):
    MODEL_ID = "MR-TAE-noBiGRU"

    def __init__(self):
        super().__init__(AblationConfig(False, True, True, True, True))


class MRTAENoSwin(GenericUNetDenoiser):
    MODEL_ID = "MR-TAE-noSwin"

    def __init__(self):
        super().__init__(AblationConfig(True, False, True, True, True))


class MRTAENoAttn(GenericUNetDenoiser):
    MODEL_ID = "MR-TAE-noAttn"

    def __init__(self):
        super().__init__(AblationConfig(True, True, False, True, True))


class MRTAENoMTL(GenericUNetDenoiser):
    MODEL_ID = "MR-TAE-noMTL"

    def __init__(self):
        super().__init__(AblationConfig(True, True, True, False, True))


class MRTAENoWavelet(GenericUNetDenoiser):
    MODEL_ID = "MR-TAE-noWavelet"

    def __init__(self):
        # NOTE: current generic model uses maxpool/transpose path by design.
        super().__init__(AblationConfig(True, True, True, True, False))


class MWCNNBiGRU(GenericUNetDenoiser):
    MODEL_ID = "MWCNN-BiGRU"

    def __init__(self):
        super().__init__(AblationConfig(True, False, True, True, True))


class MWCNNSwin(GenericUNetDenoiser):
    MODEL_ID = "MWCNN-Swin"

    def __init__(self):
        super().__init__(AblationConfig(False, True, True, True, True))


class UNetBiGRUSwin(GenericUNetDenoiser):
    MODEL_ID = "UNet-BiGRU-Swin"

    def __init__(self):
        super().__init__(AblationConfig(True, True, False, True, False))


class UNetBiGRU(GenericUNetDenoiser):
    MODEL_ID = "UNet-BiGRU"

    def __init__(self):
        super().__init__(AblationConfig(True, False, False, True, False))


class UNetAttn(GenericUNetDenoiser):
    MODEL_ID = "UNet-Attn"

    def __init__(self):
        super().__init__(AblationConfig(False, False, True, True, False))
