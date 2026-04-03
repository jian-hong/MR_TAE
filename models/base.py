"""Base denoiser abstractions and ablation config."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class AblationConfig:
    """Boolean toggles for MR-TAE family ablations."""

    use_bigru: bool = True
    use_swin: bool = True
    use_attention_gates: bool = True
    use_mtl: bool = True
    use_wavelet: bool = True


class BaseDenoiser(nn.Module):
    """Common denoiser API used by all variants."""

    MODEL_ID = "base-denoiser"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def denoise_only(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward(x)
        return y

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_metadata(self) -> Dict[str, str]:
        return {"model_id": self.MODEL_ID}
