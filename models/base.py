"""Base denoiser abstractions and ablation config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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
    loss_config: str = "A"  # A/B/C/D — used by loss weighting / logging


def build_model_from_registry(
    model_id: str, registry: Dict[str, Any], training_cfg: Dict[str, Any]
) -> "BaseDenoiser":
    """Build `GenericUNetDenoiser` from `config/model_registry.yaml` entry."""
    from .variants import GenericUNetDenoiser

    models = registry.get("models") or registry
    entry = models.get(model_id)
    if not entry:
        raise ValueError(
            f"Model '{model_id}' not in registry. Available: {list(models.keys())}"
        )
    cfg = AblationConfig(
        use_bigru=entry.get("use_bigru", True),
        use_swin=entry.get("use_swin", True),
        use_attention_gates=entry.get("use_attention_gates", True),
        use_mtl=entry.get("use_mtl", True),
        use_wavelet=entry.get("use_wavelet_pooling", True),
        loss_config=str(entry.get("loss_config", "A")),
    )
    base_ch = int(training_cfg.get("encoder_base_ch", 32))
    return GenericUNetDenoiser(cfg, base_channels=base_ch, model_id=model_id)


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
