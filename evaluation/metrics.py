"""Shared metrics for evaluation and quick benchmarks."""

from __future__ import annotations

import torch


def compute_ncc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean NCC over batch (1D signals)."""
    p = pred.detach().float().flatten(1)
    t = target.detach().float().flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + eps
    return (num / den).mean().item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(((pred.float() - target.float()) ** 2).mean()).item()
