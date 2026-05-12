"""
UnifiedTrainer — one epoch / validate / snapshot for orchestrator + MR-TAE models.

Uses mr_tae_fusion PDSignalDataset with optional curriculum override per epoch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from mr_tae_fusion.config import Config
from mr_tae_fusion.data.dataset import PDSignalDataset
from mr_tae_fusion.training.losses import MultiTaskLoss

logger = logging.getLogger(__name__)


def _ncc_tensor(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    p = pred.detach().float().flatten(1)
    t = target.detach().float().flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + eps
    return (num / den).mean().item()


def _rmse_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(((pred - target) ** 2).mean()).item()


class UnifiedTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        model_id: str,
        batch_size: int,
        use_amp: bool = True,
        use_grad_checkpointing: bool = False,
        train_samples: int = 4000,
        val_samples: int = 800,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.config_dict = config
        self.model_id = model_id
        self.batch_size = batch_size
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_grad_ckpt = use_grad_checkpointing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._cfg = Config()
        # Match training signal length / rate from project
        if "sample_rate" in config:
            self._cfg.signal.sample_rate = float(config["sample_rate"])
        te = int(config.get("total_epochs", 100))
        self._cfg.training.total_epochs = te

        self.train_ds = PDSignalDataset(
            config=self._cfg,
            num_samples=train_samples,
            epoch=0,
            total_epochs=te,
            mode="train",
            seed=seed,
        )
        self.val_ds = PDSignalDataset(
            config=self._cfg,
            num_samples=val_samples,
            epoch=0,
            total_epochs=te,
            mode="val",
            seed=seed + 1,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        self.use_mtl = getattr(model, "cfg", None) and getattr(
            model.cfg, "use_mtl", True
        )
        if self.use_mtl:
            self.criterion = MultiTaskLoss(
                charbonnier_eps=float(config.get("charbonnier_eps", 1e-3)),
            )
            self.criterion.to(self.device)
            params = list(model.parameters()) + list(self.criterion.parameters())
        else:
            self.criterion = None
            params = model.parameters()

        lr = float(config.get("learning_rate", 3e-4))
        wd = float(config.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        self.scaler = GradScaler(enabled=self.use_amp)

    def _apply_curriculum(self, epoch: int, snr_range: Tuple[float, float], noise_type: str) -> None:
        self.train_ds.update_epoch(epoch)
        self.train_ds.set_curriculum_override(snr_range, noise_type)
        self.val_ds.update_epoch(epoch)
        self.val_ds.set_curriculum_override(snr_range, noise_type)

    def train_epoch(
        self, epoch: int, phase: int, snr_range: Tuple[float, float], noise_type: str
    ) -> Dict[str, float]:
        self.model.train()
        if self.use_mtl:
            self.criterion.train()
        self._apply_curriculum(epoch, snr_range, noise_type)

        losses = []
        nccs = []
        for batch in tqdm(self.train_loader, desc=f"train {self.model_id} ep{epoch}", leave=False):
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            mask = batch["mask"].long().to(self.device).clamp(0, 3)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                den, seg = self.model(noisy)
                if self.use_mtl:
                    loss, _ = self.criterion(den, clean, seg, mask)
                else:
                    loss = F.smooth_l1_loss(den, clean)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.use_mtl:
                torch.nn.utils.clip_grad_norm_(self.criterion.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())
            nccs.append(_ncc_tensor(den, clean))

        return {
            "train_loss": float(np.mean(losses)),
            "train_ncc": float(np.mean(nccs)),
        }

    @torch.no_grad()
    def validate(
        self, epoch: int, phase: int, snr_range: Tuple[float, float], noise_type: str
    ) -> Dict[str, float]:
        self.model.eval()
        if self.use_mtl:
            self.criterion.eval()
        self._apply_curriculum(epoch, snr_range, noise_type)

        losses = []
        nccs = []
        rmses = []
        for batch in self.val_loader:
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            mask = batch["mask"].long().to(self.device).clamp(0, 3)
            with autocast(enabled=self.use_amp):
                den, seg = self.model(noisy)
                if self.use_mtl:
                    loss, _ = self.criterion(den, clean, seg, mask)
                else:
                    loss = F.smooth_l1_loss(den, clean)
            losses.append(loss.item())
            nccs.append(_ncc_tensor(den, clean))
            rmses.append(_rmse_tensor(den, clean))

        return {
            "val_loss": float(np.mean(losses)),
            "val_ncc": float(np.mean(nccs)),
            "val_rmse": float(np.mean(rmses)),
        }

    @torch.no_grad()
    def save_snapshot_figure(self, path: str) -> None:
        """Save noisy / denoised / clean for one validation batch."""
        self.model.eval()
        batch = next(iter(self.val_loader))
        noisy = batch["noisy"].to(self.device)
        clean = batch["clean"].to(self.device)
        with autocast(enabled=self.use_amp):
            den, _ = self.model(noisy)
        n = min(3, noisy.shape[0])
        t = np.arange(noisy.shape[-1]) / float(self._cfg.signal.sample_rate) * 1e6
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i in range(n):
            axes[i].plot(t, noisy[i, 0].cpu().numpy(), alpha=0.5, label="noisy")
            axes[i].plot(t, den[i, 0].cpu().numpy(), label="denoised")
            axes[i].plot(t, clean[i, 0].cpu().numpy(), alpha=0.7, label="clean")
            axes[i].legend(fontsize=8)
            axes[i].set_ylabel("amp")
        axes[-1].set_xlabel("time (µs)")
        fig.suptitle(self.model_id)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
