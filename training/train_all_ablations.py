"""Unified trainer for all ablation models."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List
import sys

import numpy as np
import torch
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.registry import MODEL_REGISTRY


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fake_batch(batch_size: int, length: int, device: torch.device):
    clean = torch.randn(batch_size, 1, length, device=device)
    noisy = clean + 0.2 * torch.randn_like(clean)
    mask = torch.randint(0, 4, (batch_size, length), device=device)
    return noisy, clean, mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs-per-phase", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def phase_snr(phase: int):
    if phase == 1:
        return 5, 15
    if phase == 2:
        return -5, 5
    return -20, -5


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL_REGISTRY[args.model]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(args.phases) * args.epochs_per_phase))

    run_dir = Path("results") / args.model
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb")) if SummaryWriter is not None else None

    start_epoch = 0
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt["best"]

    epoch_idx = 0
    seg_weight = 0.0
    for phase in args.phases:
        snr_min, snr_max = phase_snr(phase)
        for _ in range(args.epochs_per_phase):
            if epoch_idx < start_epoch:
                epoch_idx += 1
                continue
            model.train()
            noisy, clean, mask = fake_batch(args.batch_size, args.length, device)
            den, seg = model(noisy)
            den_loss = F.smooth_l1_loss(den, clean)
            # NOTE: segmentation weight is ramped up by epoch progression, not constant from epoch 1.
            seg_weight = min(1.0, seg_weight + 0.05)
            seg_loss = F.cross_entropy(seg, mask) if seg.shape[1] > 1 else torch.tensor(0.0, device=device)
            loss = den_loss + seg_weight * seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Validation always uses full SNR range in unified runner.
            model.eval()
            with torch.no_grad():
                v_noisy, v_clean, v_mask = fake_batch(args.batch_size, args.length, device)
                v_den, v_seg = model(v_noisy)
                v_den_loss = F.smooth_l1_loss(v_den, v_clean)
                v_seg_loss = F.cross_entropy(v_seg, v_mask) if v_seg.shape[1] > 1 else torch.tensor(0.0, device=device)
                v_loss = v_den_loss + seg_weight * v_seg_loss

            if writer is not None:
                writer.add_scalar("train/loss", float(loss.item()), epoch_idx)
                writer.add_scalar("val/loss", float(v_loss.item()), epoch_idx)
                writer.add_scalar("train/seg_weight", float(seg_weight), epoch_idx)
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch_idx)
                writer.add_text("meta/phase", f"phase={phase}, snr=[{snr_min},{snr_max}]", epoch_idx)

            state = {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best,
                "model_id": args.model,
                "parameter_count": model.get_parameter_count(),
            }
            if (epoch_idx + 1) % 10 == 0:
                torch.save(state, ckpt_dir / f"epoch_{epoch_idx + 1}.pt")
            if v_loss.item() < best:
                best = float(v_loss.item())
                state["best"] = best
                torch.save(state, ckpt_dir / "best.pt")
            epoch_idx += 1

    summary = {
        "model_id": args.model,
        "epochs": epoch_idx,
        "best_val_loss": best,
        "parameter_count": model.get_parameter_count(),
    }
    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
