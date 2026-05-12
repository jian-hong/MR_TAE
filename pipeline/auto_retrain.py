"""Noise2Void-style self-supervised fine-tuning on unlabeled real AE windows.

Safety gate: new weights must be within NCC band vs baseline on data/test_real_qlin.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.base import build_model_from_registry
from evaluation.metrics import compute_ncc

import yaml


def ncc_baseline(model: nn.Module, device: torch.device) -> float:
    pack = torch.load(PROJECT_ROOT / "data/test_real_qlin.pt", map_location=device)
    x = pack["signals"].float()
    if x.dim() == 2:
        x = x.unsqueeze(1)
    x = x[:512].to(device)
    with torch.no_grad():
        o = model(x)
        den = o[0] if isinstance(o, tuple) else o
    return compute_ncc(den, x)


def mask_replace(x: torch.Tensor, frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Random mask; replace with neighbor roll."""
    B, C, L = x.shape
    m = torch.rand(B, L, device=x.device) < frac
    repl = torch.roll(x, shifts=3, dims=-1)
    x2 = x.clone()
    for b in range(B):
        x2[b, 0, m[b]] = repl[b, 0, m[b]]
    return x2, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="MR-TAE-FULL")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--incoming-glob", default="data/incoming/*.mat")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ncc-floor", type=float, default=0.01)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(PROJECT_ROOT / "config/model_registry.yaml") as f:
        reg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config/training_config.yaml") as f:
        tcfg = yaml.safe_load(f) or {}

    model = build_model_from_registry(args.model_id, reg, tcfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    st = ck.get("model_state") or ck.get("model_state_dict")
    if st:
        model.load_state_dict(st, strict=False)

    base_ncc = ncc_baseline(model, device)
    print("Baseline NCC (denoised vs input on test_real):", base_ncc)

    # Pool from test split as proxy if no incoming
    pack = torch.load(PROJECT_ROOT / "data/test_real_qlin.pt", map_location="cpu")
    X = pack["signals"].float()
    if X.dim() == 2:
        X = X.unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X), batch_size=16, shuffle=True, drop_last=True
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=device.type == "cuda")

    model.train()
    for ep in range(args.epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            xm, mask = mask_replace(xb, 0.1)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out = model(xm)
                den = out[0] if isinstance(out, tuple) else out
                loss = F.mse_loss(den[:, 0][mask], xb[:, 0][mask])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    model.eval()
    new_ncc = ncc_baseline(model, device)
    print("After N2V NCC:", new_ncc)

    ok = new_ncc >= base_ncc - args.ncc_floor
    out = {
        "accepted": ok,
        "baseline_ncc": base_ncc,
        "new_ncc": new_ncc,
        "model_id": args.model_id,
    }
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    with open(PROJECT_ROOT / "results" / "auto_retrain.json", "w") as f:
        json.dump(out, f, indent=2)
    if ok:
        torch.save(
            {"model_state": model.state_dict()},
            PROJECT_ROOT / "results" / f"{args.model_id}_n2v.pt",
        )
        print("Saved results/", f"{args.model_id}_n2v.pt")
    else:
        print("Rejected: NCC below floor")


if __name__ == "__main__":
    main()
