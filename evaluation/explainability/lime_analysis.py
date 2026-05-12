"""Perturbation-based importance (LIME-style segments over 2048 samples)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-id", default="MR-TAE-FULL")
    ap.add_argument("--test-data", default="data/test_real_qlin.pt")
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--segments", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.base import build_model_from_registry
    import yaml

    with open(PROJECT_ROOT / "config/model_registry.yaml") as f:
        reg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config/training_config.yaml") as f:
        tcfg = yaml.safe_load(f) or {}

    m = build_model_from_registry(args.model_id, reg, tcfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    st = ck.get("model_state") or ck.get("model_state_dict")
    if st:
        m.load_state_dict(st, strict=False)
    m.eval()

    pack = torch.load(PROJECT_ROOT / args.test_data, map_location="cpu")
    X = pack["signals"].float()
    if X.dim() == 2:
        X = X.unsqueeze(1)
    L = X.shape[-1]
    seg = args.segments
    seg_len = L // seg

    out_dir = PROJECT_ROOT / "results" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    imp_list = []
    with torch.no_grad():
        for w in range(min(args.n_windows, X.shape[0])):
            x = X[w : w + 1].to(device)
            o = m(x)
            full = o[0] if isinstance(o, tuple) else o
            importance = np.zeros(seg)
            for s in range(seg):
                xp = x.clone()
                a, b = s * seg_len, (s + 1) * seg_len
                xp[:, :, a:b] = 0
                op = m(xp)
                den = op[0] if isinstance(op, tuple) else op
                d = torch.mean(torch.abs(full - den)).item()
                importance[s] = d
            imp_list.append(importance)

    mean_imp = np.mean(imp_list, axis=0)
    plt.figure(figsize=(9, 3))
    plt.bar(np.arange(seg), mean_imp, color="#58c4dc")
    plt.xlabel("segment")
    plt.ylabel("1 - NCC (perturb masked)")
    plt.title("LIME-style segment importance (mean over windows)")
    plt.tight_layout()
    plt.savefig(out_dir / "lime_window_examples.png", dpi=150)
    plt.close()
    print("Wrote", out_dir / "lime_window_examples.png")


if __name__ == "__main__":
    main()
