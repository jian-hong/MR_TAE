"""SHAP DeepExplainer on encoder-ish front-end (first conv stack)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import shap
except ImportError as e:
    raise SystemExit("Install shap: pip install shap") from e


class EncoderWrapper(torch.nn.Module):
    """First-stage U-Net encoder from GenericUNetDenoiser."""

    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__()
        self.enc1 = m.enc1
        self.down = m.down
        self.enc2 = m.enc2
        self.enc3 = m.enc3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        return e3.mean(dim=1, keepdim=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-id", default="MR-TAE-FULL")
    ap.add_argument("--test-data", default="data/test_real_qlin.pt")
    ap.add_argument("--n-background", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=200)
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

    wrap = EncoderWrapper(m).to(device)

    pack = torch.load(PROJECT_ROOT / args.test_data, map_location="cpu")
    X = pack["signals"].float()
    if X.dim() == 2:
        X = X.unsqueeze(1)
    idx = torch.randperm(X.shape[0])[: max(args.n_background + args.n_test, 256)]
    X = X[idx]

    bg = X[: args.n_background].to(device)
    tst = X[args.n_background : args.n_background + args.n_test].to(device)

    explainer = shap.DeepExplainer(wrap, bg)
    sv = explainer.shap_values(tst)

    out_dir = PROJECT_ROOT / "results" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mean abs SHAP over batch, channels
    s = np.asarray(sv)
    if s.ndim == 4:
        sabs = np.mean(np.abs(s), axis=(0, 1))
        curve = sabs[0]
    else:
        curve = np.mean(np.abs(s), axis=0).ravel()

    plt.figure(figsize=(10, 3))
    plt.plot(curve, lw=0.8)
    plt.title("Mean |SHAP| (encoder output proxy)")
    plt.xlabel("time bin")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close()

    # Example overlay
    ex = tst[0, 0].detach().cpu().numpy()
    sh0 = np.asarray(sv[0])
    if sh0.ndim > 1:
        sh0 = sh0.mean(axis=0)
    plt.figure(figsize=(10, 3))
    plt.plot(ex, label="signal", alpha=0.7)
    plt.plot(sh0 / (np.max(np.abs(sh0)) + 1e-8), label="SHAP (norm)", alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "shap_examples.png", dpi=150)
    plt.close()
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
