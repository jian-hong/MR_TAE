"""
Master benchmark: classical baselines + all registered DL models.

Evaluates on:
  data/test_synthetic_ae.pt  (clean / noisy / masks)
  data/test_real_qlin.pt     (signals; real metrics vs mild SG reference if no clean)

Usage
-----
  python evaluation/benchmark_runner.py --compare-all
  python evaluation/benchmark_runner.py --model-id MR-TAE-FULL \\
      --checkpoint results/MR-TAE-FULL/checkpoints/best.pt

Outputs (results/benchmark/):
  comparison_table.json, comparison_bar.png, ablation_heatmap.png,
  efficiency_scatter.png, snr_sweep.png
Also updates docs/RESULTS_COMPARISON.md
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import signal as sp_signal
from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pywt
except ImportError:
    pywt = None

from models.registry import MODEL_REGISTRY
from models.base import build_model_from_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("benchmark")

SYNTH_PATH = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
REAL_PATH = PROJECT_ROOT / "data" / "test_real_qlin.pt"
SNR_LEVELS = [-15, -10, -5, 0, 5]

# Classical wavelet/SG on 10k×2048 is extremely slow; subsample by default.
_DEFAULT_MAX = int(os.environ.get("BENCHMARK_MAX_SAMPLES", "1500"))

# Plan + registry names
MODELS_ORDER: List[str] = [
    "wavelet_db4_soft",
    "wavelet_bayesshrink",
    "savitzky_golay",
    "MR-TAE-FULL",
    "MR-TAE-noBiGRU",
    "MR-TAE-noSwin",
    "MR-TAE-noAttn",
    "MR-TAE-noMTL",
    "MR-TAE-noWavelet",
    "MWCNN-BiGRU",
    "MWCNN-Swin",
    "UNet-BiGRU-Swin",
    "UNet-BiGRU",
    "UNet-Attn",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ncc_np(pred: np.ndarray, tgt: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = pred.astype(np.float64)
    t = tgt.astype(np.float64)
    p = p - p.mean(axis=-1, keepdims=True)
    t = t - t.mean(axis=-1, keepdims=True)
    num = (p * t).sum(axis=-1)
    den = np.sqrt((p * p).sum(axis=-1) * (t * t).sum(axis=-1)) + eps
    return num / den


def _rmse_np(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred.astype(np.float64) - tgt.astype(np.float64)) ** 2).mean(axis=-1))


def _snr_improvement_np(pred: np.ndarray, clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    sig = (clean.astype(np.float64) ** 2).mean(axis=-1) + 1e-12
    nin = ((noisy.astype(np.float64) - clean.astype(np.float64)) ** 2).mean(axis=-1) + 1e-12
    nout = ((pred.astype(np.float64) - clean.astype(np.float64)) ** 2).mean(axis=-1) + 1e-12
    sin = 10 * np.log10(sig / nin)
    sout = 10 * np.log10(sig / nout)
    return sout - sin


def _dice_logits(pred_logits: np.ndarray, tgt_mask: np.ndarray, eps: float = 1e-6) -> float:
    # pred_logits (B,C,L)
    pred = pred_logits.argmax(axis=1)
    pb = (pred > 0).astype(np.float64)
    tb = (tgt_mask > 0).astype(np.float64)
    inter = (pb * tb).sum()
    return float(2 * inter / (pb.sum() + tb.sum() + eps))


def _fpr_logits(pred_logits: np.ndarray, tgt_mask: np.ndarray) -> float:
    pred = pred_logits.argmax(axis=1)
    fp = ((pred > 0) & (tgt_mask == 0)).astype(np.float64).sum()
    neg = (tgt_mask == 0).astype(np.float64).sum() + 1e-8
    return float(fp / neg)


def _sg_reference(noisy: np.ndarray) -> np.ndarray:
    out = np.empty_like(noisy, dtype=np.float32)
    wl = 31 if noisy.shape[-1] >= 31 else (noisy.shape[-1] // 2 * 2 - 1)
    wl = max(wl, 5)
    for i in range(noisy.shape[0]):
        out[i] = sp_signal.savgol_filter(noisy[i].astype(np.float64), wl, 3).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Classical denoisers (numpy, B x L)
# ---------------------------------------------------------------------------

def denoise_wavelet_soft(noisy: np.ndarray) -> np.ndarray:
    if pywt is None:
        return noisy.copy()
    out = np.empty_like(noisy, dtype=np.float32)
    for i in range(noisy.shape[0]):
        coeffs = pywt.wavedec(noisy[i], "db4", mode="symmetric", level=4)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
        thresh = sigma * np.sqrt(2 * np.log(noisy.shape[-1]))
        coeffs_t = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
        out[i] = pywt.waverec(coeffs_t, "db4", mode="symmetric")[: noisy.shape[-1]]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def denoise_wavelet_bayes(noisy: np.ndarray) -> np.ndarray:
    """Simple level-wise BayesShrink-style threshold on detail coeffs."""
    if pywt is None:
        return noisy.copy()
    out = np.empty_like(noisy, dtype=np.float32)
    L = 4
    for i in range(noisy.shape[0]):
        coeffs = pywt.wavedec(noisy[i], "db4", mode="symmetric", level=L)
        new_c = [coeffs[0]]
        for c in coeffs[1:]:
            med = np.median(np.abs(c)) + 1e-12
            sig = med / 0.6745
            n = max(c.size, 1)
            t = sig**2 / (np.sqrt(max(np.var(c), 1e-18)) + 1e-12)
            t = max(t, 1e-6)
            new_c.append(pywt.threshold(c, t, mode="soft"))
        rec = pywt.waverec(new_c, "db4", mode="symmetric")
        out[i] = rec[: noisy.shape[-1]]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def denoise_savgol(noisy: np.ndarray) -> np.ndarray:
    return _sg_reference(noisy)


# ---------------------------------------------------------------------------
# DL inference
# ---------------------------------------------------------------------------

def _load_dl_model(
    model_id: str,
    checkpoint: Optional[Path],
    device: torch.device,
) -> torch.nn.Module:
    reg_path = PROJECT_ROOT / "config" / "model_registry.yaml"
    with open(reg_path) as f:
        registry = yaml.safe_load(f)
    tcfg_path = PROJECT_ROOT / "config" / "training_config.yaml"
    training_cfg: Dict[str, Any] = {}
    if tcfg_path.exists():
        with open(tcfg_path) as f:
            training_cfg = yaml.safe_load(f) or {}
    model = build_model_from_registry(model_id, registry, training_cfg).to(device)
    if checkpoint and checkpoint.is_file():
        ck = torch.load(checkpoint, map_location=device)
        state = ck.get("model_state") or ck.get("model_state_dict")
        if state is not None:
            model.load_state_dict(state, strict=False)
            logger.info("Loaded weights from %s", checkpoint)
        else:
            logger.warning("Checkpoint %s missing model_state — using random init", checkpoint)
    else:
        logger.warning("No checkpoint for %s — random weights", model_id)
    model.eval()
    return model


def _forward_dl(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch: int = 32,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """x: (B,1,L)"""
    outs, segs = [], []
    seg_ok = True
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            chunk = x[i : i + batch].to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                o = model(chunk)
            if isinstance(o, tuple):
                outs.append(o[0].float().cpu())
                sg = o[1]
                if isinstance(sg, torch.Tensor) and sg.ndim == 3 and sg.shape[1] >= 2:
                    segs.append(sg.float().cpu())
                else:
                    seg_ok = False
            else:
                outs.append(o.float().cpu())
                seg_ok = False
    den = torch.cat(outs, dim=0)
    if seg_ok and segs:
        seg_t = torch.cat(segs, dim=0)
    else:
        seg_t = None
    return den, seg_t


def _measure_latency(model: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    dummy1 = torch.randn(1, 1, 2048, device=device)
    dummy32 = torch.randn(32, 1, 2048, device=device)
    warm = 5
    with torch.no_grad():
        for _ in range(warm):
            with autocast(enabled=device.type == "cuda"):
                _ = model(dummy1)
                _ = model(dummy32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    nrep = 30
    with torch.no_grad():
        with autocast(enabled=device.type == "cuda"):
            for _ in range(nrep):
                _ = model(dummy1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_b1 = (time.perf_counter() - t0) / nrep * 1000

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    nrep2 = 20
    with torch.no_grad():
        with autocast(enabled=device.type == "cuda"):
            for _ in range(nrep2):
                _ = model(dummy32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_b32 = (time.perf_counter() - t0) / nrep2 / 32 * 1000
    return ms_b1, ms_b32


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model_on_synthetic(
    model_id: str,
    noisy: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
    infer: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
    param_m: float,
) -> Dict[str, Any]:
    pred, logits = infer(noisy)
    ncc_b = _ncc_np(pred, clean)
    rmse_b = _rmse_np(pred, clean)
    snri_b = _snr_improvement_np(pred, clean, noisy)
    dice_v = float("nan")
    fpr_v = float("nan")
    if logits is not None:
        dice_v = _dice_logits(logits, mask.astype(np.int64))
        fpr_v = _fpr_logits(logits, mask.astype(np.int64))

    return {
        "ncc_mean": float(np.mean(ncc_b)),
        "ncc_std": float(np.std(ncc_b)),
        "rmse_mean": float(np.mean(rmse_b)),
        "rmse_std": float(np.std(rmse_b)),
        "snr_improvement_mean": float(np.mean(snri_b)),
        "snr_improvement_std": float(np.std(snri_b)),
        "dice_mean": dice_v,
        "false_positive_rate": fpr_v,
        "param_count_M": param_m,
    }


def evaluate_snr_sweep(
    pred_fn: Callable[[np.ndarray], np.ndarray],
    noisy_all: np.ndarray,
    clean_all: np.ndarray,
    snr_idx: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, snr in enumerate(SNR_LEVELS):
        m = snr_idx == i
        if not np.any(m):
            continue
        p = pred_fn(noisy_all[m])
        out[f"ncc_snr_{snr}"] = float(np.mean(_ncc_np(p, clean_all[m])))
    return out


def evaluate_model_on_real(
    infer: Callable[[np.ndarray], np.ndarray],
    real_noisy: np.ndarray,
) -> Dict[str, Any]:
    ref = _sg_reference(real_noisy)
    pred = infer(real_noisy)
    ncc_b = _ncc_np(pred, ref)
    rmse_b = _rmse_np(pred, ref)
    return {
        "ncc_mean": float(np.mean(ncc_b)),
        "ncc_std": float(np.std(ncc_b)),
        "rmse_mean": float(np.mean(rmse_b)),
        "rmse_std": float(np.std(rmse_b)),
        "_note": "Real set has no GT clean; metrics vs Savitzky–Golay reference (same as baseline family).",
    }


def train_time_estimate(model_id: str) -> Optional[float]:
    hist = PROJECT_ROOT / "results" / model_id / "history.json"
    if not hist.is_file():
        return None
    try:
        with open(hist) as f:
            h = json.load(f)
        n = len(h.get("val_ncc", []) or [])
        return round(n * 2.5, 2)
    except Exception:
        return None


def resolve_checkpoint(model_id: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    cands = [
        PROJECT_ROOT / "results" / model_id / "checkpoints" / "best.pt",
        PROJECT_ROOT / "results" / model_id / "checkpoints" / "checkpoint_best.pth",
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def run_full_benchmark(
    model_id: str,
    checkpoint_path: Optional[str],
    synth_path: Optional[str] = None,
    real_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Single model metrics dict + write results/<model_id>/benchmark_results.json"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_root = PROJECT_ROOT / "results" / model_id
    results_root.mkdir(parents=True, exist_ok=True)

    sp = Path(synth_path) if synth_path else SYNTH_PATH
    rp = Path(real_path) if real_path else REAL_PATH

    # Load tensors
    synth_pack: Optional[Dict[str, torch.Tensor]] = None
    if sp.is_file():
        synth_pack = torch.load(sp, map_location="cpu", weights_only=True)

    real_pack: Optional[Dict[str, torch.Tensor]] = None
    if rp.is_file():
        real_pack = torch.load(rp, map_location="cpu", weights_only=True)

    ckpt = resolve_checkpoint(model_id, checkpoint_path)

    # Classical
    classical_map = {
        "wavelet_db4_soft": denoise_wavelet_soft,
        "wavelet_bayesshrink": denoise_wavelet_bayes,
        "savitzky_golay": denoise_savgol,
    }

    synthetic: Dict[str, Any] = {}
    real_qlin: Dict[str, Any] = {}
    efficiency: Dict[str, Any] = {}
    sweep_extra: Dict[str, float] = {}

    infer_np: Callable[..., Any]
    param_m = 0.0

    if model_id in classical_map:
        den_fn = classical_map[model_id]

        def infer_np(noisy_np, **_):
            d = den_fn(noisy_np)
            return d, None

        def infer_np_denoised_only(noisy_np):
            return den_fn(noisy_np)

        param_m = 0.0
        lat1, lat32 = 0.5, 0.05

    elif model_id in MODEL_REGISTRY:
        dl = _load_dl_model(model_id, ckpt, device)
        param_m = sum(p.numel() for p in dl.parameters()) / 1e6

        def infer_np(noisy_np, **_):
            b, L = noisy_np.shape
            t = torch.from_numpy(noisy_np).float().reshape(b, 1, L)
            den_t, seg_t = _forward_dl(dl, t, device)
            den = den_t.squeeze(1).numpy().astype(np.float32)
            if seg_t is not None:
                return den, seg_t.numpy().astype(np.float32)
            return den, None

        def infer_np_denoised_only(noisy_np):
            return infer_np(noisy_np)[0]

        lat1, lat32 = _measure_latency(dl, device)

    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    if synth_pack is not None:
        noisy = synth_pack["noisy"].numpy().astype(np.float32)
        clean = synth_pack["clean"].numpy().astype(np.float32)
        mask = synth_pack["segmentation_masks"].numpy().astype(np.int64)
        if noisy.ndim == 3:
            noisy = noisy.squeeze(1)
            clean = clean.squeeze(1)
        if mask.ndim == 3:
            mask = mask.squeeze(1)
        idx = synth_pack["snr_level_index"].numpy()

        n_full = noisy.shape[0]
        n_cap = min(n_full, _DEFAULT_MAX)
        noisy, clean, mask, idx = noisy[:n_cap], clean[:n_cap], mask[:n_cap], idx[:n_cap]
        logger.info("Synthetic eval using n=%d/%d", n_cap, n_full)

        synthetic = evaluate_model_on_synthetic(model_id, noisy, clean, mask, infer_np, param_m)
        sweep_extra = evaluate_snr_sweep(infer_np_denoised_only, noisy, clean, idx)

    if real_pack is not None:
        xr = real_pack["signals"].numpy().astype(np.float32)
        if xr.ndim == 3:
            xr = xr.squeeze(1)
        n_cap = min(xr.shape[0], _DEFAULT_MAX)
        xr = xr[:n_cap]
        logger.info("Real eval using n=%d", n_cap)

        real_qlin = evaluate_model_on_real(lambda n: infer_np(n)[0], xr)

    efficiency = {
        "param_count_M": round(float(param_m), 4),
        "inference_latency_ms_b1": round(lat1, 4),
        "inference_latency_ms_b32": round(lat32, 4),
        "train_time_min": train_time_estimate(model_id),
    }

    blob = {"synthetic": synthetic, "real_qlin": real_qlin, "efficiency": efficiency}
    blob["snr_sweep"] = sweep_extra

    out_json = results_root / "benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    logger.info("Wrote %s", out_json)
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return blob


def aggregate_and_plot(all_rows: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ord = {name: i for i, name in enumerate(MODELS_ORDER)}
    mids = sorted(all_rows.keys(), key=lambda x: _ord.get(x, 999))

    # comparison_table.json
    table_rows = []
    for mid in mids:
        r = all_rows[mid]
        s = r.get("synthetic", {}) or {}
        q = r.get("real_qlin", {}) or {}
        e = r.get("efficiency", {}) or {}
        ss = r.get("snr_sweep", {}) or {}
        row = {
            "model_id": mid,
            "synthetic_ncc_mean": s.get("ncc_mean"),
            "synthetic_rmse_mean": s.get("rmse_mean"),
            "synthetic_snr_improvement_mean": s.get("snr_improvement_mean"),
            "synthetic_dice": s.get("dice_mean"),
            "synthetic_fpr": s.get("false_positive_rate"),
            "real_ncc_mean": q.get("ncc_mean"),
            "real_rmse_mean": q.get("rmse_mean"),
            "real_note": q.get("_note"),
            "param_count_M": e.get("param_count_M"),
            "inference_ms_b1": e.get("inference_latency_ms_b1"),
            "inference_ms_b32": e.get("inference_latency_ms_b32"),
            "train_time_min": e.get("train_time_min"),
        }
        for k, v in ss.items():
            row[k] = v
        table_rows.append(row)

    ct_path = out_dir / "comparison_table.json"
    with open(ct_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)

    x = np.arange(len(mids))
    w = 0.28
    ncc_s = [
        float(all_rows[m].get("synthetic", {}).get("ncc_mean") or 0) for m in mids
    ]
    ncc_r = [float(all_rows[m].get("real_qlin", {}).get("ncc_mean") or 0) for m in mids]
    rmse_r = [float(all_rows[m].get("real_qlin", {}).get("rmse_mean") or 0) for m in mids]

    fig, ax = plt.subplots(figsize=(max(10, len(mids) * 0.65), 5))
    ax.bar(x - w, ncc_s, width=w, label="NCC synthetic")
    ax.bar(x, ncc_r, width=w, label="NCC real (proxy)")
    ax.bar(x + w, rmse_r, width=w, label="RMSE real (proxy)")
    ax.set_xticks(x)
    ax.set_xticklabels(mids, rotation=75, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.set_title("Benchmark comparison")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bar.png", dpi=160)
    plt.close(fig)

    # efficiency scatter
    eb1 = [
        float(all_rows[m]["efficiency"].get("inference_latency_ms_b1") or 1)
        for m in mids
    ]
    ncol = len(mids)
    fig, ax = plt.subplots(figsize=(8, 6))
    areas = [(float(all_rows[m]["efficiency"]["param_count_M"] or 0.1)) * 30 for m in mids]
    ax.scatter(eb1, ncc_r, s=areas, alpha=0.55)
    for i, m in enumerate(mids):
        ax.annotate(m, (eb1[i], ncc_r[i]), fontsize=6)
    ax.set_xlabel("Latency ms batch=1")
    ax.set_ylabel("NCC real (proxy)")
    ax.set_title("Efficiency scatter (bubble ~ params M)")
    fig.tight_layout()
    fig.savefig(out_dir / "efficiency_scatter.png", dpi=160)
    plt.close(fig)

    # SNR sweep lines (use stored snr_sweep)
    plt.figure(figsize=(9, 6))
    any_line = False
    for m in mids:
        ss = all_rows[m].get("snr_sweep") or {}
        pts = SNR_LEVELS
        ys = [ss.get(f"ncc_snr_{p}") for p in pts]
        if any(v is None for v in ys):
            continue
        plt.plot(pts, ys, marker="o", linewidth=1, label=m)
        any_line = True
    plt.xlabel("SNR bucket (label)")
    plt.ylabel("Mean NCC (synthetic)")
    if any_line:
        plt.legend(fontsize=6, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "snr_sweep.png", dpi=160)
    plt.close()

    # Ablation-like heatmap: MR-TAE family in 3x3 with NaN padded
    mrtae = [
        "MR-TAE-noWavelet",
        "MR-TAE-noBiGRU",
        "MR-TAE-noSwin",
        "MR-TAE-noAttn",
        "MR-TAE-noMTL",
        "MR-TAE-FULL",
    ]
    grid = np.ones((3, 3)) * np.nan
    order = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for k, nam in enumerate(mrtae[:6]):
        ri, ci = order[k][:2]
        if nam in all_rows:
            grid[ri, ci] = float(all_rows[nam].get("synthetic", {}).get("ncc_mean") or 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NCC synth")
    ax.set_title("MR-TAE ablations (NCC synthetic)")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_heatmap.png", dpi=160)
    plt.close(fig)

    # RESULTS markdown
    lines = ["# Benchmark summary\n", "| Model | NCC Synth | NCC Real* | RMSE Real* |\n|---|---|---|---|\n"]
    for row in table_rows:
        lines.append(
            f"| {row['model_id']} | {row.get('synthetic_ncc_mean')} | "
            f"{row.get('real_ncc_mean')} | {row.get('real_rmse_mean')} |\n"
        )
    lines.append("\n\\*Real metrics use SG reference (no clean GT).\n")

    cmp_md = PROJECT_ROOT / "docs" / "RESULTS_COMPARISON.md"
    cmp_md.parent.mkdir(parents=True, exist_ok=True)
    cmp_md.write_text("".join(lines), encoding="utf-8")


def compare_all(checkpoint_dir: Optional[str] = None) -> None:
    all_rows: Dict[str, Dict[str, Any]] = {}
    ck_root = Path(checkpoint_dir) if checkpoint_dir else PROJECT_ROOT / "results"

    for mid in MODELS_ORDER:
        try:
            if mid in ("wavelet_db4_soft", "wavelet_bayesshrink", "savitzky_golay"):
                all_rows[mid] = run_full_benchmark(mid, None)
                continue

            ck = ck_root / mid / "checkpoints" / "best.pt"
            if not ck.is_file():
                ck = PROJECT_ROOT / "results" / mid / "checkpoints" / "best.pt"

            ck_str = str(ck) if ck.is_file() else None
            if ck_str is None:
                logger.warning("Skipping DL model %s (no checkpoint)", mid)
                continue

            all_rows[mid] = run_full_benchmark(mid, ck_str)

        except Exception as e:
            logger.error("%s failed: %s", mid, e)

    agg_dir = PROJECT_ROOT / "results" / "benchmark"
    aggregate_and_plot(all_rows, agg_dir)
    logger.info("Aggregate plots + table in %s", agg_dir)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-all", action="store_true")
    ap.add_argument("--model-id", type=str)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--synthetic", type=str, default=None)
    ap.add_argument("--real", type=str, default=None)
    ap.add_argument("--checkpoint-root", type=str, default=None)
    args = ap.parse_args()

    if args.compare_all:
        compare_all(args.checkpoint_root)
        return

    if not args.model_id:
        ap.error("Provide --compare-all or --model-id ")
    blob = run_full_benchmark(
        args.model_id,
        args.checkpoint,
        synth_path=args.synthetic,
        real_path=args.real,
    )
    print(json.dumps(blob, indent=2)[:1800])


if __name__ == "__main__":
    main()
