"""
Optuna Hyperband study for MR-TAE-FULL (Phase-1 proxy: warm SNR range only).

Uses models.base.build_model_from_registry + mr_tae_fusion training loop.
Storage: sqlite:///hpo/optuna.db (resumable)

Usage:
  .venv\\Scripts\\python.exe hpo/optuna_study.py --n-trials 10
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

try:
    from optuna.integration import MLflowCallback

    _MLFLOW_CB = True
except ImportError:
    _MLFLOW_CB = False

from mr_tae_fusion.config import Config
from mr_tae_fusion.data.dataset import PDSignalDataset
from mr_tae_fusion.training.losses import MultiTaskLoss
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.curricular_scheduler import CurricularScheduler, PhaseConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("optuna")

PHASE1_EPOCHS = 30
MODEL_ID = "MR-TAE-FULL"


def objective(trial: optuna.Trial) -> float:
    from models.base import build_model_from_registry

    reg_path = PROJECT_ROOT / "config/model_registry.yaml"
    with open(reg_path) as f:
        registry = yaml.safe_load(f)
    base_cfg_path = PROJECT_ROOT / "config/training_config.yaml"
    training_cfg: dict = {}
    if base_cfg_path.is_file():
        with open(base_cfg_path) as f:
            training_cfg = yaml.safe_load(f) or {}

    encoder_ch = trial.suggest_categorical("encoder_base_ch", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    charb = trial.suggest_float("charbonnier_eps", 1e-5, 1e-2, log=True)

    training_cfg["encoder_base_ch"] = encoder_ch
    training_cfg["learning_rate"] = lr
    training_cfg["weight_decay"] = weight_decay
    training_cfg["charbonnier_eps"] = charb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_registry(MODEL_ID, registry, training_cfg).to(device)
    criterion = MultiTaskLoss(
        charbonnier_eps=charb,
        initial_sigma_recon=float(training_cfg.get("initial_sigma_recon", 1.0)),
        initial_sigma_seg=float(training_cfg.get("initial_sigma_seg", 1.0)),
        dice_weight=float(training_cfg.get("dice_weight", 1.0)),
    ).to(device)

    cfg = Config()
    train_ds = PDSignalDataset(config=cfg, num_samples=2000, mode="train", seed=42)
    val_ds = PDSignalDataset(config=cfg, num_samples=400, mode="val", seed=43)

    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    bs = 64 if n_params_m < 25 else 32

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = GradScaler() if device.type == "cuda" else None

    sched = CurricularScheduler(
        phases=[
            PhaseConfig(5.0, 15.0, epochs=PHASE1_EPOCHS, noise_type="wgn"),
        ]
    )

    best_ncc = -1.0

    for epoch in range(PHASE1_EPOCHS):
        train_ds.set_curriculum_override(sched.get_snr_range(epoch), sched.get_noise_type(epoch))
        val_ds.set_curriculum_override(sched.get_snr_range(epoch), sched.get_noise_type(epoch))
        train_ds.update_epoch(epoch)
        val_ds.update_epoch(epoch)

        model.train()
        criterion.train()
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            mask = batch["mask"].long().to(device).clamp(0, 3)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with autocast(enabled=device.type == "cuda"):
                    pred, pred_seg = model(noisy)
                    loss, _ = criterion(pred, clean, pred_seg, mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, pred_seg = model(noisy)
                loss, _ = criterion(pred, clean, pred_seg, mask)
                loss.backward()
                optimizer.step()

        model.eval()
        total_ncc = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                pred, _ = model(noisy)
                p = pred.float().flatten(1) - pred.float().flatten(1).mean(1, keepdim=True)
                t = clean.float().flatten(1) - clean.float().flatten(1).mean(1, keepdim=True)
                ncc = (p * t).sum(1) / (
                    torch.sqrt((p**2).sum(1) * (t**2).sum(1)) + 1e-8
                )
                total_ncc += ncc.mean().item()
                n_batches += 1

        val_ncc = total_ncc / max(n_batches, 1)
        best_ncc = max(best_ncc, val_ncc)
        trial.report(val_ncc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    del model, criterion, optimizer, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_ncc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--study-name", type=str, default="mr-tae-full-hpo")
    ap.add_argument(
        "--storage",
        type=str,
        default="sqlite:///" + (PROJECT_ROOT / "hpo" / "optuna.db").resolve().as_posix(),
    )
    ap.add_argument("--out", type=str, default="config/best_hparams.yaml")
    args = ap.parse_args()

    (PROJECT_ROOT / "hpo").mkdir(parents=True, exist_ok=True)

    callbacks = []
    if _MLFLOW_CB:
        callbacks.append(MLflowCallback(metric_name="val_ncc", create_experiment=True))

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner(
            min_resource=5, max_resource=PHASE1_EPOCHS, reduction_factor=3
        ),
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        objective, n_trials=args.n_trials, n_jobs=1, callbacks=callbacks
    )

    best = study.best_trial
    logger.info("Best trial %s val_ncc=%.4f params=%s", best.number, best.value, best.params)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(
            {"best_trial": best.number, "val_ncc": best.value, "params": best.params},
            f,
            default_flow_style=False,
        )

    try:
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as ovm

        fig = ovm.plot_param_importances(study)
        imp_path = PROJECT_ROOT / "results" / "hpo"
        imp_path.mkdir(parents=True, exist_ok=True)
        fig.figure.savefig(imp_path / "param_importance.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        logger.warning("Could not save param importance plot: %s", e)


if __name__ == "__main__":
    main()
