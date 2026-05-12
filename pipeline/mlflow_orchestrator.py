"""
pipeline/mlflow_orchestrator.py — Resilient Training Orchestrator

Supports:
  --resume          Resume from session_state.json (auto-detects where we stopped)
  --run-all         Train all models from scratch (or continue if session exists)
  --model <id>      Train a single model only
  --hparams <path>  Override hyperparameters from a yaml file

WiFi/crash resilience:
  Every epoch writes to session_state.json and logs/training_live.log.
  On restart, session is auto-loaded and training continues from last checkpoint.
  RESUME_PROMPT.md is updated every epoch so you always have a paste-ready
  instruction for the next Claude/Cursor session.

Usage:
    .venv\\Scripts\\activate   (Windows)
    python pipeline/mlflow_orchestrator.py --resume      # safest default
    python pipeline/mlflow_orchestrator.py --run-all
    python pipeline/mlflow_orchestrator.py --model MR-TAE-FULL
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import torch
import yaml

# --------------------------------------------------------------------------
# Make project root importable from anywhere
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training import RTX4070Manager, CurricularScheduler, Trainer
from training.resilience import ResilienceManager, get_resume_point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger("orchestrator")


# --------------------------------------------------------------------------
# Graceful interrupt handler
# --------------------------------------------------------------------------
_shutdown_requested = False


def _handle_sigint(sig, frame):
    global _shutdown_requested
    print("\n[INTERRUPT] Ctrl+C received — saving checkpoint and exiting cleanly...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_sigint)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_registry(path: str = "config/model_registry.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_training_config(
    path: str = "config/training_config.yaml",
    hparams_override: str | None = None,
) -> dict:
    cfg: dict = {}
    if Path(path).exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    if hparams_override and Path(hparams_override).exists():
        with open(hparams_override) as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)
        logger.info("Applied hyperparameter overrides from %s", hparams_override)
    return cfg


def build_model(model_id: str, registry: dict, training_cfg: dict) -> torch.nn.Module:
    """Instantiate model using AblationConfig flags from registry."""
    from models.base import build_model_from_registry
    return build_model_from_registry(model_id, registry, training_cfg)


def build_criterion(model, training_cfg: dict | None = None) -> torch.nn.Module:
    """Select loss function; reads charbonnier_eps / sigmas / dice from training_cfg."""
    from mr_tae_fusion.training.losses import MultiTaskLoss

    t = training_cfg or {}
    return MultiTaskLoss(
        initial_sigma_recon=float(t.get("initial_sigma_recon", 1.0)),
        initial_sigma_seg=float(t.get("initial_sigma_seg", 1.0)),
        dice_weight=float(t.get("dice_weight", 1.0)),
        charbonnier_eps=float(t.get("charbonnier_eps", 1e-3)),
    )


def build_loaders(
    batch_size: int,
    n_train: int = 4000,
    n_val: int = 800,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple:
    from mr_tae_fusion.data.dataset import PDSignalDataset
    from mr_tae_fusion.config import Config
    from torch.utils.data import DataLoader

    cfg = Config()
    train_ds = PDSignalDataset(config=cfg, num_samples=n_train, mode="train", seed=seed)
    val_ds = PDSignalDataset(config=cfg, num_samples=n_val, mode="val", seed=seed + 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------
# Single model run (delegates to Trainer.train())
# --------------------------------------------------------------------------

def run_model(
    model_id: str,
    registry: dict,
    training_cfg: dict,
    device: str = "cuda",
    results_root: str = "results",
    resilience: ResilienceManager | None = None,
    resume_ckpt: str | None = None,
    mlflow_run_id: str | None = None,
) -> dict:
    logger.info("=" * 60)
    logger.info("Model: %s", model_id)
    logger.info("=" * 60)

    mem = RTX4070Manager()
    mem.cleanup_between_models()

    model = build_model(model_id, registry, training_cfg)
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Params: %.2f M", n_params_M)

    batch_size = mem.get_safe_batch_size(n_params_M)
    logger.info("Batch size: %d", batch_size)

    tcfg = training_cfg
    criterion = build_criterion(model, tcfg)

    train_loader, val_loader = build_loaders(
        batch_size=batch_size,
        n_train=int(tcfg.get("n_train", 4000)),
        n_val=int(tcfg.get("n_val", 800)),
        seed=int(tcfg.get("seed", 42)),
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=bool(tcfg.get("pin_memory", True)),
    )

    scheduler = CurricularScheduler()

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=results_root,
        model_id=model_id,
        scheduler=scheduler,
        lr=float(tcfg.get("learning_rate", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        mlflow_run_id=mlflow_run_id,
        resilience=resilience,
    )

    history = trainer.train(resume_from=resume_ckpt, phase=None)

    # Run benchmark after training
    best_ckpt = str(trainer.ckpt_dir / "best.pt")
    _run_benchmark(model_id, best_ckpt)

    mem.cleanup_between_models()
    return history


def _run_benchmark(model_id: str, checkpoint_path: str) -> None:
    try:
        from evaluation.benchmark_runner import run_full_benchmark
        run_full_benchmark(model_id, checkpoint_path)
    except Exception as e:
        logger.warning("Benchmark failed for %s: %s", model_id, e)


# --------------------------------------------------------------------------
# Comparison report
# --------------------------------------------------------------------------

def generate_comparison_report(results_root: str = "results") -> None:
    """Aggregate all benchmark_results.json files into RESULTS_COMPARISON.md."""
    import json
    import glob

    rows = []
    for jf in sorted(glob.glob(f"{results_root}/*/benchmark_results.json")):
        try:
            with open(jf) as f:
                d = json.load(f)
            mid = Path(jf).parent.name
            rows.append((
                mid,
                d.get("efficiency", {}).get("param_count_M", "-"),
                d.get("synthetic", {}).get("ncc_mean", "-"),
                d.get("real_qlin", {}).get("ncc_mean", "-"),
                d.get("real_qlin", {}).get("rmse_mean", "-"),
                d.get("synthetic", {}).get("false_positive_rate", "-"),
                d.get("efficiency", {}).get("inference_latency_ms_b1", "-"),
            ))
        except Exception:
            pass

    lines = [
        "# Results Comparison\n",
        "",
        "| Model | Params(M) | NCC-Synth | NCC-Real | RMSE-Real | FPR | Latency(ms) |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n")

    out = Path("docs/RESULTS_COMPARISON.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(lines))
    logger.info("Comparison report written to %s", out)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    # Run relative to project root
    os.chdir(PROJECT_ROOT)
    Path("logs").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="AE-PD Resilient MLflow Orchestrator")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from session_state.json (recommended)")
    parser.add_argument("--run-all", action="store_true",
                        help="Train all registry models (continues if session exists)")
    parser.add_argument("--model", type=str, default=None,
                        help="Train a single model ID")
    parser.add_argument("--hparams", type=str, default=None,
                        help="YAML overrides for training config")
    parser.add_argument("--registry", type=str, default="config/model_registry.yaml")
    parser.add_argument("--training-config", type=str, default="config/training_config.yaml")
    parser.add_argument("--results", type=str, default=None,
                        help="Override results root")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Venv check
    if ".venv" not in sys.prefix and "myenv" not in sys.prefix:
        logger.warning("You may not be inside .venv. Run: .venv\\Scripts\\activate")

    print("\n[*] AE-PD Denoising -- Resilient Training Orchestrator")
    print(f"   CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.48, 0)
        print("   VRAM: Restricted to 48% to allow FYP_MASTER dual-training")
    print(f"   Session: session_state.json")
    print(f"   Live log: logs/training_live.log")
    print(f"   Resume prompt: RESUME_PROMPT.md\n")

    registry = load_registry(args.registry)
    training_cfg = load_training_config(args.training_config, args.hparams)
    results_root = args.results or str(training_cfg.get("results_root", "results"))

    # Determine queue
    models_dict = registry.get("models", registry)
    all_model_ids = list(models_dict.keys())

    def model_complexity(mid: str) -> int:
        e = models_dict[mid]
        return (int(e.get("use_bigru", 0)) + int(e.get("use_swin", 0))
                + int(e.get("use_wavelet_pooling", 0)) + int(e.get("use_attention_gates", 0)))

    all_model_ids.sort(key=model_complexity)

    if args.model:
        if args.model not in models_dict:
            logger.error("Model '%s' not in registry. Available: %s", args.model, list(models_dict.keys()))
            sys.exit(1)
        model_ids_to_run = [args.model]
    else:
        model_ids_to_run = all_model_ids

    # Resilience session
    resilience = ResilienceManager(project_root=".", model_id="")
    resilience.on_training_start(model_ids_to_run)
    resume_info = get_resume_point("session_state.json")
    skip_models = set(resume_info.get("skip_models", []))

    logger.info("Queue (%d models): %s", len(model_ids_to_run), model_ids_to_run)
    if skip_models:
        logger.info("Already done/skipped: %s", sorted(skip_models))
    if resume_info.get("resume_model"):
        logger.info("Will resume model: %s from epoch %s",
                     resume_info["resume_model"], resume_info.get("resume_epoch", 0))

    # ── Main loop ────────────────────────────────────────────────────────────
    for model_id in model_ids_to_run:
        if _shutdown_requested:
            print("[SHUTDOWN] Stopping before next model.")
            break

        if model_id in skip_models:
            logger.info("Skipping %s (done/skipped)", model_id)
            continue

        resilience.model_id = model_id
        resilience.on_model_start(model_id)

        # Decide checkpoint + MLflow run for resume
        model_state = None
        try:
            model_state = resilience.load_session()
        except Exception:
            model_state = None

        resume_ckpt = None
        mlflow_run_id = None
        if args.resume and model_state:
            resume_ckpt = model_state.get("best_checkpoint")
            mlflow_run_id = model_state.get("mlflow_run_id")

        try:
            history = run_model(
                model_id=model_id,
                registry=registry,
                training_cfg=training_cfg,
                device=args.device,
                results_root=results_root,
                resilience=resilience,
                resume_ckpt=resume_ckpt,
                mlflow_run_id=mlflow_run_id,
            )

            # Mark model complete
            try:
                final_metrics = {}
                if isinstance(history, dict) and history.get("val_ncc"):
                    final_metrics["val_ncc_best"] = max(history["val_ncc"])
                resilience.on_model_complete(model_id, final_metrics)
            except Exception:
                pass

        except Exception as e:
            logger.error("Model %s failed: %s", model_id, e, exc_info=True)
            try:
                resilience.on_model_complete(model_id, {"error": str(e)})
            except Exception:
                pass
            continue

        if _shutdown_requested:
            print("[SHUTDOWN] Training interrupted. Run with --resume to continue.")
            break

    generate_comparison_report(results_root)

    if not _shutdown_requested:
        print("\n[COMPLETE] All models trained.")
        print("Run: python evaluation/benchmark_runner.py --compare-all")
        print("Dashboard: python dashboard/app.py")


if __name__ == "__main__":
    main()
