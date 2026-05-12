"""
training/resilience.py — Session Persistence & Smart Early Stopping

Handles:
  1. Resume-from-checkpoint after any crash, WiFi drop, or Cursor restart
  2. Dynamic early stopping with adaptive threshold (handles cold-start)
  3. Persistent JSON session log that survives process death
  4. Auto-prompt file written after every epoch so the agent can pick up exactly
     where it left off, even mid-queue

Usage inside trainer:
    resilience = ResilienceManager(project_root=".", model_id="MR-TAE-noSwin")
    resume_state = resilience.load_session()   # None if fresh start
    ...
    resilience.on_epoch_end(epoch, metrics)    # call every epoch
    resilience.on_model_complete(model_id, results)
"""

import os
import json
import time
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading


# ---------------------------------------------------------------------------
# Session state — written to disk after every epoch
# ---------------------------------------------------------------------------
SESSION_FILE = "session_state.json"   # project root — survives crashes


@dataclass
class EpochRecord:
    epoch: int
    phase: int
    train_loss: float
    val_ncc: float
    val_rmse: float
    timestamp: str
    checkpoint_path: Optional[str] = None


@dataclass
class ModelSession:
    model_id: str
    status: str           # "running" | "done" | "skipped" | "queued"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    best_ncc: float = 0.0
    best_epoch: int = 0
    best_checkpoint: Optional[str] = None
    current_epoch: int = 0
    current_phase: int = 1
    skip_reason: Optional[str] = None
    epochs: List[dict] = None

    def __post_init__(self):
        if self.epochs is None:
            self.epochs = []


class EarlyStopException(Exception):
    """Raised when resilience manager decides to skip a model."""
    pass


# ---------------------------------------------------------------------------
# Main resilience manager
# ---------------------------------------------------------------------------
class ResilienceManager:
    """
    Drop this into the trainer. Call on_epoch_end() every epoch.
    Everything is written to session_state.json atomically so a crash
    at any point leaves the file in a valid state.
    """

    def __init__(self, project_root: str = ".", model_id: str = ""):
        self.root = Path(project_root)
        self.model_id = model_id
        self.session_path = self.root / SESSION_FILE
        self.log_path = self.root / "logs" / "training_live.log"
        self.prompt_path = self.root / "RESUME_PROMPT.md"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._session: Dict[str, Any] = self._load_or_init_session()

        # Dynamic early stopping state
        self._ncc_history: List[float] = []
        self._dynamic_threshold: Optional[float] = None

    # -----------------------------------------------------------------------
    # Session load / init
    # -----------------------------------------------------------------------
    def _load_or_init_session(self) -> dict:
        if self.session_path.exists():
            try:
                with open(self.session_path) as f:
                    data = json.load(f)
                self._log(f"[RESUME] Loaded existing session: {self.session_path}")
                return data
            except (json.JSONDecodeError, KeyError):
                self._log("[WARN] session_state.json corrupted — starting fresh")

        return {
            "project": "AE-PD-Denoising",
            "created": datetime.now().isoformat(),
            "last_updated": None,
            "queue": [],          # ordered list of model_ids to train
            "models": {},         # model_id -> ModelSession dict
            "completed_tasks": [],# tasks beyond training that are done
            "notes": [],
        }

    def _save_session(self):
        """Atomic write: write to .tmp then rename — crash-safe."""
        self._session["last_updated"] = datetime.now().isoformat()
        tmp = self.session_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._session, f, indent=2)
        # On Windows, we need to handle the case where target exists
        if self.session_path.exists():
            self.session_path.unlink()
        tmp.rename(self.session_path)

    # -----------------------------------------------------------------------
    # Public API — called by trainer
    # -----------------------------------------------------------------------
    def load_session(self) -> Optional[dict]:
        """
        Returns resume state for current model_id, or None for fresh start.
        
        Usage:
            state = resilience.load_session()
            start_epoch = state["current_epoch"] + 1 if state else 0
            start_phase = state["current_phase"] if state else 1
        """
        model_data = self._session["models"].get(self.model_id)
        if model_data and model_data.get("status") == "running":
            ep = model_data.get("current_epoch", 0)
            ph = model_data.get("current_phase", 1)
            ckpt = model_data.get("best_checkpoint")
            self._log(f"[RESUME] {self.model_id} — resuming from epoch {ep}, phase {ph}")
            if ckpt:
                self._log(f"[RESUME] Loading checkpoint: {ckpt}")
            # Restore NCC history for dynamic early stopping
            self._ncc_history = [r["val_ncc"] for r in model_data.get("epochs", [])]
            return model_data
        return None

    def on_training_start(self, queue: List[str]):
        """Call once before the orchestrator loop."""
        with self._lock:
            if not self._session["queue"]:
                self._session["queue"] = queue
            for mid in queue:
                if mid not in self._session["models"]:
                    self._session["models"][mid] = asdict(ModelSession(
                        model_id=mid, status="queued"
                    ))
            self._save_session()
            self._update_resume_prompt()

    def on_model_start(self, model_id: str):
        with self._lock:
            m = self._session["models"].setdefault(model_id, asdict(
                ModelSession(model_id=model_id, status="running")
            ))
            m["status"] = "running"
            m["start_time"] = datetime.now().isoformat()
            self._ncc_history = []
            self._dynamic_threshold = None
            self._save_session()
            self._log(f"[START] {model_id}")
            self._update_resume_prompt()

    def on_epoch_end(self, epoch: int, phase: int, metrics: dict) -> dict:
        """
        Called after every epoch. Returns a dict:
            {"action": "continue" | "skip_model", "reason": str}
        
        Trainer must respect "skip_model" and move to next model.
        """
        val_ncc = metrics.get("val_ncc", metrics.get("val/ncc", 0.0))
        train_loss = metrics.get("train_loss", metrics.get("train/loss", 0.0))
        val_rmse = metrics.get("val_rmse", metrics.get("val/rmse", 999.0))

        with self._lock:
            m = self._session["models"].setdefault(
                self.model_id, asdict(ModelSession(model_id=self.model_id, status="running"))
            )
            # Update current position
            m["current_epoch"] = epoch
            m["current_phase"] = phase

            # Track best
            if val_ncc > m.get("best_ncc", 0.0):
                m["best_ncc"] = val_ncc
                m["best_epoch"] = epoch

            # Append epoch record
            record = {
                "epoch": epoch, "phase": phase,
                "train_loss": round(train_loss, 6),
                "val_ncc": round(val_ncc, 6),
                "val_rmse": round(val_rmse, 6),
                "timestamp": datetime.now().isoformat(),
            }
            m.setdefault("epochs", []).append(record)

            # Save + refresh log/prompt every epoch
            self._save_session()
            self._append_live_log(epoch, phase, metrics)
            self._update_resume_prompt()

        # Dynamic early stopping check
        self._ncc_history.append(val_ncc)
        decision = self._check_early_stop(epoch, val_ncc)
        if decision["action"] == "skip_model":
            with self._lock:
                self._session["models"][self.model_id]["status"] = "skipped"
                self._session["models"][self.model_id]["skip_reason"] = decision["reason"]
                self._save_session()
                self._update_resume_prompt()
            self._log(f"[SKIP] {self.model_id} — {decision['reason']}")
        return decision

    def on_checkpoint_saved(self, epoch: int, checkpoint_path: str):
        with self._lock:
            m = self._session["models"].get(self.model_id, {})
            m["best_checkpoint"] = str(checkpoint_path)
            self._save_session()
        self._log(f"[CKPT] Saved {checkpoint_path} (epoch {epoch})")

    def set_mlflow_run_id(self, run_id: str) -> None:
        """Persist MLflow run id so resumed training can continue the same run."""
        if not run_id:
            return
        with self._lock:
            m = self._session["models"].setdefault(
                self.model_id, asdict(ModelSession(model_id=self.model_id, status="running"))
            )
            m["mlflow_run_id"] = run_id
            self._save_session()

    def get_mlflow_run_id(self) -> Optional[str]:
        with self._lock:
            m = self._session["models"].get(self.model_id, {})
            return m.get("mlflow_run_id")

    def on_model_complete(self, model_id: str, final_metrics: dict):
        with self._lock:
            m = self._session["models"].get(model_id, {})
            m["status"] = "done"
            m["end_time"] = datetime.now().isoformat()
            m.update({f"final_{k}": v for k, v in final_metrics.items()})
            self._save_session()
            self._log(f"[DONE] {model_id} — NCC={final_metrics.get('test_real_ncc', '?')}")
            self._update_resume_prompt()

    def add_note(self, note: str):
        """Add a human-readable note to the session (e.g. 'WiFi dropped at 14:32')."""
        with self._lock:
            self._session.setdefault("notes", []).append({
                "time": datetime.now().isoformat(), "note": note
            })
            self._save_session()

    # -----------------------------------------------------------------------
    # Dynamic early stopping
    # -----------------------------------------------------------------------
    def _check_early_stop(self, epoch: int, current_ncc: float) -> dict:
        """
        Dynamic early stopping that adapts its threshold based on actual training history.

        Cold start problem: if only 5 epochs have run, we have no population statistics
        yet, so we never stop early before MIN_EPOCHS_BEFORE_STOP.

        After MIN_EPOCHS_BEFORE_STOP:
          - Compute the best NCC improvement per epoch over a rolling window
          - Set threshold = mean_improvement * THRESHOLD_FACTOR
          - If recent improvement < threshold for PATIENCE epochs: skip model

        This avoids the trap of stopping a model that just trains slowly.
        """
        MIN_EPOCHS_BEFORE_STOP = 15   # never stop before this many epochs
        WINDOW = 10                    # rolling window for improvement rate
        PATIENCE = 8                   # consecutive epochs below threshold before stop
        THRESHOLD_FACTOR = 0.05        # improvement must be >= 5% of mean improvement rate

        if epoch < MIN_EPOCHS_BEFORE_STOP:
            return {"action": "continue", "reason": "cold start — too early to judge"}

        if len(self._ncc_history) < WINDOW + PATIENCE:
            return {"action": "continue", "reason": "insufficient history"}

        # Compute improvement rates over the full history so far
        improvements = [
            max(0, self._ncc_history[i] - self._ncc_history[i-1])
            for i in range(1, len(self._ncc_history))
        ]

        if not improvements or max(improvements) == 0:
            # NCC never improved at all — this model is stuck
            return {
                "action": "skip_model",
                "reason": f"NCC never improved over {epoch} epochs (stuck at {current_ncc:.4f})"
            }

        # Dynamic threshold: fraction of the mean improvement rate seen so far
        mean_improvement = sum(improvements) / len(improvements)
        if self._dynamic_threshold is None or epoch % 10 == 0:
            # Recalibrate every 10 epochs
            self._dynamic_threshold = mean_improvement * THRESHOLD_FACTOR
            self._log(
                f"[THRESHOLD] {self.model_id} epoch {epoch}: "
                f"mean_improve={mean_improvement:.6f} "
                f"threshold={self._dynamic_threshold:.6f}"
            )

        # Check if last PATIENCE epochs are all below threshold
        recent = improvements[-PATIENCE:]
        if all(imp < self._dynamic_threshold for imp in recent):
            return {
                "action": "skip_model",
                "reason": (
                    f"NCC improvement ({max(recent):.6f}) below dynamic threshold "
                    f"({self._dynamic_threshold:.6f}) for {PATIENCE} consecutive epochs. "
                    f"Best NCC={max(self._ncc_history):.4f} at epoch {self._ncc_history.index(max(self._ncc_history))+1}"
                )
            }

        return {"action": "continue", "reason": "ok"}

    # -----------------------------------------------------------------------
    # Live log + resume prompt
    # -----------------------------------------------------------------------
    def _log(self, message: str):
        """Append to the live log file."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"
        with open(self.log_path, "a") as f:
            f.write(line)

    def _append_live_log(self, epoch: int, phase: int, metrics: dict):
        ncc = metrics.get("val_ncc", metrics.get("val/ncc", 0))
        rmse = metrics.get("val_rmse", metrics.get("val/rmse", 0))
        loss = metrics.get("train_loss", metrics.get("train/loss", 0))
        gpu_t = metrics.get("gpu_temp", metrics.get("gpu/temp_c", 0))
        vram = metrics.get("vram_gb", metrics.get("gpu/vram_used_gb", 0))
        self._log(
            f"[EPOCH] {self.model_id} | ep={epoch} ph={phase} | "
            f"loss={loss:.4f} ncc={ncc:.4f} rmse={rmse:.4f} | "
            f"gpu={gpu_t}°C vram={vram:.1f}GB"
        )

    def _update_resume_prompt(self):
        """
        Write RESUME_PROMPT.md to the project root.
        This file contains an exact natural-language instruction for the agent
        (or human) to paste into a new Cursor/Claude session to continue from
        exactly where it left off.
        """
        s = self._session
        models = s.get("models", {})
        queue = s.get("queue", [])
        notes = s.get("notes", [])

        # Determine current state
        running = [m for m, d in models.items() if d.get("status") == "running"]
        done = [m for m, d in models.items() if d.get("status") == "done"]
        skipped = [m for m, d in models.items() if d.get("status") == "skipped"]
        queued = [m for m, d in models.items() if d.get("status") == "queued"]

        lines = [
            "# RESUME PROMPT",
            f"*Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Last update: {s.get('last_updated', 'never')}*",
            "",
            "---",
            "",
            "## Paste this into a new Claude / Cursor session to continue:",
            "",
            "```",
            "I am resuming the AE-PD Denoising research pipeline.",
            f"Project: Intelligent Partial Discharge Denoising — Oo Jian Hong, University of Malaya 2026",
            "",
            "## Current state (from session_state.json):",
            "",
        ]

        if running:
            m_data = models[running[0]]
            ep = m_data.get("current_epoch", 0)
            ph = m_data.get("current_phase", 1)
            ncc = m_data.get("best_ncc", 0)
            ckpt = m_data.get("best_checkpoint", "none")
            lines += [
                f"ACTIVE MODEL: {running[0]}",
                f"  Stopped at: epoch {ep}, phase {ph}",
                f"  Best NCC so far: {ncc:.4f}",
                f"  Resume checkpoint: {ckpt}",
                "",
            ]

        lines += [
            f"COMPLETED: {', '.join(done) if done else 'none'}",
            f"SKIPPED:   {', '.join(skipped) if skipped else 'none'}",
            f"REMAINING: {', '.join(queued) if queued else 'none'}",
            "",
        ]

        for sk in skipped:
            reason = models[sk].get("skip_reason", "")
            lines.append(f"  {sk} was skipped because: {reason}")

        if notes:
            lines.append("")
            lines.append("NOTES:")
            for n in notes[-5:]:  # last 5 notes
                lines.append(f"  [{n['time'][:16]}] {n['note']}")

        lines += [
            "",
            "## Instructions for the agent:",
            "",
            "1. Activate venv: .venv\\Scripts\\activate (Windows) or source .venv/bin/activate (Linux)",
            "2. Verify: python -c \"import torch; print(torch.cuda.get_device_name(0))\"",
            "3. Read session_state.json — it has the full state of every model",
            "4. If any model has status='running': resume it from its checkpoint with",
            "   python pipeline/mlflow_orchestrator.py --resume",
            "5. If no model is running: continue with the next 'queued' model",
            "6. Do NOT restart already-completed or skipped models",
            "7. After all models done: run evaluation/benchmark_runner.py --compare-all",
            "",
            "The session_state.json and logs/training_live.log have full details.",
            "```",
            "",
            "---",
            "",
            "## Quick status table",
            "",
            "| Model | Status | Best NCC | Epoch | Skip reason |",
            "|---|---|---|---|---|",
        ]

        for mid in queue:
            d = models.get(mid, {})
            st = d.get("status", "queued")
            ncc = f"{d.get('best_ncc', 0):.4f}" if d.get("best_ncc") else "—"
            ep = str(d.get("current_epoch", 0))
            reason = d.get("skip_reason", "—") or "—"
            lines.append(f"| {mid} | {st} | {ncc} | {ep} | {reason[:50]} |")

        lines += [
            "",
            "---",
            f"*Log file: logs/training_live.log*",
            f"*Session file: session_state.json*",
            f"*To add a note: resilience.add_note('your message')*",
        ]

        with open(self.prompt_path, "w") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Orchestrator integration helper
# ---------------------------------------------------------------------------
def get_resume_point(session_path: str = "session_state.json") -> dict:
    """
    Standalone function — call at orchestrator startup to get resume instructions.
    
    Returns:
        {
          "resume_model": "MR-TAE-noSwin" or None,
          "resume_epoch": 23,
          "resume_phase": 1,
          "resume_checkpoint": "checkpoints/MR-TAE-noSwin/best.pth",
          "skip_models": ["MR-TAE-noAttn"],   # already done or explicitly skipped
          "pending_models": ["MR-TAE-noBiGRU", ...],
        }
    """
    p = Path(session_path)
    if not p.exists():
        return {"resume_model": None, "skip_models": [], "pending_models": []}

    with open(p) as f:
        s = json.load(f)

    models = s.get("models", {})
    queue = s.get("queue", [])

    skip = [m for m in queue if models.get(m, {}).get("status") in ("done", "skipped")]
    running = [m for m in queue if models.get(m, {}).get("status") == "running"]
    pending = [m for m in queue if models.get(m, {}).get("status") in ("queued", None)]

    if running:
        rm = running[0]
        d = models[rm]
        return {
            "resume_model": rm,
            "resume_epoch": d.get("current_epoch", 0),
            "resume_phase": d.get("current_phase", 1),
            "resume_checkpoint": d.get("best_checkpoint"),
            "skip_models": skip,
            "pending_models": pending,
        }

    return {
        "resume_model": None,
        "skip_models": skip,
        "pending_models": pending,
    }
