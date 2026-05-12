"""
training/memory_manager.py — RTX 4070 Laptop GPU memory and OOM management.

Prevents OOM and handles batch-size fallback gracefully.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class RTX4070Manager:
    """
    Safe training manager for RTX 4070 Laptop (12 GB VRAM).

    Rules enforced:
    - Always use AMP (torch.cuda.amp)
    - Always use gradient checkpointing for models > 40 M params
    - Monitor GPU temp every CHECK_INTERVAL calls; pause if > PAUSE_C
    - Auto-reduce batch size by 50 % on OOM, then retry
    - Log VRAM peak after each epoch via return value
    """

    MAX_VRAM_GB = 11.0
    TEMP_WARNING_C = 78
    TEMP_PAUSE_C = 82
    TEMP_RESUME_C = 72

    # Suggested safe batch sizes indexed by model param count (M)
    _BATCH_TABLE = [
        (5,  128),
        (20,  64),
        (50,  32),
        (999, 16),
    ]

    def __init__(self) -> None:
        self._handle = None
        if _NVML_OK and torch.cuda.is_available():
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Batch size recommendation
    # ------------------------------------------------------------------

    def get_safe_batch_size(self, model_param_count_M: float) -> int:
        for threshold, bs in self._BATCH_TABLE:
            if model_param_count_M < threshold:
                return bs
        return 16

    # ------------------------------------------------------------------
    # Temperature monitoring
    # ------------------------------------------------------------------

    def get_gpu_temp(self) -> Optional[int]:
        if self._handle is None:
            return None
        try:
            return pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            return None

    def check_thermal(self) -> bool:
        """Return True if safe to train, False if overheating."""
        temp = self.get_gpu_temp()
        if temp is None:
            return True
        if temp >= self.TEMP_PAUSE_C:
            logger.warning("GPU temp %d°C >= %d°C — pausing training.", temp, self.TEMP_PAUSE_C)
            return False
        if temp >= self.TEMP_WARNING_C:
            logger.warning("GPU temp %d°C — approaching throttle limit.", temp)
        return True

    def wait_for_cool(self, poll_interval_s: float = 10.0) -> None:
        """Block until GPU cools below TEMP_RESUME_C."""
        import time
        while True:
            temp = self.get_gpu_temp()
            if temp is None or temp <= self.TEMP_RESUME_C:
                break
            logger.info("GPU at %d°C, waiting to cool below %d°C …", temp, self.TEMP_RESUME_C)
            time.sleep(poll_interval_s)

    # ------------------------------------------------------------------
    # VRAM reporting
    # ------------------------------------------------------------------

    def get_vram_used_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024 ** 3

    def get_vram_peak_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024 ** 3

    def reset_peak_vram(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # OOM recovery
    # ------------------------------------------------------------------

    def handle_oom(self, current_batch_size: int) -> int:
        """
        Call inside an except RuntimeError (OOM) block.
        Clears CUDA cache, halves batch size, returns new size.
        Raises RuntimeError if batch size would drop below 4.
        """
        torch.cuda.empty_cache()
        gc.collect()
        new_bs = max(4, current_batch_size // 2)
        if new_bs == current_batch_size:
            raise RuntimeError("OOM at minimum batch size 4 — cannot recover.")
        logger.warning(
            "OOM caught — reducing batch size: %d → %d", current_batch_size, new_bs
        )
        return new_bs

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    @staticmethod
    def should_use_grad_ckpt(model: torch.nn.Module, force: bool = False) -> bool:
        n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
        return force or n_params_M > 40.0

    @staticmethod
    def enable_grad_ckpt(model: torch.nn.Module) -> None:
        """Enable gradient checkpointing on all supported sub-modules."""
        enabled = 0
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
                enabled += 1
        if enabled:
            logger.info("Gradient checkpointing enabled on %d sub-modules.", enabled)
        else:
            logger.info(
                "Model has no gradient_checkpointing_enable(); "
                "use torch.utils.checkpoint.checkpoint() manually in forward()."
            )

    # ------------------------------------------------------------------
    # Cleanup between model runs
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_between_models() -> None:
        """Free VRAM fully between training different model variants."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # MLflow-ready stats dict
    # ------------------------------------------------------------------

    def get_gpu_metrics(self) -> dict:
        return {
            "gpu/vram_used_gb":  round(self.get_vram_used_gb(), 3),
            "gpu/vram_peak_gb":  round(self.get_vram_peak_gb(), 3),
            "gpu/temp_c":        self.get_gpu_temp() or 0,
        }
