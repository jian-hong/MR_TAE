"""
training/heat_monitor.py — Background GPU temperature watchdog thread.

Usage:
    monitor = HeatMonitor()
    monitor.start()
    ...
    # In training loop:
    if monitor.too_hot.is_set():
        monitor.wait_cool()
    ...
    monitor.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class HeatMonitor(threading.Thread):
    """
    Daemon thread that polls GPU temperature every CHECK_INTERVAL_S seconds.

    Sets  `too_hot`  threading.Event when temp >= PAUSE_C.
    Clears `too_hot`  threading.Event when temp drops to <= RESUME_C.

    The trainer checks `monitor.too_hot.is_set()` at every batch start.
    """

    WARN_C:   int   = 78
    PAUSE_C:  int   = 82
    RESUME_C: int   = 72
    CHECK_INTERVAL_S: float = 30.0
    LOG_INTERVAL_CHECKS: int = 4   # log temp every 4 checks (~2 min)

    def __init__(self, log_file: Optional[str] = None) -> None:
        super().__init__(daemon=True, name="HeatMonitor")
        self.too_hot: threading.Event = threading.Event()
        self._stop_flag: threading.Event = threading.Event()
        self._handle = None
        self._log_file = log_file
        self._check_count = 0

        if _NVML_OK:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

    # ------------------------------------------------------------------

    def _get_temp(self) -> Optional[int]:
        if self._handle is None:
            return None
        try:
            return pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            return None

    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("HeatMonitor started (pause=%d°C, resume=%d°C).",
                    self.PAUSE_C, self.RESUME_C)
        while not self._stop_flag.is_set():
            temp = self._get_temp()
            self._check_count += 1

            if temp is not None:
                if self._check_count % self.LOG_INTERVAL_CHECKS == 0:
                    logger.debug("GPU temp: %d°C", temp)

                if self._log_file:
                    try:
                        with open(self._log_file, "a") as f:
                            f.write(f"{time.strftime('%H:%M:%S')} GPU {temp}°C\n")
                    except OSError:
                        pass

                if temp >= self.PAUSE_C:
                    if not self.too_hot.is_set():
                        logger.warning(
                            "GPU overheating: %d°C >= %d°C — "
                            "setting too_hot flag, trainer will pause.",
                            temp, self.PAUSE_C,
                        )
                    self.too_hot.set()

                elif temp <= self.RESUME_C and self.too_hot.is_set():
                    logger.info(
                        "GPU cooled to %d°C — clearing too_hot flag, "
                        "training will resume.", temp
                    )
                    self.too_hot.clear()

                elif temp >= self.WARN_C:
                    logger.warning("GPU temp warning: %d°C", temp)

            self._stop_flag.wait(timeout=self.CHECK_INTERVAL_S)

        logger.info("HeatMonitor stopped.")

    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop_flag.set()

    @property
    def current_temp(self) -> int:
        """Last read GPU temperature (°C), or 0 if unknown."""
        t = self._get_temp()
        return int(t) if t is not None else 0

    def wait_if_hot(self) -> None:
        """Alias for orchestrator: block until GPU is cool enough."""
        self.wait_cool()

    def wait_cool(self, poll_s: float = 5.0) -> None:
        """Block the calling thread until GPU is no longer too hot."""
        while self.too_hot.is_set():
            logger.info("Waiting for GPU to cool …")
            time.sleep(poll_s)
