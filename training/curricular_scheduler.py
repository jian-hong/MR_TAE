"""
training/curricular_scheduler.py — Phase-aware SNR curriculum.

Phase 1: SNR [ 5, 15] dB  (easy)
Phase 2: SNR [-5,  5] dB  (medium)
Phase 3: SNR [-20,-5] dB  (hard)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhaseConfig:
    snr_min: float
    snr_max: float
    epochs: int
    noise_type: str   # 'wgn' | 'wgn_impulsive' | 'composite'


PHASE_DEFAULTS = [
    PhaseConfig(snr_min= 5.0, snr_max=15.0, epochs=30, noise_type="wgn"),
    PhaseConfig(snr_min=-5.0, snr_max= 5.0, epochs=30, noise_type="wgn_impulsive"),
    PhaseConfig(snr_min=-20.0, snr_max=-5.0, epochs=40, noise_type="composite"),
]


class CurricularScheduler:
    """
    Maps absolute epoch number → (phase_index, snr_range, noise_type).
    Phase indices are 1-based (1, 2, 3).
    """

    def __init__(self, phases: list[PhaseConfig] | None = None) -> None:
        self.phases = phases or PHASE_DEFAULTS
        # cumulative epoch boundaries
        self._boundaries: list[int] = []
        total = 0
        for p in self.phases:
            total += p.epochs
            self._boundaries.append(total)
        self.total_epochs = total

    def get_phase(self, epoch: int) -> int:
        """Return 1-based phase index for the given epoch (0-based)."""
        for i, boundary in enumerate(self._boundaries):
            if epoch < boundary:
                return i + 1
        return len(self.phases)

    def get_snr_range(self, epoch: int) -> Tuple[float, float]:
        p = self.phases[self.get_phase(epoch) - 1]
        return p.snr_min, p.snr_max

    def get_noise_type(self, epoch: int) -> str:
        return self.phases[self.get_phase(epoch) - 1].noise_type

    def info(self, epoch: int) -> dict:
        phase = self.get_phase(epoch)
        snr = self.get_snr_range(epoch)
        return {
            "phase": phase,
            "snr_range": snr,
            "noise_type": self.get_noise_type(epoch),
        }
