# MR-TAE-GEM: Gradient Episodic Memory Enhanced Training
# Fixes for: Validation Mirage, Catastrophic Forgetting, Phantom TGAN

from .train_mr_tae_gem import main as train_main
from .gem_dataset import GEMDataset
from .colored_noise import ColoredNoiseGenerator, FlickerNoise, SubstationNoise

__all__ = [
    'train_main',
    'GEMDataset', 
    'ColoredNoiseGenerator',
    'FlickerNoise',
    'SubstationNoise'
]
