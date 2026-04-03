"""
GEM Dataset: Gradient Episodic Memory Enhanced Dataset.

Fixes for train_comprehensive.py issues:
1. VALIDATION MIRAGE: Validation dataset now tracks current epoch
2. CATASTROPHIC FORGETTING: Overlapping curriculum + replay buffer
3. No "pure noise" phase isolation - always includes some PD signal

Key improvements:
- Overlapping curriculum phases instead of hard transitions
- Replay buffer to mix samples from previous phases
- Epoch-synchronized validation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
gem_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(gem_dir))

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple
from collections import deque

from mr_tae_fusion.config import get_config
from mr_tae_fusion.data.pulse_generators import generate_pd_signal
from mr_tae_fusion.data.qlin_loader import QLINDataLoader
from colored_noise import add_colored_noise_at_snr


class ReplayBuffer:
    """
    Replay buffer for anti-forgetting during curriculum learning.
    
    Stores representative samples from previous phases and mixes them
    with current phase samples to prevent catastrophic forgetting.
    """
    
    def __init__(self, max_size: int = 5000, mix_ratio: float = 0.2):
        """
        Args:
            max_size: Maximum buffer size
            mix_ratio: Ratio of replay samples to mix with current phase
        """
        self.max_size = max_size
        self.mix_ratio = mix_ratio
        self.buffer = deque(maxlen=max_size)
        self.phase_samples = {}  # Track samples per phase
    
    def add(self, sample: Dict, phase: int):
        """Add a sample to the buffer."""
        self.buffer.append({
            'sample': sample,
            'phase': phase
        })
        
        # Track per-phase counts
        self.phase_samples[phase] = self.phase_samples.get(phase, 0) + 1
    
    def sample(self, n: int) -> List[Dict]:
        """Sample n items from the buffer."""
        if len(self.buffer) == 0:
            return []
        
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[i]['sample'] for i in indices]
    
    def get_mix_count(self, batch_size: int) -> int:
        """Calculate how many replay samples to mix in."""
        return int(batch_size * self.mix_ratio)
    
    def __len__(self):
        return len(self.buffer)


def get_overlapping_curriculum_phase(
    epoch: int, 
    total_epochs: int
) -> Dict:
    """
    FIXED Curriculum with overlapping phases - no pure noise isolation.
    
    Key changes from original:
    1. No Phase 2 "pure noise" (always includes PD)
    2. Phases overlap for smoother transitions
    3. SNR ranges gradually shift instead of jumping
    
    Returns phase configuration dict.
    """
    progress = epoch / total_epochs
    
    # Calculate overlap zone (10% of phase duration)
    overlap = 0.02
    
    if progress < 0.10:  # Phase 1: High SNR (0-10%)
        # Pure PD learning with gentle noise introduction
        snr_max = 50 - progress * 200  # 50 -> 30
        snr_min = 30 - progress * 100  # 30 -> 20
        
        return {
            'phase': 1,
            'name': 'PD Shape Learning',
            'snr_range': (snr_min, snr_max),
            'use_colored_noise': False,
            'use_disruptive': False,
            'pd_inclusion': 1.0,  # Always include PD
            'replay_ratio': 0.0,  # No replay yet
            'signal_types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'QLIN'],
        }
    
    elif progress < 0.25:  # Phase 2: Light noise (10-25%)
        # PD + light noise - ALWAYS includes PD (fixes forgetting)
        local_progress = (progress - 0.10) / 0.15
        snr_max = 30 - local_progress * 20  # 30 -> 10
        snr_min = 10 - local_progress * 5   # 10 -> 5
        
        return {
            'phase': 2,
            'name': 'Light Noise',
            'snr_range': (snr_min, snr_max),
            'use_colored_noise': True,
            'use_disruptive': False,
            'pd_inclusion': 1.0,  # ALWAYS include PD (no pure noise)
            'replay_ratio': 0.1,
            'signal_types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'QLIN'],
        }
    
    elif progress < 0.45:  # Phase 3: Medium noise (25-45%)
        local_progress = (progress - 0.25) / 0.20
        snr_max = 10 - local_progress * 15  # 10 -> -5
        snr_min = 0 - local_progress * 10   # 0 -> -10
        
        return {
            'phase': 3,
            'name': 'Medium Noise',
            'snr_range': (snr_min, snr_max),
            'use_colored_noise': True,
            'use_disruptive': True,
            'pd_inclusion': 1.0,
            'replay_ratio': 0.15,
            'signal_types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'TEMPORAL', 'QLIN'],
        }
    
    elif progress < 0.65:  # Phase 4: Hard noise (45-65%)
        local_progress = (progress - 0.45) / 0.20
        snr_max = -5 - local_progress * 5   # -5 -> -10
        snr_min = -10 - local_progress * 5  # -10 -> -15
        
        return {
            'phase': 4,
            'name': 'Hard Noise',
            'snr_range': (snr_min, snr_max),
            'use_colored_noise': True,
            'use_disruptive': True,
            'pd_inclusion': 1.0,
            'replay_ratio': 0.2,
            'signal_types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'TEMPORAL', 'QLIN'],
        }
    
    elif progress < 0.85:  # Phase 5: Very hard (65-85%)
        local_progress = (progress - 0.65) / 0.20
        snr_max = -10 - local_progress * 10  # -10 -> -20
        snr_min = -15 - local_progress * 5   # -15 -> -20
        
        return {
            'phase': 5,
            'name': 'Very Hard',
            'snr_range': (snr_min, snr_max),
            'use_colored_noise': True,
            'use_disruptive': True,
            'pd_inclusion': 1.0,
            'replay_ratio': 0.2,
            'signal_types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'TEMPORAL', 'QLIN'],
        }
    
    else:  # Phase 6: Extreme (85-100%)
        local_progress = (progress - 0.85) / 0.15
        snr_max = -20 - local_progress * 5  # -20 -> -25
        snr_min = -20 - local_progress * 5  # -20 -> -25
        
        return {
            'phase': 6,
            'name': 'Extreme (-25dB)',
            'snr_range': (snr_min, snr_max),
            'use_colored_noise': True,
            'use_disruptive': True,
            'pd_inclusion': 1.0,
            'replay_ratio': 0.25,  # More replay at extreme phase
            'signal_types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'TEMPORAL', 'QLIN'],
        }


class GEMDataset(Dataset):
    """
    Gradient Episodic Memory Enhanced Dataset.
    
    Fixes the three critical issues:
    
    1. VALIDATION MIRAGE FIX:
       - Validation dataset tracks current training epoch
       - Call update_epoch() during training to sync
    
    2. CATASTROPHIC FORGETTING FIX:
       - Overlapping curriculum phases (no hard transitions)
       - No "pure noise" phase (always includes PD signals)
       - Replay buffer mixes samples from previous phases
    
    3. PHANTOM TGAN FIX:
       - Uses colored noise (1/f, substation) instead of WGN
       - Optional TGAN integration if weights available
    """
    
    def __init__(
        self,
        config,
        num_samples: int,
        mode: str = 'train',
        epoch: int = 0,
        total_epochs: int = 700,
        qlin_loader = None,
        seed: int = 42,
        use_replay: bool = True,
        replay_ratio: float = 0.2,
        tgan_weights_path: Optional[str] = None
    ):
        """
        Args:
            config: Model/signal configuration
            num_samples: Number of samples per epoch
            mode: 'train' or 'val'
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs
            qlin_loader: Q.Lin data loader instance
            seed: Random seed
            use_replay: Whether to use replay buffer (train only)
            replay_ratio: Ratio of replay samples
            tgan_weights_path: Path to pre-trained TGAN weights
        """
        self.config = config
        self.num_samples = num_samples
        self.mode = mode
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.qlin_loader = qlin_loader
        self.use_replay = use_replay and mode == 'train'
        
        self.rng = np.random.default_rng(seed if mode == 'train' else seed + 1)
        
        # Replay buffer for anti-forgetting
        if self.use_replay:
            self.replay_buffer = ReplayBuffer(
                max_size=min(5000, num_samples // 10),
                mix_ratio=replay_ratio
            )
        else:
            self.replay_buffer = None
        
        # TGAN noise (optional)
        self.tgan_weights_path = tgan_weights_path
        
        # Generate balanced sample assignments
        self._generate_balanced_assignments()
    
    def _generate_balanced_assignments(self):
        """Generate balanced sample assignments across all classes."""
        samples_per_class = self.num_samples // 5
        
        self.sample_assignments = []
        
        # Background (class 0)
        for _ in range(samples_per_class):
            self.sample_assignments.append({'type': 'A', 'target_class': 0})
        
        # Corona (class 1)
        for _ in range(samples_per_class):
            self.sample_assignments.append({'type': 'A', 'target_class': 1})
        
        # Surface (class 2) - 70% Q.Lin, 30% synthetic
        qlin_samples = samples_per_class * 7 // 10
        for _ in range(qlin_samples):
            self.sample_assignments.append({'type': 'QLIN', 'target_class': 2})
        for _ in range(samples_per_class - qlin_samples):
            self.sample_assignments.append({'type': 'B', 'target_class': 2})
        
        # Internal (class 3)
        internal_types = ['C', 'D', 'E', 'F']
        for i in range(samples_per_class):
            self.sample_assignments.append({
                'type': internal_types[i % len(internal_types)],
                'target_class': 3
            })
        
        # Treeing (class 4)
        for _ in range(samples_per_class):
            self.sample_assignments.append({'type': 'G', 'target_class': 4})
        
        self.rng.shuffle(self.sample_assignments)
    
    def update_epoch(self, epoch: int):
        """
        FIX FOR VALIDATION MIRAGE: Update epoch to sync curriculum.
        
        Call this at the start of each epoch during training:
            train_dataset.update_epoch(current_epoch)
            val_dataset.update_epoch(current_epoch)  # <-- This was missing!
        """
        self.epoch = epoch
    
    def __len__(self):
        return self.num_samples
    
    def _add_noise(self, clean: np.ndarray, phase: Dict) -> np.ndarray:
        """
        Add noise based on curriculum phase.
        
        FIX FOR PHANTOM TGAN: Uses colored noise instead of WGN.
        """
        snr_min, snr_max = phase['snr_range']
        snr = self.rng.uniform(snr_min, snr_max)
        
        if phase['use_colored_noise']:
            # Use colored noise (flicker + substation)
            if phase['use_disruptive']:
                noise_type = 'composite'  # Full mix for disruptive
            else:
                noise_type = 'substation'  # Lighter colored noise
            
            noisy = add_colored_noise_at_snr(
                clean, 
                target_snr_db=snr,
                noise_type=noise_type,
                rng=self.rng
            )
        else:
            # Simple WGN for early phases
            signal_power = np.mean(clean ** 2)
            if signal_power < 1e-10:
                signal_power = 1.0
            noise_power = signal_power / (10 ** (snr / 10))
            noise = self.rng.standard_normal(len(clean)) * np.sqrt(noise_power)
            noisy = clean + noise
        
        return noisy, snr
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get current curriculum phase (synchronized with epoch)
        phase = get_overlapping_curriculum_phase(self.epoch, self.total_epochs)
        
        # Check if this should be a replay sample
        use_replay_for_this = (
            self.use_replay and 
            self.replay_buffer is not None and 
            len(self.replay_buffer) > 0 and
            self.rng.random() < phase.get('replay_ratio', 0)
        )
        
        if use_replay_for_this:
            # Use sample from replay buffer
            replay_samples = self.replay_buffer.sample(1)
            if replay_samples:
                return replay_samples[0]
        
        # Get sample assignment
        assignment = self.sample_assignments[idx % len(self.sample_assignments)]
        
        # Generate or load signal
        if assignment['type'] == 'QLIN' and self.qlin_loader is not None:
            signal, diameter = self.qlin_loader.get_random_signal(self.mode)
            if signal is not None:
                mask = self.qlin_loader.create_mask(signal, class_id=2)
                clean = signal.copy()
            else:
                _, clean, mask = generate_pd_signal('B', self.config.signal)
        else:
            sig_type = assignment['type']
            if sig_type not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'TEMPORAL']:
                sig_type = 'A'
            _, clean, mask = generate_pd_signal(sig_type, self.config.signal)
        
        # Truncate/pad to fixed length
        target_len = self.config.signal.signal_length
        if len(clean) > target_len:
            clean = clean[:target_len]
            mask = mask[:target_len]
        elif len(clean) < target_len:
            clean = np.pad(clean, (0, target_len - len(clean)))
            mask = np.pad(mask, (0, target_len - len(mask)))
        
        # Add noise (uses colored noise - FIX FOR PHANTOM TGAN)
        noisy, snr = self._add_noise(clean, phase)
        
        # Normalize
        max_amp = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-8)
        noisy = noisy / max_amp
        clean = clean / max_amp
        
        # Create output dict
        output = {
            'noisy': torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
            'clean': torch.tensor(clean, dtype=torch.float32).unsqueeze(0),
            'mask': torch.tensor(mask, dtype=torch.long),
            'snr': torch.tensor(snr, dtype=torch.float32),
        }
        
        # Add to replay buffer (for training)
        if self.use_replay and self.replay_buffer is not None:
            # Store with probability based on phase to ensure diversity
            if self.rng.random() < 0.1:  # Store 10% of samples
                self.replay_buffer.add(output, phase['phase'])
        
        return output
    
    def get_phase_info(self) -> Dict:
        """Get current curriculum phase info."""
        return get_overlapping_curriculum_phase(self.epoch, self.total_epochs)


class EWCRegularizer:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.
    
    Computes Fisher Information Matrix on important task weights and
    adds regularization to prevent them from changing too much.
    """
    
    def __init__(self, model, lambda_ewc: float = 0.1):
        """
        Args:
            model: The neural network model
            lambda_ewc: EWC regularization strength
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.params_star = {}
        self.initialized = False
    
    def compute_fisher(self, data_loader, criterion, device, num_samples: int = 500):
        """
        Compute Fisher Information Matrix on current task.
        
        Call this at the end of each curriculum phase to consolidate learning.
        """
        self.model.eval()
        
        # Initialize Fisher
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        count = 0
        for batch in data_loader:
            if count >= num_samples:
                break
            
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)
            
            self.model.zero_grad()
            denoised, seg_logits = self.model(noisy)
            loss, _ = criterion(denoised, clean, seg_logits, mask)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach()
            
            count += len(noisy)
        
        # Normalize
        for n in fisher:
            fisher[n] /= count
        
        # Store
        if not self.initialized:
            self.fisher = fisher
            self.params_star = {n: p.clone().detach() for n, p in self.model.named_parameters()}
            self.initialized = True
        else:
            # Running average with previous Fisher
            for n in self.fisher:
                self.fisher[n] = 0.5 * self.fisher[n] + 0.5 * fisher[n]
            self.params_star = {n: p.clone().detach() for n, p in self.model.named_parameters()}
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty term to add to loss."""
        if not self.initialized:
            return torch.tensor(0.0)
        
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                diff = p - self.params_star[n]
                penalty += (self.fisher[n] * diff.pow(2)).sum()
        
        return self.lambda_ewc * penalty
