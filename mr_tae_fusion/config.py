"""
Configuration settings for MR-TAE-Fusion framework.

Centralizes all hyperparameters, model architecture settings, and training configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class SignalConfig:
    """Signal generation and processing parameters."""
    
    # Sampling parameters (matching MATLAB code)
    sample_rate: float = 1000e6  # 1 GHz sampling
    duration: float = 2e-6  # 2 μs duration
    signal_length: int = 2001  # Number of samples
    
    # Pulse parameters
    dep_tau1_range: tuple = (20e-9, 50e-9)  # Decay time constant
    dep_tau2_range: tuple = (1e-9, 10e-9)   # Rise time constant
    dop_tau_range: tuple = (10e-9, 50e-9)   # Damping coefficient
    dop_fc_range: tuple = (1e6, 300e6)      # Carrier frequency range
    amplitude_range: tuple = (2.0, 10.0)    # Pulse amplitude
    
    # Class mapping: Types A-G → 5 physics-based classes
    # 0: Background, 1: Corona, 2: Surface, 3: Internal, 4: Treeing
    type_to_class: dict = field(default_factory=lambda: {
        'A': 1,  # Sparse PD → Corona
        'B': 2,  # Spike-dense → Surface  
        'C': 3,  # 10mm → Internal
        'D': 3,  # 18mm → Internal
        'E': 3,  # 20mm → Internal
        'F': 3,  # 25mm → Internal
        'G': 4,  # Treeing → Treeing
        'MIXED': 0,  # Mixed signals (multi-class)
    })
    num_classes: int = 5  # Background + 4 defect types (Corona, Surface, Internal, Treeing)


@dataclass
class NoiseConfig:
    """Noise generation parameters."""
    
    # Bernoulli-Gaussian impulsive noise
    impulse_probability: float = 0.01  # Sparsity parameter p
    impulse_variance_factor: float = 10.0  # σ_imp² >> σ_background²
    
    # White Gaussian noise
    wgn_amplitude: float = 0.08
    
    # Powerline interference
    powerline_freq: float = 50e6
    powerline_amplitude: float = 0.025
    harmonic_amplitude: float = 0.015
    
    # Narrowband interference
    narrowband_freq: float = 80e6
    narrowband_amplitude: float = 0.03
    
    # SNR targets for curriculum learning
    snr_range_phase1: tuple = (0, 10)    # Warm-up: +10 to 0 dB
    snr_range_phase2: tuple = (-10, 0)   # Robustness: 0 to -10 dB
    snr_range_phase3: tuple = (-20, -5)  # Reality: -5 to -20 dB


@dataclass
class WaveletConfig:
    """Wavelet transform parameters."""
    
    wavelet_type: str = 'db4'  # Daubechies 4 - best for asymmetric PD pulses
    decomposition_levels: int = 3  # MWCNN encoder depth
    
    # Filter lengths for db4
    filter_length: int = 8  # db4 has 8 coefficients


@dataclass 
class ModelConfig:
    """Model architecture parameters."""
    
    # Input/Output
    in_channels: int = 1
    signal_length: int = 2001
    num_classes: int = 5  # Background + Corona + Surface + Internal + Treeing
    
    # MWCNN Encoder
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    encoder_kernel_size: int = 3
    
    # Bottleneck
    bottleneck_channels: int = 256
    
    # BiGRU
    gru_hidden_size: int = 128
    gru_num_layers: int = 2
    gru_bidirectional: bool = True
    gru_dropout: float = 0.1
    
    # Swin Transformer 1D
    swin_embed_dim: int = 256
    swin_num_heads: int = 8
    swin_window_size: int = 32
    swin_depth: int = 2
    swin_mlp_ratio: float = 4.0
    swin_dropout: float = 0.1
    swin_attention_dropout: float = 0.1
    
    # Attention Gates
    attention_intermediate_channels: int = 64
    
    # Decoder
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    
    # Fusion Head
    fusion_channels: int = 64


@dataclass
class TrainingConfig:
    """Training parameters."""
    
    # Curriculum learning phases
    phase1_epochs: int = 20   # Warm-up
    phase2_epochs: int = 30   # Robustness  
    phase3_epochs: int = 50   # Reality
    total_epochs: int = 100
    
    # Batch size and learning rate
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    lr_schedule: str = 'cosine'  # 'cosine' or 'step'
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    
    # Multi-task loss weights (initial values, will be learned)
    initial_sigma_recon: float = 1.0
    initial_sigma_seg: float = 1.0
    
    # Dice loss weight in segmentation loss
    dice_weight: float = 1.0
    
    # Validation
    val_frequency: int = 5  # Validate every N epochs
    
    # Early stopping
    patience: int = 20
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Data paths and loading parameters."""
    
    # Paths to real data
    mat_data_dir: str = r"D:\New folder (2)\Data\OneDrive_2025-12-13\Datset from Q.lin"
    noise_data_path: str = r"D:\New folder (2)\Data\noisy_minus10dB_18mm.mat"
    
    # Dataset split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Synthetic data generation
    samples_per_type: int = 500  # Per signal type A-F
    
    # Data augmentation
    use_tgan: bool = True
    tgan_samples: int = 1000


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    signal: SignalConfig = field(default_factory=SignalConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        """Ensure consistency across configs."""
        self.model.signal_length = self.signal.signal_length
        self.model.num_classes = self.signal.num_classes


# Global config instance
def get_config() -> Config:
    """Get default configuration."""
    return Config()
