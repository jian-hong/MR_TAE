"""
Configuration settings for MR-TAE-Fusion framework.

Centralizes all hyperparameters, model architecture settings, and training configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class SignalConfig:
    """Signal generation and processing parameters.

    Acoustic Emission (AE) sensor band: 20–500 kHz.
    Sampling at 2 MHz satisfies Nyquist for the 500 kHz upper bound.
    All time constants in seconds, frequencies in Hz.
    """

    # Sampling — AE piezoelectric acquisition
    sample_rate: float = 2e6       # 2 MHz (Nyquist for 500 kHz AE band)
    duration: float = 1.024e-3     # ~1 ms observation window
    signal_length: int = 2048      # samples per window

    # DEP (Damped Exponential Pulse) — μs-scale for AE propagation
    dep_tau1_range: tuple = (10e-6, 80e-6)   # Decay: 10–80 μs
    dep_tau2_range: tuple = (1e-6, 8e-6)     # Rise:  1–8 μs

    # DOP (Damped Oscillatory Pulse) — kHz carrier within AE band
    dop_tau_range: tuple = (5e-6, 60e-6)     # Damping: 5–60 μs
    dop_fc_range: tuple = (30e3, 400e3)      # Carrier: 30–400 kHz

    # Pulse amplitude (normalised to unit peak after generation)
    amplitude_range: tuple = (0.3, 1.0)
    
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
    """AE-band noise generation parameters.

    Every component maps to a real acquisition source:
      WGN            → DAQ ADC quantisation + preamp thermal noise
      Powerline hum  → ground-loop EMI in cable / DAQ chassis
      Magnetostriction → tank-wall vibration at 2× mains (sensor mount)
      Mechanical imp. → pump transients, valve clicks, loose parts
      Machinery tones → bearing harmonics, fan BPF, oil-flow turbulence

    Amplitudes normalised relative to unit-peak PD pulse.
    Frequencies in Hz, time constants in seconds.
    """

    # ── Broadband electronics (DAQ + preamp thermal) ──
    wgn_amplitude: float = 0.04          # σ ∈ [0.02, 0.06]

    # ── Powerline hum (cable-shield / ground-loop coupling) ──
    powerline_freq: float = 50.0         # 50 Hz (EU/Asia) or 60 Hz (Americas)
    powerline_amplitude: float = 0.015   # ∈ [0.008, 0.025]
    harmonic_amplitude: float = 0.008    # per-harmonic decay start

    # ── Magnetostriction (tank-wall 2×mains vibration) ──
    magnetostriction_amplitude: float = 0.012  # ∈ [0.005, 0.020]

    # ── Narrowband machinery tones (bearings, fans, turbulence) ──
    narrowband_freq: float = 85e3        # primary tone centre
    narrowband_amplitude: float = 0.04   # max per tone ∈ [0.02, 0.06]
    num_machinery_tones: int = 2         # ∈ [1, 3]
    machinery_freq_range: tuple = (25e3, 200e3)   # Hz

    # ── Mechanical impulse transients (pump, valve, rain, rattles) ──
    impulse_probability: float = 0.005   # legacy compat — see mechanical model
    impulse_variance_factor: float = 8.0
    mechanical_impulse_fc_range: tuple = (30e3, 80e3)   # Hz
    mechanical_impulse_tau_range: tuple = (3e-6, 15e-6)  # seconds

    # ── SNR targets for curriculum learning ──
    snr_range_phase1: tuple = (5, 15)    # Easy (PD clearly visible)
    snr_range_phase2: tuple = (-5, 5)    # Medium (PD partially masked)
    snr_range_phase3: tuple = (-20, -5)  # Hard (PD buried in noise)


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
    signal_length: int = 2048
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
