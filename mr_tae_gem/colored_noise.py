"""
Colored Noise Generators for Realistic PD Signal Training.

Implements:
- 1/f (Flicker) Noise: Power spectral density proportional to 1/f^α
- Substation Interference: Colored noise with power frequency harmonics
- TGAN Integration: Optional loading of pre-trained TGAN for realistic noise

These generators address the "Phantom TGAN" issue where training only used
white Gaussian noise, which doesn't represent real substation environments.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def generate_flicker_noise(
    signal_length: int,
    alpha: float = 1.0,
    amplitude: float = 1.0,
    fs: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate 1/f noise (Flicker noise) using FFT method.
    
    The power spectral density of 1/f noise is S(f) ∝ 1/f^α
    where α controls the "color":
    - α = 0: White noise
    - α = 1: Pink noise (1/f)
    - α = 2: Brown noise (1/f²)
    
    Real substation noise typically has α ≈ 0.8-1.2.
    
    Args:
        signal_length: Length of output signal
        alpha: Power law exponent (default 1.0 for pink noise)
        amplitude: Output amplitude scaling
        fs: Sampling frequency (for proper frequency scaling)
        rng: Random number generator for reproducibility
    
    Returns:
        1/f noise signal
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate white noise in frequency domain
    n_fft = signal_length
    white_spectrum = rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)
    
    # Create 1/f filter
    freqs = np.fft.fftfreq(n_fft, d=1/fs)
    freqs[0] = 1e-10  # Avoid division by zero at DC
    
    # 1/f^(alpha/2) because we're filtering amplitude, not power
    filter_spectrum = 1.0 / (np.abs(freqs) ** (alpha / 2) + 1e-10)
    
    # Apply filter
    colored_spectrum = white_spectrum * filter_spectrum
    
    # Transform back to time domain
    noise = np.real(np.fft.ifft(colored_spectrum))
    
    # Normalize and scale
    noise = noise - np.mean(noise)
    if np.std(noise) > 1e-10:
        noise = noise / np.std(noise)
    noise = noise * amplitude
    
    return noise


def generate_substation_noise(
    signal_length: int,
    fs: float = 10e6,
    powerline_freq: float = 50.0,
    num_harmonics: int = 7,
    harmonic_decay: float = 0.7,
    flicker_alpha: float = 1.0,
    flicker_weight: float = 0.3,
    amplitude: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate realistic substation interference noise.
    
    Combines:
    1. Power frequency harmonics (50/60Hz and harmonics)
    2. 1/f background noise (equipment & environmental)
    3. Random phase variations per harmonic
    
    Args:
        signal_length: Output length
        fs: Sampling frequency
        powerline_freq: Fundamental power frequency (50Hz or 60Hz)
        num_harmonics: Number of harmonics to include
        harmonic_decay: Decay factor for each successive harmonic
        flicker_alpha: Alpha for 1/f noise component
        flicker_weight: Weight of 1/f component vs harmonics
        amplitude: Overall output amplitude
        rng: Random number generator
    
    Returns:
        Substation interference noise
    """
    if rng is None:
        rng = np.random.default_rng()
    
    t = np.arange(signal_length) / fs
    
    # Generate power frequency harmonics with random phases
    harmonic_noise = np.zeros(signal_length)
    current_amp = 1.0
    
    for h in range(1, num_harmonics + 1):
        freq = h * powerline_freq
        phase = rng.uniform(0, 2 * np.pi)
        harmonic_noise += current_amp * np.sin(2 * np.pi * freq * t + phase)
        current_amp *= harmonic_decay
    
    # Normalize harmonics
    if np.std(harmonic_noise) > 1e-10:
        harmonic_noise = harmonic_noise / np.std(harmonic_noise)
    
    # Generate 1/f background
    flicker_component = generate_flicker_noise(
        signal_length, alpha=flicker_alpha, amplitude=1.0, fs=fs, rng=rng
    )
    
    # Combine components
    combined = (1 - flicker_weight) * harmonic_noise + flicker_weight * flicker_component
    
    # Normalize and scale
    if np.std(combined) > 1e-10:
        combined = combined / np.std(combined)
    combined = combined * amplitude
    
    return combined


class FlickerNoise:
    """
    Stateful 1/f noise generator for dataset integration.
    
    Provides consistent noise generation with optional caching
    for reproducibility across training runs.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        amplitude: float = 1.0,
        seed: Optional[int] = None
    ):
        self.alpha = alpha
        self.amplitude = amplitude
        self.rng = np.random.default_rng(seed)
    
    def generate(self, signal_length: int, fs: float = 1.0) -> np.ndarray:
        """Generate a sample of 1/f noise."""
        return generate_flicker_noise(
            signal_length=signal_length,
            alpha=self.alpha,
            amplitude=self.amplitude,
            fs=fs,
            rng=self.rng
        )
    
    def reset(self, seed: Optional[int] = None):
        """Reset the random generator."""
        self.rng = np.random.default_rng(seed)


class SubstationNoise:
    """
    Stateful substation interference generator.
    
    Mimics real-world power equipment noise with harmonics
    and 1/f background.
    """
    
    def __init__(
        self,
        powerline_freq: float = 50.0,
        num_harmonics: int = 7,
        harmonic_decay: float = 0.7,
        flicker_alpha: float = 1.0,
        flicker_weight: float = 0.3,
        amplitude: float = 1.0,
        seed: Optional[int] = None
    ):
        self.powerline_freq = powerline_freq
        self.num_harmonics = num_harmonics
        self.harmonic_decay = harmonic_decay
        self.flicker_alpha = flicker_alpha
        self.flicker_weight = flicker_weight
        self.amplitude = amplitude
        self.rng = np.random.default_rng(seed)
    
    def generate(self, signal_length: int, fs: float = 10e6) -> np.ndarray:
        """Generate a sample of substation noise."""
        return generate_substation_noise(
            signal_length=signal_length,
            fs=fs,
            powerline_freq=self.powerline_freq,
            num_harmonics=self.num_harmonics,
            harmonic_decay=self.harmonic_decay,
            flicker_alpha=self.flicker_alpha,
            flicker_weight=self.flicker_weight,
            amplitude=self.amplitude,
            rng=self.rng
        )
    
    def reset(self, seed: Optional[int] = None):
        """Reset the random generator."""
        self.rng = np.random.default_rng(seed)


class ColoredNoiseGenerator:
    """
    Unified colored noise generator combining multiple noise types.
    
    This fixes the "Phantom TGAN" issue by providing realistic noise
    that matches real substation environments instead of just WGN.
    
    Noise types available:
    - 'flicker': 1/f noise (pink noise)
    - 'brown': 1/f² noise (Brownian noise)
    - 'substation': Power harmonics + 1/f background
    - 'composite': Mix of all types
    - 'tgan': TGAN-generated noise (if weights available)
    """
    
    def __init__(
        self,
        noise_type: str = 'composite',
        tgan_weights_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.noise_type = noise_type
        self.rng = np.random.default_rng(seed)
        self.tgan_generator = None
        
        # Initialize sub-generators
        self.flicker_gen = FlickerNoise(alpha=1.0, seed=seed)
        self.brown_gen = FlickerNoise(alpha=2.0, seed=seed + 1 if seed else None)
        self.substation_gen = SubstationNoise(seed=seed + 2 if seed else None)
        
        # Try to load TGAN if path provided
        if tgan_weights_path and Path(tgan_weights_path).exists():
            self._load_tgan(tgan_weights_path)
    
    def _load_tgan(self, weights_path: str):
        """Attempt to load pre-trained TGAN generator."""
        try:
            import torch
            import sys
            from pathlib import Path
            
            # Add project root to path
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            
            from mr_tae_fusion.models.tgan import TGANNoiseLoader
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.tgan_generator = TGANNoiseLoader(
                weights_path=weights_path,
                device=device
            )
            print(f"✓ TGAN loaded from {weights_path}")
        except Exception as e:
            print(f"⚠ TGAN not available: {e}")
            self.tgan_generator = None
    
    def generate(
        self, 
        signal_length: int, 
        noise_type: Optional[str] = None,
        fs: float = 10e6
    ) -> np.ndarray:
        """
        Generate colored noise of specified type.
        
        Args:
            signal_length: Output length
            noise_type: Override default type ('flicker', 'brown', 'substation', 'composite', 'tgan')
            fs: Sampling frequency
        
        Returns:
            Colored noise signal
        """
        noise_type = noise_type or self.noise_type
        
        if noise_type == 'flicker':
            return self.flicker_gen.generate(signal_length, fs)
        
        elif noise_type == 'brown':
            return self.brown_gen.generate(signal_length, fs)
        
        elif noise_type == 'substation':
            return self.substation_gen.generate(signal_length, fs)
        
        elif noise_type == 'tgan' and self.tgan_generator is not None:
            return self.tgan_generator.get_single_noise()[:signal_length]
        
        elif noise_type == 'composite':
            # Mix multiple noise types for maximum realism
            flicker = self.flicker_gen.generate(signal_length, fs)
            substation = self.substation_gen.generate(signal_length, fs)
            wgn = self.rng.standard_normal(signal_length)
            
            # Weighted combination
            weights = [0.4, 0.4, 0.2]  # flicker, substation, wgn
            combined = weights[0] * flicker + weights[1] * substation + weights[2] * wgn
            
            # Normalize
            if np.std(combined) > 1e-10:
                combined = combined / np.std(combined)
            
            return combined
        
        else:
            # Fallback to flicker
            return self.flicker_gen.generate(signal_length, fs)
    
    def add_to_signal(
        self,
        clean_signal: np.ndarray,
        target_snr_db: float,
        noise_type: Optional[str] = None,
        fs: float = 10e6
    ) -> np.ndarray:
        """
        Add colored noise to a clean signal at specified SNR.
        
        Args:
            clean_signal: Clean input signal
            target_snr_db: Target SNR in dB
            noise_type: Noise type (uses default if None)
            fs: Sampling frequency
        
        Returns:
            Noisy signal
        """
        signal_length = len(clean_signal)
        
        # Generate noise
        noise = self.generate(signal_length, noise_type, fs)
        
        # Calculate signal power
        signal_power = np.mean(clean_signal ** 2)
        if signal_power < 1e-10:
            signal_power = 1.0
        
        # Calculate required noise power
        noise_power = signal_power / (10 ** (target_snr_db / 10))
        
        # Scale noise to achieve target SNR
        current_noise_power = np.mean(noise ** 2)
        if current_noise_power > 1e-10:
            scale_factor = np.sqrt(noise_power / current_noise_power)
            noise = noise * scale_factor
        
        return clean_signal + noise


def add_colored_noise_at_snr(
    clean_signal: np.ndarray,
    target_snr_db: float,
    noise_type: str = 'composite',
    fs: float = 10e6,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Convenience function to add colored noise at specified SNR.
    
    This is a drop-in replacement for add_simple_noise/add_disruptive_noise
    in the original training script.
    
    Args:
        clean_signal: Clean input signal
        target_snr_db: Target SNR in dB
        noise_type: Type of colored noise
        fs: Sampling frequency
        rng: Random number generator
    
    Returns:
        Noisy signal
    """
    if rng is None:
        rng = np.random.default_rng()
    
    signal_length = len(clean_signal)
    
    # Generate base colored noise
    if noise_type == 'flicker':
        noise = generate_flicker_noise(signal_length, alpha=1.0, fs=fs, rng=rng)
    elif noise_type == 'substation':
        noise = generate_substation_noise(signal_length, fs=fs, rng=rng)
    else:
        # Composite: mix flicker + substation + light WGN
        flicker = generate_flicker_noise(signal_length, alpha=1.0, fs=fs, rng=rng)
        substation = generate_substation_noise(signal_length, fs=fs, rng=rng)
        wgn = rng.standard_normal(signal_length)
        noise = 0.4 * flicker + 0.4 * substation + 0.2 * wgn
        
        # Normalize
        if np.std(noise) > 1e-10:
            noise = noise / np.std(noise)
    
    # Scale to target SNR
    signal_power = np.mean(clean_signal ** 2)
    if signal_power < 1e-10:
        signal_power = 1.0
    
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    current_noise_power = np.mean(noise ** 2)
    
    if current_noise_power > 1e-10:
        scale_factor = np.sqrt(noise_power / current_noise_power)
        noise = noise * scale_factor
    
    # Add impulsive component at low SNR
    if target_snr_db < 0:
        impulse_density = 0.02 * abs(target_snr_db) / 25  # More impulses at lower SNR
        impulse_mask = rng.random(signal_length) < impulse_density
        impulse_amp = np.sqrt(noise_power) * 10
        noise[impulse_mask] += impulse_amp * rng.choice([-1, 1], size=np.sum(impulse_mask))
    
    return clean_signal + noise
