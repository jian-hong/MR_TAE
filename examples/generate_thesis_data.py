#!/usr/bin/env python
"""
generate_thesis_data.py - Generate synthetic PD signals for thesis comparison.

Creates Clean and Noisy signals at various SNR levels, saved as .mat files
for compatibility with MATLAB comparison scripts.

Usage:
    python generate_thesis_data.py --output_dir Thesis_Data --num_signals 200
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Try to import scipy for .mat file saving
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Will save as .npy files instead of .mat")

from mr_tae_fusion.config import get_config
from mr_tae_fusion.data.pulse_generators import generate_dep, generate_dop
from mr_tae_fusion.data.noise_generators import (
    generate_composite_noise, 
    generate_wgn,
    generate_impulsive_noise,
    add_noise_at_snr
)


# --- Configuration ---
DEFAULT_FS = 100e6           # 100 MHz sampling
DEFAULT_DURATION = 20e-6     # 20 microseconds
DEFAULT_NUM_SIGNALS = 200
DEFAULT_SNR_LEVELS = [-5, -10, -15, -20]


def generate_damped_oscillatory_pulse(t, A=1.0, tau1=1e-6, tau2=0.1e-6, fc=20e6):
    """
    Standard Damped Oscillatory Pulse (DOP) formula from IEEE papers.
    
    This is the canonical form used in PD literature.
    """
    pulse = A * (np.exp(-t / tau1) - np.exp(-t / tau2)) * np.sin(2 * np.pi * fc * t)
    return pulse


def generate_random_pd_signal(t, signal_type='random'):
    """
    Generate a random PD-like signal with randomized parameters.
    
    Simulates different defect types (Corona, Surface, Internal) with
    varying damping constants and oscillation frequencies.
    """
    rng = np.random.default_rng()
    
    if signal_type == 'corona':
        # Corona: high frequency, fast damping
        tau1 = rng.uniform(0.3e-6, 0.8e-6)
        tau2 = rng.uniform(0.02e-6, 0.05e-6)
        fc = rng.uniform(20e6, 40e6)
    elif signal_type == 'surface':
        # Surface: medium frequency
        tau1 = rng.uniform(0.5e-6, 1.2e-6)
        tau2 = rng.uniform(0.05e-6, 0.1e-6)
        fc = rng.uniform(15e6, 30e6)
    elif signal_type == 'internal':
        # Internal: lower frequency, slower damping
        tau1 = rng.uniform(1.0e-6, 2.0e-6)
        tau2 = rng.uniform(0.1e-6, 0.2e-6)
        fc = rng.uniform(10e6, 20e6)
    else:
        # Random: uniform sampling
        tau1 = rng.uniform(0.5e-6, 1.5e-6)
        tau2 = rng.uniform(0.02e-6, 0.1e-6)
        fc = rng.uniform(10e6, 30e6)
    
    signal = generate_damped_oscillatory_pulse(t, 1.0, tau1, tau2, fc)
    
    # Normalize to unit amplitude
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    return signal, {'tau1': tau1, 'tau2': tau2, 'fc': fc}


def generate_thesis_data(
    output_dir: str,
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    num_signals: int = DEFAULT_NUM_SIGNALS,
    snr_levels: list = DEFAULT_SNR_LEVELS,
    use_tgan: bool = False,
    tgan_path: str = None
):
    """
    Generate complete dataset for thesis comparison.
    
    Creates:
    - Clean_Signals.mat: Clean PD reference signals
    - Noisy_SNR_XX.mat: Noisy signals at each SNR level
    
    Args:
        output_dir: Output directory for data files
        fs: Sampling frequency (Hz)
        duration: Signal duration (seconds)
        num_signals: Number of signals to generate per SNR
        snr_levels: List of target SNR levels (dB)
        use_tgan: Whether to use TGAN for noise generation
        tgan_path: Path to trained TGAN generator weights
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)
    
    print(f"Generating thesis data...")
    print(f"  Output directory: {output_dir}")
    print(f"  Sampling frequency: {fs/1e6} MHz")
    print(f"  Duration: {duration*1e6} µs")
    print(f"  Samples per signal: {n_samples}")
    print(f"  Number of signals: {num_signals}")
    print(f"  SNR levels: {snr_levels} dB")
    
    # Load TGAN if specified
    tgan_loader = None
    if use_tgan and tgan_path:
        try:
            from mr_tae_fusion.models.tgan import TGANNoiseLoader
            tgan_loader = TGANNoiseLoader(tgan_path, seq_len=n_samples)
            print(f"  Using TGAN noise from: {tgan_path}")
        except Exception as e:
            print(f"  Warning: Could not load TGAN ({e}), using synthetic noise")
    
    # --- 1. Generate Clean Signals ---
    print("\n[1/2] Generating clean signals...")
    
    clean_signals = []
    signal_types = ['corona', 'surface', 'internal', 'random']
    
    for i in range(num_signals):
        # Cycle through signal types
        sig_type = signal_types[i % len(signal_types)]
        signal, params = generate_random_pd_signal(t, sig_type)
        clean_signals.append(signal)
        
        if (i + 1) % 50 == 0:
            print(f"    Generated {i + 1}/{num_signals} clean signals")
    
    clean_signals = np.array(clean_signals, dtype=np.float32)
    
    # Save clean data
    if HAS_SCIPY:
        sio.savemat(
            str(output_dir / "Clean_Signals.mat"),
            {"clean_signals": clean_signals, "t": t, "fs": fs}
        )
        print(f"  Saved: Clean_Signals.mat")
    else:
        np.savez(
            str(output_dir / "Clean_Signals.npz"),
            clean_signals=clean_signals, t=t, fs=fs
        )
        print(f"  Saved: Clean_Signals.npz")
    
    # --- 2. Generate Noisy Signals at Each SNR ---
    print("\n[2/2] Generating noisy signals...")
    
    config = get_config()
    
    for snr in snr_levels:
        print(f"\n  Processing SNR = {snr} dB...")
        noisy_batch = []
        
        for i, clean in enumerate(clean_signals):
            # Generate composite noise
            if tgan_loader is not None:
                # TGAN-based realistic noise
                noise = tgan_loader.get_single_noise()
            else:
                # Synthetic composite noise
                noise = generate_composite_noise(t, config.noise)
            
            # Scale noise to target SNR
            noisy, actual_snr = add_noise_at_snr(clean, noise, snr)
            noisy_batch.append(noisy.astype(np.float32))
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{num_signals} signals")
        
        noisy_batch = np.array(noisy_batch, dtype=np.float32)
        
        # Save noisy data
        filename = f"Noisy_SNR_{abs(snr)}"
        if HAS_SCIPY:
            sio.savemat(
                str(output_dir / f"{filename}.mat"),
                {"noisy_signals": noisy_batch, "snr": snr}
            )
            print(f"  Saved: {filename}.mat")
        else:
            np.savez(
                str(output_dir / f"{filename}.npz"),
                noisy_signals=noisy_batch, snr=snr
            )
            print(f"  Saved: {filename}.npz")
    
    print("\n" + "="*50)
    print("Data generation complete!")
    print(f"Output: {output_dir}")
    print("="*50)
    
    return clean_signals


def main():
    parser = argparse.ArgumentParser(description="Generate thesis comparison data")
    parser.add_argument('--output_dir', type=str, default='Thesis_Data',
                        help='Output directory')
    parser.add_argument('--fs', type=float, default=100e6,
                        help='Sampling frequency (Hz)')
    parser.add_argument('--duration', type=float, default=20e-6,
                        help='Signal duration (seconds)')
    parser.add_argument('--num_signals', type=int, default=200,
                        help='Number of signals per SNR')
    parser.add_argument('--snr_levels', type=int, nargs='+', 
                        default=[-5, -10, -15, -20],
                        help='SNR levels to generate')
    parser.add_argument('--use_tgan', action='store_true',
                        help='Use TGAN for noise generation')
    parser.add_argument('--tgan_path', type=str, default=None,
                        help='Path to trained TGAN generator')
    
    args = parser.parse_args()
    
    generate_thesis_data(
        output_dir=args.output_dir,
        fs=args.fs,
        duration=args.duration,
        num_signals=args.num_signals,
        snr_levels=args.snr_levels,
        use_tgan=args.use_tgan,
        tgan_path=args.tgan_path
    )


if __name__ == '__main__':
    main()
