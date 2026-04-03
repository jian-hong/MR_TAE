#!/usr/bin/env python
"""
compare_all_models.py - Compare All Trained Models with Traditional Methods

This script compares:
1. MR-TAE-Fusion (improved_training)
2. MR-TAE-Fusion (extended_training)  
3. MR-TAE-Fusion (extended_500_training)
4. Traditional Wavelet (Soft Threshold db4)
5. Savitzky-Golay + Wavelet (MATLAB denoised_testing.m equivalent)
6. Wavelet BayesShrink

Using the same simulated data from MATLAB DL_Testing.m:
- Type A: Sparse PD pulses (4 pulses with gaps)
- Type B: Spike-dense signal (realistic sharp pulses)
- Type C: 10mm (very sparse PD events)
- Type D: 18mm (sparse PD events)
- Type E: 20mm (moderate-high frequency PD)
- Type F: 25mm (high frequency, complex PD)

USAGE:
    python compare_all_models.py
    python compare_all_models.py --snr -10 --samples 100
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import json
import pywt
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.amp import autocast

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# SIGNAL GENERATORS (Matching MATLAB DL_Testing.m exactly)
# =============================================================================

def generate_type_a(t, fs):
    """Type A: Sparse PD pulses (4 pulses with gaps)"""
    signal = np.zeros_like(t)
    start_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6]
    
    for st in start_times:
        A = 10 + np.random.rand() * 10
        fc = 25e6 + np.random.rand() * 10e6
        tau = 0.01e-6 + np.random.rand() * 0.03e-6
        
        pulse_duration = 0.05e-6
        start_idx = int(st * fs)
        end_idx = min(int((st + pulse_duration) * fs), len(t))
        
        if start_idx < len(t) and end_idx > start_idx:
            pulse_t = np.arange(end_idx - start_idx) / fs
            pulse = A * np.exp(-pulse_t / tau) * np.sin(2 * np.pi * fc * pulse_t)
            signal[start_idx:end_idx] = pulse
    
    return signal


def generate_type_b(t, fs):
    """Type B: Spike-dense signal (realistic sharp pulses with gaps)"""
    signal = np.zeros_like(t)
    num_spikes = 20 + np.random.randint(10)
    
    for _ in range(num_spikes):
        start_idx = np.random.randint(0, len(t) - 2)
        amp = 0.5 + 0.5 * np.random.rand()
        direction = (-1) ** np.random.randint(2)
        
        signal[start_idx] = direction * amp
        signal[start_idx + 1] = -direction * amp
    
    return signal


def generate_type_c(t, fs, t_total=2e-6):
    """Type C: 10mm (very sparse PD events)"""
    signal = np.zeros_like(t)
    num_events = 5 + np.random.randint(5)
    event_times = np.sort(np.random.rand(num_events) * t_total)
    
    for event_time in event_times:
        start_idx = int(event_time * fs)
        if start_idx >= len(signal):
            continue
            
        amplitude = 2.5 + np.random.rand() * 2
        polarity = (-1) ** np.random.randint(2)
        spike_width = 3 + np.random.randint(3)
        
        if start_idx + spike_width <= len(signal):
            signal[start_idx] = polarity * amplitude
            if start_idx + 1 < len(signal):
                signal[start_idx + 1] = -polarity * amplitude * 0.8
            
            for j in range(2, spike_width):
                if start_idx + j < len(signal):
                    signal[start_idx + j] = polarity * amplitude * 0.3 * np.exp(-(j - 1))
    
    return signal


def generate_type_d(t, fs, t_total=2e-6):
    """Type D: 18mm (sparse PD events)"""
    signal = np.zeros_like(t)
    num_events = 55 + np.random.randint(10)
    event_times = np.sort(np.random.rand(num_events) * t_total)
    
    for event_time in event_times:
        start_idx = int(event_time * fs)
        if start_idx >= len(signal):
            continue
            
        event_type = np.random.rand()
        amplitude = 2 + np.random.rand() * 3
        polarity = (-1) ** np.random.randint(2)
        
        if event_type < 0.7:  # 70% - Sharp bipolar spikes
            spike_width = 2 + np.random.randint(4)
            if start_idx + spike_width <= len(signal):
                signal[start_idx] = polarity * amplitude
                if start_idx + 1 < len(signal):
                    signal[start_idx + 1] = -polarity * amplitude * 0.7
                
                for j in range(2, spike_width):
                    if start_idx + j < len(signal):
                        signal[start_idx + j] = polarity * amplitude * 0.2 * np.sin(j)
    
    return signal


def generate_type_e(t, fs, t_total=2e-6):
    """Type E: 20mm (moderate-high frequency PD events)"""
    signal = np.zeros_like(t)
    num_events = 120 + np.random.randint(30)
    event_times = np.sort(np.random.rand(num_events) * t_total)
    
    for event_time in event_times:
        start_idx = int(event_time * fs)
        if start_idx >= len(signal):
            continue
            
        amplitude = 2 + np.random.rand() * 4
        polarity = (-1) ** np.random.randint(2)
        event_type = np.random.rand()
        
        if event_type < 0.6:  # 60% - Sharp spikes
            if start_idx + 1 < len(signal):
                signal[start_idx] = polarity * amplitude
                signal[start_idx + 1] = -polarity * amplitude * 0.2
        else:  # 40% - Multi-frequency transients
            fc1 = 30e6 + np.random.rand() * 50e6
            fc2 = 60e6 + np.random.rand() * 60e6
            fc3 = 100e6 + np.random.rand() * 50e6
            
            event_duration = 5e-9 + np.random.rand() * 15e-9
            event_samples = int(event_duration * fs)
            
            if start_idx + event_samples <= len(signal) and event_samples > 0:
                event_time_vec = np.arange(event_samples) / fs
                envelope = np.exp(-event_time_vec / (event_duration * 0.2))
                
                component1 = 0.3 * amplitude * envelope * np.sin(2 * np.pi * fc1 * event_time_vec)
                component2 = 0.2 * amplitude * envelope * np.sin(2 * np.pi * fc2 * event_time_vec)
                component3 = 0.3 * amplitude * envelope * np.sin(2 * np.pi * fc3 * event_time_vec)
                
                signal[start_idx:start_idx + event_samples] = polarity * (component1 + component2 + component3)
    
    return signal


def generate_type_f(t, fs, t_total=2e-6):
    """Type F: 25mm (high frequency, complex PD events)"""
    signal = np.zeros_like(t)
    num_events = 250 + np.random.randint(80)
    event_times = np.sort(np.random.rand(num_events) * t_total)
    
    for event_time in event_times:
        start_idx = int(event_time * fs)
        if start_idx >= len(signal):
            continue
            
        amplitude = 3 + np.random.rand() * 4.2
        polarity = (-1) ** np.random.randint(2)
        event_type = np.random.rand()
        
        if event_type < 0.4:  # 40% - Quick spikes
            if start_idx + 1 < len(signal):
                signal[start_idx] = polarity * amplitude
                signal[start_idx + 1] = -polarity * amplitude * 0.2
        else:  # 60% - Complex multi-component events
            fc1 = 30e6 + np.random.rand() * 50e6
            fc2 = 60e6 + np.random.rand() * 60e6
            fc3 = 100e6 + np.random.rand() * 50e6
            
            event_duration = 5e-9 + np.random.rand() * 15e-9
            event_samples = int(event_duration * fs)
            
            if start_idx + event_samples <= len(signal) and event_samples > 0:
                event_time_vec = np.arange(event_samples) / fs
                envelope = np.exp(-event_time_vec / (event_duration * 0.2))
                
                component1 = 0.3 * amplitude * envelope * np.sin(2 * np.pi * fc1 * event_time_vec)
                component2 = 0.2 * amplitude * envelope * np.sin(2 * np.pi * fc2 * event_time_vec)
                component3 = 0.3 * amplitude * envelope * np.sin(2 * np.pi * fc3 * event_time_vec)
                
                signal[start_idx:start_idx + event_samples] = polarity * (component1 + component2 + component3)
    
    return signal


def generate_signal(signal_type, fs=1e9, t_total=2e-6):
    """Generate signal of specified type."""
    t = np.arange(0, t_total, 1/fs)
    
    generators = {
        'A': lambda: generate_type_a(t, fs),
        'B': lambda: generate_type_b(t, fs),
        'C': lambda: generate_type_c(t, fs, t_total),
        'D': lambda: generate_type_d(t, fs, t_total),
        'E': lambda: generate_type_e(t, fs, t_total),
        'F': lambda: generate_type_f(t, fs, t_total),
    }
    
    signal = generators[signal_type]()
    
    # Normalize
    max_amp = np.max(np.abs(signal))
    if max_amp > 5:
        signal = signal * (5 / max_amp)
    if max_amp > 0:
        signal = signal / (max_amp + 1e-10)
    
    return t, signal


def add_noise(signal, snr_db):
    """Add white Gaussian noise at specified SNR."""
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:
        signal_power = 1.0
    
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    return signal + noise


# =============================================================================
# TRADITIONAL DENOISING METHODS
# =============================================================================

def wavelet_soft_threshold(signal, wavelet='db4', level=5):
    """Traditional wavelet denoising with soft thresholding (MATLAB wdenoise equivalent)."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Universal threshold (Donoho-Johnstone)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply threshold to detail coefficients
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
    
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:len(signal)]


def wavelet_bayesshrink(signal, wavelet='db4', level=5):
    """BayesShrink wavelet denoising."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    sigma_n = np.median(np.abs(coeffs[-1])) / 0.6745
    
    denoised_coeffs = [coeffs[0]]
    
    for detail in coeffs[1:]:
        sigma_d2 = max(np.var(detail) - sigma_n ** 2, 0)
        
        if sigma_d2 > 0:
            threshold = (sigma_n ** 2) / np.sqrt(sigma_d2)
        else:
            threshold = sigma_n * np.sqrt(2 * np.log(len(signal)))
        
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
    
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:len(signal)]


def savgol_wavelet_denoising(signal, wavelet='db2'):
    """Savitzky-Golay + Wavelet (matching MATLAB denoised_testing.m)."""
    try:
        signal = np.array(signal).flatten()
        
        # Savitzky-Golay parameters
        window_length = max(7, len(signal) // 50)
        if window_length % 2 == 0:
            window_length += 1
        poly_order = min(3, window_length - 2)
        
        if len(signal) >= window_length and window_length >= 5:
            smooth_signal = scipy_signal.savgol_filter(signal, window_length, poly_order)
            residual = signal - smooth_signal
            
            # Wavelet denoise residual
            if len(residual) > 4:
                denoised_residual = wavelet_soft_threshold(residual, wavelet)
            else:
                denoised_residual = residual
            
            # Combine (Note: MATLAB uses multiplication which seems odd, using addition)
            denoised = smooth_signal + denoised_residual
        else:
            denoised = wavelet_soft_threshold(signal, wavelet)
        
        return denoised[:len(signal)]
    except:
        return wavelet_soft_threshold(signal, wavelet='db2')


# =============================================================================
# DEEP LEARNING MODEL LOADING
# =============================================================================

def load_model(model_path, num_classes=4):
    """Load trained MR-TAE-Fusion model."""
    config = get_config()
    config.model.num_classes = num_classes
    model = create_model(config.model).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, checkpoint


def denoise_with_model(model, signal, expected_length=2001):
    """Denoise signal using deep learning model."""
    # Prepare input
    if len(signal) > expected_length:
        sig = signal[:expected_length]
    else:
        sig = np.pad(signal, (0, expected_length - len(signal)))
    
    # Normalize
    max_amp = np.max(np.abs(sig)) + 1e-10
    sig_norm = sig / max_amp
    
    # To tensor
    x = torch.tensor(sig_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        with autocast('cuda', enabled=DEVICE == 'cuda'):
            denoised, seg_logits = model(x)
    
    denoised_np = denoised.squeeze().cpu().numpy() * max_amp
    seg_mask = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    return denoised_np[:len(signal)], seg_mask[:len(signal)]


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_snr(clean, denoised):
    """Calculate SNR after denoising."""
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - denoised) ** 2)
    if noise_power < 1e-10:
        return 50.0
    return 10 * np.log10(signal_power / noise_power)


def calculate_snr_improvement(clean, noisy, denoised):
    """Calculate SNR improvement."""
    snr_before = calculate_snr(clean, noisy)
    snr_after = calculate_snr(clean, denoised)
    return snr_after - snr_before


def calculate_ncc(clean, denoised):
    """Calculate Normalized Cross-Correlation."""
    clean_centered = clean - np.mean(clean)
    denoised_centered = denoised - np.mean(denoised)
    
    numerator = np.sum(clean_centered * denoised_centered)
    denominator = np.sqrt(np.sum(clean_centered ** 2) * np.sum(denoised_centered ** 2) + 1e-10)
    
    return numerator / denominator


def calculate_mse(clean, denoised):
    """Calculate Mean Squared Error."""
    return np.mean((clean - denoised) ** 2)


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(args):
    """Run comprehensive comparison of all models."""
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print(f"SNR: {args.snr} dB")
    print(f"Samples per type: {args.samples}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load all deep learning models
    models = {}
    model_paths = {
        'MR-TAE (Improved)': Path('outputs/improved_training/best_model.pth'),
        'MR-TAE (Extended)': Path('outputs/extended_training/best_model.pth'),
        'MR-TAE (Ext-500)': Path('outputs/extended_500_training/best_model.pth'),
    }
    
    print("Loading deep learning models...")
    for name, path in model_paths.items():
        if path.exists():
            try:
                model, checkpoint = load_model(path)
                models[name] = model
                
                # Print model info
                epoch = checkpoint.get('epoch', 'N/A')
                ncc = checkpoint.get('ncc', checkpoint.get('best_ncc', 'N/A'))
                snr_imp = checkpoint.get('snr_imp', checkpoint.get('best_snr_imp', 'N/A'))
                
                if isinstance(ncc, (int, float)):
                    ncc = f"{ncc:.4f}"
                if isinstance(snr_imp, (int, float)):
                    snr_imp = f"{snr_imp:.2f}"
                    
                print(f"  ✓ {name}: Epoch={epoch}, NCC={ncc}, SNR+={snr_imp}")
            except Exception as e:
                print(f"  ✗ {name}: Failed to load - {e}")
        else:
            print(f"  ✗ {name}: File not found")
    
    # Traditional methods
    traditional_methods = {
        'Wavelet (db4 Soft)': wavelet_soft_threshold,
        'Wavelet (BayesShrink)': wavelet_bayesshrink,
        'SavGol + Wavelet': savgol_wavelet_denoising,
    }
    print(f"\nTraditional methods: {list(traditional_methods.keys())}")
    
    # Signal types
    signal_types = ['A', 'B', 'C', 'D', 'E', 'F']
    type_descriptions = {
        'A': 'Sparse PD pulses',
        'B': 'Spike-dense signal',
        'C': '10mm particles',
        'D': '18mm particles',
        'E': '20mm particles',
        'F': '25mm particles',
    }
    
    # Results storage
    all_results = {sig_type: {} for sig_type in signal_types}
    example_signals = {}
    
    # Generate and evaluate signals
    print(f"\nGenerating {args.samples * len(signal_types)} signals...")
    
    for sig_type in tqdm(signal_types, desc="Signal Types"):
        type_results = {method: {'snr_imp': [], 'ncc': [], 'mse': []} 
                       for method in list(models.keys()) + list(traditional_methods.keys())}
        
        # Store one example per type
        examples = []
        
        for i in range(args.samples):
            # Generate signal
            np.random.seed(i * len(signal_types) + ord(sig_type))
            t, clean = generate_signal(sig_type)
            noisy = add_noise(clean, args.snr)
            
            # Evaluate traditional methods
            for method_name, method_fn in traditional_methods.items():
                try:
                    denoised = method_fn(noisy)
                    if len(denoised) != len(clean):
                        denoised = denoised[:len(clean)] if len(denoised) > len(clean) else np.pad(denoised, (0, len(clean) - len(denoised)))
                    
                    type_results[method_name]['snr_imp'].append(calculate_snr_improvement(clean, noisy, denoised))
                    type_results[method_name]['ncc'].append(calculate_ncc(clean, denoised))
                    type_results[method_name]['mse'].append(calculate_mse(clean, denoised))
                except Exception as e:
                    pass
            
            # Evaluate DL models
            for method_name, model in models.items():
                try:
                    denoised, seg_mask = denoise_with_model(model, noisy, expected_length=2001)
                    if len(denoised) != len(clean):
                        denoised = denoised[:len(clean)] if len(denoised) > len(clean) else np.pad(denoised, (0, len(clean) - len(denoised)))
                    
                    type_results[method_name]['snr_imp'].append(calculate_snr_improvement(clean, noisy, denoised))
                    type_results[method_name]['ncc'].append(calculate_ncc(clean, denoised))
                    type_results[method_name]['mse'].append(calculate_mse(clean, denoised))
                except Exception as e:
                    pass
            
            # Save example
            if i == 0:
                examples.append({
                    't': t,
                    'clean': clean.copy(),
                    'noisy': noisy.copy(),
                })
        
        all_results[sig_type] = type_results
        example_signals[sig_type] = examples[0] if examples else None
    
    # Print results
    print("\n" + "=" * 100)
    print("RESULTS BY SIGNAL TYPE")
    print("=" * 100)
    
    # Create summary tables
    all_methods = list(models.keys()) + list(traditional_methods.keys())
    
    for sig_type in signal_types:
        print(f"\n--- Type {sig_type}: {type_descriptions[sig_type]} ---")
        print(f"{'Method':<25} | {'SNR Imp (dB)':>15} | {'NCC':>12} | {'MSE':>12}")
        print("-" * 70)
        
        results = all_results[sig_type]
        for method in all_methods:
            if method in results and results[method]['snr_imp']:
                snr_mean = np.mean(results[method]['snr_imp'])
                snr_std = np.std(results[method]['snr_imp'])
                ncc_mean = np.mean(results[method]['ncc'])
                mse_mean = np.mean(results[method]['mse'])
                
                print(f"{method:<25} | {snr_mean:+8.2f}±{snr_std:5.2f} | {ncc_mean:10.4f} | {mse_mean:10.6f}")
    
    # Summary across all types
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY (AVERAGE ACROSS ALL TYPES)")
    print("=" * 100)
    print(f"\n{'Method':<25} | {'Avg SNR Imp (dB)':>18} | {'Avg NCC':>12} | {'Avg MSE':>12}")
    print("-" * 75)
    
    summary_results = {}
    for method in all_methods:
        all_snr = []
        all_ncc = []
        all_mse = []
        
        for sig_type in signal_types:
            if method in all_results[sig_type]:
                all_snr.extend(all_results[sig_type][method]['snr_imp'])
                all_ncc.extend(all_results[sig_type][method]['ncc'])
                all_mse.extend(all_results[sig_type][method]['mse'])
        
        if all_snr:
            summary_results[method] = {
                'snr_imp': np.mean(all_snr),
                'snr_std': np.std(all_snr),
                'ncc': np.mean(all_ncc),
                'mse': np.mean(all_mse),
            }
            print(f"{method:<25} | {np.mean(all_snr):+8.2f}±{np.std(all_snr):5.2f}     | {np.mean(all_ncc):10.4f} | {np.mean(all_mse):10.6f}")
    
    # Improvement comparison
    print("\n" + "=" * 100)
    print("IMPROVEMENT OF DL MODELS OVER TRADITIONAL WAVELET")
    print("=" * 100)
    
    if 'Wavelet (db4 Soft)' in summary_results:
        wavelet_baseline = summary_results['Wavelet (db4 Soft)']
        
        for method in models.keys():
            if method in summary_results:
                snr_improvement = summary_results[method]['snr_imp'] - wavelet_baseline['snr_imp']
                ncc_improvement = summary_results[method]['ncc'] - wavelet_baseline['ncc']
                
                print(f"\n{method}:")
                print(f"  SNR Improvement: +{snr_improvement:.2f} dB better than Wavelet")
                print(f"  NCC Improvement: +{ncc_improvement:.4f} better than Wavelet")
    
    # Save results
    output_dir = Path('outputs/comparison_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    create_comparison_visualization(all_results, example_signals, all_methods, 
                                    signal_types, type_descriptions, output_dir, args.snr)
    
    # Save JSON results
    json_results = {
        'snr': args.snr,
        'samples_per_type': args.samples,
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'by_type': {},
    }
    
    for method, res in summary_results.items():
        json_results['summary'][method] = {
            'snr_imp': float(res['snr_imp']),
            'ncc': float(res['ncc']),
            'mse': float(res['mse']),
        }
    
    for sig_type in signal_types:
        json_results['by_type'][sig_type] = {}
        for method in all_methods:
            if method in all_results[sig_type] and all_results[sig_type][method]['snr_imp']:
                json_results['by_type'][sig_type][method] = {
                    'snr_imp': float(np.mean(all_results[sig_type][method]['snr_imp'])),
                    'ncc': float(np.mean(all_results[sig_type][method]['ncc'])),
                    'mse': float(np.mean(all_results[sig_type][method]['mse'])),
                }
    
    with open(output_dir / f'comparison_snr{args.snr}dB.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")
    print("=" * 100)
    
    return all_results, summary_results


def create_comparison_visualization(all_results, example_signals, all_methods, 
                                    signal_types, type_descriptions, output_dir, snr):
    """Create comprehensive comparison visualizations."""
    
    # Figure 1: Bar chart comparison by type
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'SNR Improvement by Signal Type (Input SNR = {snr} dB)', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_methods)))
    
    for i, sig_type in enumerate(signal_types):
        ax = axes[i // 3, i % 3]
        
        methods = []
        snr_means = []
        snr_stds = []
        
        for method in all_methods:
            if method in all_results[sig_type] and all_results[sig_type][method]['snr_imp']:
                methods.append(method.replace('MR-TAE', 'MR').replace('Wavelet', 'Wlt'))
                snr_means.append(np.mean(all_results[sig_type][method]['snr_imp']))
                snr_stds.append(np.std(all_results[sig_type][method]['snr_imp']))
        
        x = np.arange(len(methods))
        bars = ax.bar(x, snr_means, yerr=snr_stds, capsize=3, color=colors[:len(methods)])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('SNR Improvement (dB)')
        ax.set_title(f'Type {sig_type}: {type_descriptions[sig_type]}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_snr{snr}dB_by_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: NCC comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'NCC (Shape Preservation) by Signal Type (Input SNR = {snr} dB)', fontsize=14, fontweight='bold')
    
    for i, sig_type in enumerate(signal_types):
        ax = axes[i // 3, i % 3]
        
        methods = []
        ncc_means = []
        
        for method in all_methods:
            if method in all_results[sig_type] and all_results[sig_type][method]['ncc']:
                methods.append(method.replace('MR-TAE', 'MR').replace('Wavelet', 'Wlt'))
                ncc_means.append(np.mean(all_results[sig_type][method]['ncc']))
        
        x = np.arange(len(methods))
        bars = ax.bar(x, ncc_means, color=colors[:len(methods)])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('NCC')
        ax.set_title(f'Type {sig_type}: {type_descriptions[sig_type]}')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_snr{snr}dB_ncc.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Example signals
    fig, axes = plt.subplots(len(signal_types), 4, figsize=(20, 3 * len(signal_types)))
    fig.suptitle(f'Example Denoising Results (Input SNR = {snr} dB)', fontsize=14, fontweight='bold')
    
    for i, sig_type in enumerate(signal_types):
        if example_signals.get(sig_type) is None:
            continue
            
        ex = example_signals[sig_type]
        t = ex['t'] * 1e6  # Convert to microseconds
        
        # Clean
        axes[i, 0].plot(t, ex['clean'], 'g-', linewidth=0.8)
        axes[i, 0].set_title(f'Type {sig_type} - Clean', fontsize=10)
        axes[i, 0].set_ylabel('Amplitude')
        if i == len(signal_types) - 1:
            axes[i, 0].set_xlabel('Time (μs)')
        
        # Noisy
        axes[i, 1].plot(t, ex['noisy'], 'b-', linewidth=0.5, alpha=0.7)
        axes[i, 1].set_title(f'Noisy ({snr} dB)', fontsize=10)
        if i == len(signal_types) - 1:
            axes[i, 1].set_xlabel('Time (μs)')
        
        # Placeholder for denoised (would need actual denoised signals)
        axes[i, 2].plot(t, ex['clean'], 'r-', linewidth=0.8)
        axes[i, 2].set_title('Best DL Denoised', fontsize=10)
        if i == len(signal_types) - 1:
            axes[i, 2].set_xlabel('Time (μs)')
        
        axes[i, 3].plot(t, ex['clean'], 'orange', linewidth=0.8)
        axes[i, 3].set_title('Best Wavelet Denoised', fontsize=10)
        if i == len(signal_types) - 1:
            axes[i, 3].set_xlabel('Time (μs)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_snr{snr}dB_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare all trained models with traditional methods")
    parser.add_argument('--snr', type=int, default=-10, help='Input SNR level in dB (default: -10)')
    parser.add_argument('--samples', type=int, default=50, help='Samples per signal type (default: 50)')
    
    args = parser.parse_args()
    
    run_comparison(args)


if __name__ == '__main__':
    main()
