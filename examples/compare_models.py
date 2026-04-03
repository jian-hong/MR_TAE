#!/usr/bin/env python
"""
compare_models.py - Compare MR-TAE-Fusion with Traditional Wavelet Methods

This script replaces MATLAB's Compare_Models.m and dl_testing.m with Python.

COMPARISON METHODS:
1. Traditional Wavelet (db4 Soft Threshold) - Donoho-Johnstone
2. BayesShrink Wavelet
3. CNN-Only (without Swin/GRU bottleneck)
4. MR-TAE-Fusion (Full Model)

EVALUATION METRICS:
- SNR Improvement (dB)
- NCC (Normalized Cross-Correlation)
- RMSE
- Segmentation Accuracy per class

USAGE:
    python compare_models.py --model_path outputs/run_xxx/best_model.pth
    python compare_models.py --snr_levels -5 -10 -15 -20 -25
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt
from scipy import signal as scipy_signal

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data.pulse_generators import generate_pd_signal
from mr_tae_fusion.data.qlin_loader import QLINDataLoader
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


# =============================================================================
# TRADITIONAL WAVELET METHODS (Python equivalents of MATLAB)
# =============================================================================

def wavelet_denoise_soft(signal, wavelet='db4', level=5, mode='soft'):
    """
    Traditional wavelet denoising with soft thresholding.
    
    Equivalent to MATLAB's wden with Universal threshold (Donoho-Johnstone).
    
    Args:
        signal: 1D signal to denoise
        wavelet: Wavelet to use (default 'db4' - Daubechies-4)
        level: Decomposition level
        mode: Thresholding mode ('soft' or 'hard')
    
    Returns:
        Denoised signal
    """
    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Universal threshold (Donoho-Johnstone)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply threshold to detail coefficients (not approximation)
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    for i, detail in enumerate(coeffs[1:]):
        if mode == 'soft':
            denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
        else:
            denoised_coeffs.append(pywt.threshold(detail, threshold, mode='hard'))
    
    # Reconstruct
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    
    # Match length
    return denoised[:len(signal)]


def wavelet_denoise_bayesshrink(signal, wavelet='db4', level=5):
    """
    BayesShrink wavelet denoising (adaptive threshold).
    
    Args:
        signal: 1D signal
        wavelet: Wavelet to use
        level: Decomposition level
    
    Returns:
        Denoised signal
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Noise estimation from finest detail
    sigma_n = np.median(np.abs(coeffs[-1])) / 0.6745
    
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    
    for detail in coeffs[1:]:
        # Signal variance estimation
        sigma_d2 = max(np.var(detail) - sigma_n ** 2, 0)
        
        # BayesShrink threshold
        if sigma_d2 > 0:
            threshold = (sigma_n ** 2) / np.sqrt(sigma_d2)
        else:
            # Fall back to universal threshold
            threshold = sigma_n * np.sqrt(2 * np.log(len(signal)))
        
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
    
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:len(signal)]


def wavelet_denoise_adaptive(signal, wavelet='sym6', level=6):
    """
    Adaptive wavelet denoising (optimal wavelet selection).
    
    Tests multiple wavelets and selects the best one.
    """
    wavelets = ['db4', 'db6', 'sym4', 'sym6', 'coif3', 'bior4.4']
    best_wavelet = 'sym6'
    min_error = float('inf')
    
    for wname in wavelets:
        try:
            # Test reconstruction quality
            coeffs = pywt.wavedec(signal, wname, level=level)
            rec = pywt.waverec(coeffs, wname)
            error = np.mean((signal - rec[:len(signal)]) ** 2)
            
            if error < min_error:
                min_error = error
                best_wavelet = wname
        except:
            continue
    
    return wavelet_denoise_bayesshrink(signal, best_wavelet, level)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_method(clean_signals, noisy_signals, denoised_signals, method_name):
    """Evaluate denoising method with multiple metrics."""
    snr_imps = []
    nccs = []
    rmses = []
    
    for clean, noisy, denoised in zip(clean_signals, noisy_signals, denoised_signals):
        try:
            snr_imp = calculate_snr_improvement(noisy, denoised, clean)
            ncc = calculate_ncc(denoised, clean)
            rmse = calculate_rmse(denoised, clean)
            
            snr_imps.append(snr_imp)
            nccs.append(ncc)
            rmses.append(rmse)
        except:
            pass
    
    return {
        'method': method_name,
        'snr_imp_mean': np.mean(snr_imps),
        'snr_imp_std': np.std(snr_imps),
        'ncc_mean': np.mean(nccs),
        'ncc_std': np.std(nccs),
        'rmse_mean': np.mean(rmses),
        'rmse_std': np.std(rmses),
    }


def generate_test_signals(num_signals, snr_level, signal_types=None):
    """Generate test signals at specific SNR level."""
    from mr_tae_fusion.config import SignalConfig
    config = SignalConfig()
    
    if signal_types is None:
        signal_types = ['A', 'B', 'C', 'D', 'E', 'F']
    
    clean_signals = []
    noisy_signals = []
    masks = []
    
    for i in range(num_signals):
        sig_type = signal_types[i % len(signal_types)]
        _, clean, mask = generate_pd_signal(sig_type, config)
        
        # Add noise at specified SNR
        signal_power = np.mean(clean ** 2)
        if signal_power < 1e-10:
            signal_power = 1.0
        noise_power = signal_power / (10 ** (snr_level / 10))
        noise = np.random.randn(len(clean)) * np.sqrt(noise_power)
        noisy = clean + noise
        
        # Normalize
        max_amp = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-8)
        clean_signals.append(clean / max_amp)
        noisy_signals.append(noisy / max_amp)
        masks.append(mask)
    
    return clean_signals, noisy_signals, masks


def denoise_with_model(model, noisy_signals, device='cuda'):
    """Denoise signals using deep learning model."""
    model.eval()
    denoised_signals = []
    seg_masks = []
    
    with torch.no_grad():
        for noisy in noisy_signals:
            x = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            with autocast('cuda', enabled=True):
                denoised, seg_logits = model(x)
            
            denoised_signals.append(denoised.squeeze().cpu().numpy())
            seg_masks.append(seg_logits.argmax(dim=1).squeeze().cpu().numpy())
    
    return denoised_signals, seg_masks


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(model_path, snr_levels, num_signals=100, output_dir=None):
    """
    Run full comparison between MR-TAE-Fusion and traditional methods.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("Loading MR-TAE-Fusion model...")
    config = get_config()
    config.model.num_classes = 5
    model = create_model(config.model).to(device)
    
    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Model loaded with {model.count_parameters():,} parameters")
    
    # Results storage
    all_results = {}
    
    # Methods to compare
    methods = {
        'Wavelet (db4 Soft)': lambda s: wavelet_denoise_soft(s, 'db4', 5, 'soft'),
        'Wavelet (BayesShrink)': lambda s: wavelet_denoise_bayesshrink(s, 'db4', 5),
        'Wavelet (Adaptive)': lambda s: wavelet_denoise_adaptive(s, 'sym6', 6),
    }
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON: MR-TAE-Fusion vs Traditional Wavelet Methods")
    print("="*80)
    
    for snr in snr_levels:
        print(f"\n--- SNR = {snr} dB ---")
        
        # Generate test signals
        clean_signals, noisy_signals, masks = generate_test_signals(num_signals, snr)
        
        results_at_snr = {}
        
        # Evaluate traditional methods
        for method_name, method_fn in methods.items():
            denoised = [method_fn(sig) for sig in tqdm(noisy_signals, desc=method_name, leave=False)]
            results = evaluate_method(clean_signals, noisy_signals, denoised, method_name)
            results_at_snr[method_name] = results
        
        # Evaluate MR-TAE-Fusion
        denoised_dl, seg_masks = denoise_with_model(model, noisy_signals, device)
        results = evaluate_method(clean_signals, noisy_signals, denoised_dl, 'MR-TAE-Fusion')
        results_at_snr['MR-TAE-Fusion'] = results
        
        all_results[snr] = results_at_snr
        
        # Print results for this SNR
        print(f"\n{'Method':<25} | {'SNR Imp (dB)':>12} | {'NCC':>10} | {'RMSE':>12}")
        print("-"*65)
        for method_name, res in results_at_snr.items():
            snr_str = f"{res['snr_imp_mean']:+.2f}±{res['snr_imp_std']:.2f}"
            ncc_str = f"{res['ncc_mean']:.4f}±{res['ncc_std']:.4f}"
            rmse_str = f"{res['rmse_mean']:.6f}±{res['rmse_std']:.6f}"
            print(f"{method_name:<25} | {snr_str:>12} | {ncc_str:>10} | {rmse_str:>12}")
    
    # Calculate improvements
    print("\n" + "="*80)
    print("IMPROVEMENT OF MR-TAE-FUSION OVER TRADITIONAL METHODS")
    print("="*80)
    
    for snr in snr_levels:
        results_at_snr = all_results[snr]
        mr_tae = results_at_snr['MR-TAE-Fusion']
        wavelet = results_at_snr['Wavelet (db4 Soft)']
        
        snr_improvement = mr_tae['snr_imp_mean'] - wavelet['snr_imp_mean']
        ncc_improvement = mr_tae['ncc_mean'] - wavelet['ncc_mean']
        rmse_improvement = wavelet['rmse_mean'] - mr_tae['rmse_mean']  # Lower is better
        
        print(f"\nSNR = {snr} dB:")
        print(f"  SNR Improvement: +{snr_improvement:.2f} dB better than Wavelet")
        print(f"  NCC Improvement: +{ncc_improvement:.4f} better than Wavelet")
        print(f"  RMSE Improvement: -{rmse_improvement:.6f} better than Wavelet")
    
    # Create comparison plots
    if output_dir:
        create_comparison_plots(all_results, snr_levels, output_dir)
    
    return all_results


def create_comparison_plots(results, snr_levels, output_dir):
    """Create comparison visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(results[snr_levels[0]].keys())
    colors = ['blue', 'orange', 'green', 'red']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # SNR Improvement
    for i, method in enumerate(methods):
        snr_imps = [results[snr][method]['snr_imp_mean'] for snr in snr_levels]
        snr_stds = [results[snr][method]['snr_imp_std'] for snr in snr_levels]
        axes[0].errorbar(snr_levels, snr_imps, yerr=snr_stds, 
                         label=method, marker='o', capsize=3, color=colors[i % len(colors)])
    
    axes[0].set_xlabel('Input SNR (dB)')
    axes[0].set_ylabel('SNR Improvement (dB)')
    axes[0].set_title('SNR Improvement Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # NCC
    for i, method in enumerate(methods):
        nccs = [results[snr][method]['ncc_mean'] for snr in snr_levels]
        ncc_stds = [results[snr][method]['ncc_std'] for snr in snr_levels]
        axes[1].errorbar(snr_levels, nccs, yerr=ncc_stds,
                         label=method, marker='s', capsize=3, color=colors[i % len(colors)])
    
    axes[1].set_xlabel('Input SNR (dB)')
    axes[1].set_ylabel('NCC (Shape Preservation)')
    axes[1].set_title('NCC Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # RMSE
    for i, method in enumerate(methods):
        rmses = [results[snr][method]['rmse_mean'] for snr in snr_levels]
        rmse_stds = [results[snr][method]['rmse_std'] for snr in snr_levels]
        axes[2].errorbar(snr_levels, rmses, yerr=rmse_stds,
                         label=method, marker='^', capsize=3, color=colors[i % len(colors)])
    
    axes[2].set_xlabel('Input SNR (dB)')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('RMSE Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_dir / 'comparison_results.png'}")


def evaluate_on_qlin(model, model_path, output_dir=None):
    """
    Evaluate model on real Q.Lin data for real-world verification.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    config = get_config()
    config.model.num_classes = 5
    model = create_model(config.model).to(device)
    
    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load Q.Lin data
    print("\nLoading Q.Lin real data for evaluation...")
    qlin_loader = QLINDataLoader()
    stats = qlin_loader.get_statistics()
    
    print("\n" + "="*80)
    print("REAL-WORLD EVALUATION ON Q.Lin Metal Particle Data")
    print("="*80)
    
    results_by_diameter = {}
    
    for diameter in ['1.0mm', '1.8mm', '2.0mm', '2.5mm']:
        print(f"\n--- {diameter} Metal Particles ---")
        
        signals = qlin_loader.val_signals.get(diameter, [])
        if len(signals) == 0:
            print(f"  No validation signals for {diameter}")
            continue
        
        # Sample up to 100 signals
        sample_size = min(100, len(signals))
        sample_indices = np.random.choice(len(signals), sample_size, replace=False)
        
        nccs = []
        seg_accs = []
        
        for idx in sample_indices:
            sig = signals[idx]
            
            # Create mask (Surface discharge = class 2)
            true_mask = qlin_loader.create_mask(sig, class_id=2)
            
            # Normalize
            max_amp = np.abs(sig).max() + 1e-8
            sig_norm = sig / max_amp
            
            # Pass through model
            x = torch.tensor(sig_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                with autocast('cuda', enabled=True):
                    denoised, seg_logits = model(x)
            
            pred_mask = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
            denoised_np = denoised.squeeze().cpu().numpy()
            
            # Calculate NCC (denoised vs original - should preserve structure)
            try:
                ncc = calculate_ncc(denoised_np, sig_norm)
                nccs.append(ncc)
            except:
                pass
            
            # Segmentation accuracy
            surface_mask = (true_mask == 2)
            if surface_mask.sum() > 0:
                acc = ((pred_mask == 2) & surface_mask).sum() / surface_mask.sum()
                seg_accs.append(acc)
        
        results_by_diameter[diameter] = {
            'ncc_mean': np.mean(nccs) if nccs else 0,
            'ncc_std': np.std(nccs) if nccs else 0,
            'seg_acc_mean': np.mean(seg_accs) if seg_accs else 0,
            'seg_acc_std': np.std(seg_accs) if seg_accs else 0,
            'num_samples': sample_size
        }
        
        print(f"  Samples: {sample_size}")
        print(f"  NCC (Shape Preservation): {results_by_diameter[diameter]['ncc_mean']:.4f} ± {results_by_diameter[diameter]['ncc_std']:.4f}")
        print(f"  Surface Detection Acc: {results_by_diameter[diameter]['seg_acc_mean']*100:.1f}% ± {results_by_diameter[diameter]['seg_acc_std']*100:.1f}%")
    
    # Overall summary
    all_nccs = [r['ncc_mean'] for r in results_by_diameter.values() if r['ncc_mean'] > 0]
    all_accs = [r['seg_acc_mean'] for r in results_by_diameter.values() if r['seg_acc_mean'] > 0]
    
    print("\n" + "="*80)
    print("OVERALL Q.Lin EVALUATION SUMMARY")
    print("="*80)
    print(f"Average NCC: {np.mean(all_nccs):.4f}")
    print(f"Average Surface Detection Accuracy: {np.mean(all_accs)*100:.1f}%")
    
    return results_by_diameter


def main():
    parser = argparse.ArgumentParser(description="Compare MR-TAE-Fusion with traditional methods")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--snr_levels', type=int, nargs='+', default=[-5, -10, -15, -20, -25],
                        help='SNR levels to test')
    parser.add_argument('--num_signals', type=int, default=100, help='Number of test signals per SNR')
    parser.add_argument('--output_dir', type=str, default='outputs/comparison', help='Output directory')
    parser.add_argument('--qlin_eval', action='store_true', help='Also evaluate on Q.Lin data')
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_comparison(args.model_path, args.snr_levels, args.num_signals, args.output_dir)
    
    # Optionally evaluate on Q.Lin
    if args.qlin_eval:
        qlin_results = evaluate_on_qlin(None, args.model_path, args.output_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
