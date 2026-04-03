#!/usr/bin/env python
"""
evaluate_results.py - Evaluate and visualize training results.

Loads the trained Baseline and SOTA models and generates comparison visualizations.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path("outputs/extended_training")
BATCH_SIZE = 32


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    config = get_config()
    model = create_model(config.model).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {checkpoint_path.name}")
    print(f"  Best Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  SNR Improvement: {checkpoint.get('snr_imp', 'N/A'):.2f} dB")
    print(f"  NCC: {checkpoint.get('ncc', 'N/A'):.4f}")
    
    return model, checkpoint


def evaluate_model(model, val_loader, model_name):
    """Evaluate model on validation set."""
    model.eval()
    
    snr_improvements = []
    nccs = []
    rmses = []
    snr_levels = {-5: [], -10: [], -15: [], -20: []}
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating"):
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            input_snr = batch['snr']
            
            with autocast():
                denoised, _ = model(noisy)
            
            noisy_np = noisy.squeeze(1).cpu().numpy()
            clean_np = clean.squeeze(1).cpu().numpy()
            denoised_np = denoised.squeeze(1).cpu().numpy()
            
            for i in range(len(noisy_np)):
                try:
                    snr_imp = calculate_snr_improvement(noisy_np[i], denoised_np[i], clean_np[i])
                    ncc = calculate_ncc(denoised_np[i], clean_np[i])
                    rmse = calculate_rmse(denoised_np[i], clean_np[i])
                    
                    snr_improvements.append(snr_imp)
                    nccs.append(ncc)
                    rmses.append(rmse)
                    
                    # Track by SNR level
                    snr_val = int(round(input_snr[i].item()))
                    for level in snr_levels.keys():
                        if abs(snr_val - level) < 3:
                            snr_levels[level].append(snr_imp)
                            break
                except:
                    pass
    
    results = {
        'snr_imp_mean': np.mean(snr_improvements),
        'snr_imp_std': np.std(snr_improvements),
        'ncc_mean': np.mean(nccs),
        'ncc_std': np.std(nccs),
        'rmse_mean': np.mean(rmses),
        'rmse_std': np.std(rmses),
        'snr_by_level': {k: np.mean(v) if v else 0 for k, v in snr_levels.items()}
    }
    
    return results


def visualize_comparison(model_baseline, model_sota, val_loader, save_path, num_examples=8):
    """Generate side-by-side comparison visualization."""
    model_baseline.eval()
    model_sota.eval()
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 3*num_examples))
    fig.suptitle('MR-TAE-Fusion: Baseline vs SOTA Comparison', fontsize=16, fontweight='bold')
    
    columns = ['Noisy Input', 'Baseline (MSE)', 'SOTA (Charbonnier)', 'Ground Truth']
    for ax, col in zip(axes[0], columns):
        ax.set_title(col, fontweight='bold', fontsize=12)
    
    examples_collected = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if examples_collected >= num_examples:
                break
            
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            snr = batch['snr']
            
            with autocast():
                denoised_baseline, _ = model_baseline(noisy)
                denoised_sota, _ = model_sota(noisy)
            
            noisy_np = noisy.squeeze(1).cpu().numpy()
            clean_np = clean.squeeze(1).cpu().numpy()
            baseline_np = denoised_baseline.squeeze(1).cpu().numpy()
            sota_np = denoised_sota.squeeze(1).cpu().numpy()
            
            for i in range(min(len(noisy_np), num_examples - examples_collected)):
                idx = examples_collected
                
                # Calculate metrics
                try:
                    snr_imp_base = calculate_snr_improvement(noisy_np[i], baseline_np[i], clean_np[i])
                    snr_imp_sota = calculate_snr_improvement(noisy_np[i], sota_np[i], clean_np[i])
                    ncc_base = calculate_ncc(baseline_np[i], clean_np[i])
                    ncc_sota = calculate_ncc(sota_np[i], clean_np[i])
                except:
                    snr_imp_base = snr_imp_sota = 0
                    ncc_base = ncc_sota = 0
                
                # Plot
                axes[idx, 0].plot(noisy_np[i], 'b-', alpha=0.7, linewidth=0.5)
                axes[idx, 0].set_ylabel(f'SNR: {snr[i].item():.0f}dB', fontsize=10)
                axes[idx, 0].grid(True, alpha=0.3)
                
                axes[idx, 1].plot(baseline_np[i], 'orange', linewidth=0.8)
                axes[idx, 1].set_title(f'SNR↑: {snr_imp_base:.1f}dB | NCC: {ncc_base:.3f}', fontsize=9)
                axes[idx, 1].grid(True, alpha=0.3)
                
                axes[idx, 2].plot(sota_np[i], 'r-', linewidth=0.8)
                axes[idx, 2].set_title(f'SNR↑: {snr_imp_sota:.1f}dB | NCC: {ncc_sota:.3f}', fontsize=9)
                axes[idx, 2].grid(True, alpha=0.3)
                
                axes[idx, 3].plot(clean_np[i], 'g-', linewidth=0.8)
                axes[idx, 3].grid(True, alpha=0.3)
                
                examples_collected += 1
    
    for ax in axes[-1]:
        ax.set_xlabel('Sample')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_table(results_baseline, results_sota):
    """Create a comparison table."""
    print("\n" + "="*70)
    print("                      PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Metric':<25} | {'Baseline (MSE)':<20} | {'SOTA (Charbonnier)':<20}")
    print("-"*70)
    print(f"{'SNR Improvement (dB)':<25} | {results_baseline['snr_imp_mean']:.2f} ± {results_baseline['snr_imp_std']:.2f} | {results_sota['snr_imp_mean']:.2f} ± {results_sota['snr_imp_std']:.2f}")
    print(f"{'NCC (Shape Fidelity)':<25} | {results_baseline['ncc_mean']:.4f} ± {results_baseline['ncc_std']:.4f} | {results_sota['ncc_mean']:.4f} ± {results_sota['ncc_std']:.4f}")
    print(f"{'RMSE':<25} | {results_baseline['rmse_mean']:.6f} ± {results_baseline['rmse_std']:.6f} | {results_sota['rmse_mean']:.6f} ± {results_sota['rmse_std']:.6f}")
    print("-"*70)
    
    print("\nSNR Improvement by Input SNR Level:")
    print(f"{'Input SNR':<15} | {'Baseline':<15} | {'SOTA':<15} | {'Improvement':<15}")
    print("-"*60)
    for level in [-5, -10, -15, -20]:
        base = results_baseline['snr_by_level'].get(level, 0)
        sota = results_sota['snr_by_level'].get(level, 0)
        diff = sota - base
        print(f"{level:>10} dB   | {base:>10.2f} dB | {sota:>10.2f} dB | {diff:>+10.2f} dB")
    print("="*70)
    
    # Overall improvement
    improvement = results_sota['snr_imp_mean'] - results_baseline['snr_imp_mean']
    print(f"\n🎯 OVERALL IMPROVEMENT: SOTA is {improvement:+.2f} dB better than Baseline!")
    print("="*70)


def main():
    print("="*60)
    print("MR-TAE-Fusion Evaluation and Comparison")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load models
    print("\nLoading trained models...")
    model_baseline, ckpt_baseline = load_model(SAVE_DIR / 'baseline_best.pth')
    model_sota, ckpt_sota = load_model(SAVE_DIR / 'sota_best.pth')
    
    # Create validation dataset
    config = get_config()
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=500,
        mode='val',
        epoch=99,  # Hardest noise level
        seed=999
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate both models
    results_baseline = evaluate_model(model_baseline, val_loader, "Baseline")
    results_sota = evaluate_model(model_sota, val_loader, "SOTA")
    
    # Print comparison table
    create_summary_table(results_baseline, results_sota)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_comparison(
        model_baseline, model_sota, val_loader,
        SAVE_DIR / 'comparison_results.png', num_examples=8
    )
    
    print(f"\n✅ Evaluation complete! Results saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
