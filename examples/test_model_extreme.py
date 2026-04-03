#!/usr/bin/env python
"""
test_model_extreme.py - Test trained model on extreme samples (-5dB, multiple PD)
and visualize results. Saves model for easy use.
"""

import sys
from pathlib import Path
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data.pulse_generators import generate_pd_signal
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 5
CLASS_NAMES = ['Background', 'Corona', 'Surface', 'Internal', 'Treeing']
CLASS_COLORS = ['lightgray', 'red', 'blue', 'green', 'purple']

# Latest model from 50K x 50 epoch training
MODEL_PATH = Path("outputs/run_20251221_173412/best_model.pth")
OUTPUT_DIR = Path("outputs/model_evaluation")
SAVED_MODEL_PATH = Path("saved_models/mr_tae_fusion_50k_50ep.pth")


def add_noise(signal, snr_db):
    """Add AWGN noise at specified SNR."""
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:
        signal_power = 1.0
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise


def generate_multi_pd_signal(config, num_pds=3, snr_db=-5):
    """Generate signal with multiple different PD types at -5dB."""
    pd_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    selected_types = np.random.choice(pd_types, size=num_pds, replace=False)
    
    signal_length = config.signal.signal_length
    clean = np.zeros(signal_length)
    mask = np.zeros(signal_length, dtype=np.int64)
    
    # Create time segments for each PD type
    segment_len = signal_length // num_pds
    
    for i, pd_type in enumerate(selected_types):
        start_idx = i * segment_len
        end_idx = (i + 1) * segment_len if i < num_pds - 1 else signal_length
        
        # Generate PD signal
        _, pd_signal, pd_mask = generate_pd_signal(pd_type, config.signal)
        
        # Fit to segment
        seg_len = end_idx - start_idx
        if len(pd_signal) >= seg_len:
            clean[start_idx:end_idx] = pd_signal[:seg_len]
            mask[start_idx:end_idx] = pd_mask[:seg_len]
        else:
            clean[start_idx:start_idx+len(pd_signal)] = pd_signal
            mask[start_idx:start_idx+len(pd_mask)] = pd_mask
    
    # Add noise
    noisy = add_noise(clean, snr_db)
    
    # Normalize
    max_amp = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-8)
    noisy = noisy / max_amp
    clean = clean / max_amp
    
    return noisy, clean, mask, selected_types


def load_model():
    """Load the trained model."""
    print(f"Loading model from: {MODEL_PATH}")
    
    config = get_config()
    config.model.num_classes = NUM_CLASSES
    
    model = create_model(config.model).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"  Loaded from epoch: {epoch}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"  Model parameters: {model.count_parameters():,}")
    
    return model, config


def test_on_extreme_samples(model, config, num_samples=10, snr_db=-5):
    """Test model on extreme samples and collect metrics."""
    print(f"\n{'='*70}")
    print(f"Testing on {num_samples} extreme samples at {snr_db}dB SNR")
    print('='*70)
    
    results = []
    
    for i in range(num_samples):
        num_pds = np.random.randint(2, 5)  # 2-4 PD types per sample
        noisy, clean, mask, pd_types = generate_multi_pd_signal(config, num_pds, snr_db)
        
        # Convert to tensor
        noisy_t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            with autocast('cuda', enabled=True):
                denoised, seg_logits = model(noisy_t)
        
        denoised_np = denoised[0, 0].cpu().numpy()
        pred_mask = seg_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Calculate metrics
        snr_imp = calculate_snr_improvement(noisy, denoised_np, clean)
        ncc = calculate_ncc(denoised_np, clean)
        rmse = calculate_rmse(denoised_np, clean)
        
        # Segmentation accuracy
        correct = (pred_mask == mask).sum()
        total = len(mask)
        seg_acc = correct / total
        
        results.append({
            'noisy': noisy,
            'clean': clean,
            'denoised': denoised_np,
            'mask': mask,
            'pred_mask': pred_mask,
            'pd_types': pd_types,
            'snr_imp': snr_imp,
            'ncc': ncc,
            'rmse': rmse,
            'seg_acc': seg_acc
        })
        
        print(f"  Sample {i+1}: PDs={','.join(pd_types)} | SNR+{snr_imp:.2f}dB | NCC={ncc:.4f} | SegAcc={seg_acc*100:.1f}%")
    
    # Summary
    avg_snr = np.mean([r['snr_imp'] for r in results])
    avg_ncc = np.mean([r['ncc'] for r in results])
    avg_acc = np.mean([r['seg_acc'] for r in results])
    
    print(f"\n{'='*70}")
    print(f"SUMMARY @ {snr_db}dB:")
    print(f"  Average SNR Improvement: {avg_snr:.2f} dB")
    print(f"  Average NCC: {avg_ncc:.4f}")
    print(f"  Average Segmentation Accuracy: {avg_acc*100:.1f}%")
    print('='*70)
    
    return results


def visualize_results(results, save_dir, num_show=5):
    """Create visualization plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n = min(num_show, len(results))
    
    fig, axes = plt.subplots(n, 3, figsize=(18, 4*n))
    fig.suptitle('Model Performance on Extreme Samples (-5dB, Multiple PD Types)', 
                 fontsize=14, fontweight='bold')
    
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        r = results[i]
        
        # Noisy input
        axes[i][0].plot(r['noisy'], 'b-', linewidth=0.5, alpha=0.7)
        axes[i][0].set_title(f"Noisy Input (SNR=-5dB) | PDs: {','.join(r['pd_types'])}")
        axes[i][0].set_ylabel(f"Sample {i+1}")
        
        # Denoised vs Clean
        axes[i][1].plot(r['clean'], 'g-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        axes[i][1].plot(r['denoised'], 'r--', linewidth=1, label='Denoised')
        axes[i][1].set_title(f"Denoised vs Clean | SNR+{r['snr_imp']:.1f}dB | NCC={r['ncc']:.3f}")
        axes[i][1].legend()
        
        # Segmentation
        axes[i][2].plot(r['mask'], 'g-', linewidth=1.5, label='True Mask')
        axes[i][2].plot(r['pred_mask'], 'r--', linewidth=1, label='Predicted')
        axes[i][2].set_title(f"Segmentation | Acc={r['seg_acc']*100:.1f}%")
        axes[i][2].legend()
    
    plt.tight_layout()
    save_path = save_dir / 'extreme_samples_test.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {save_path}")
    
    # Create metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    snr_vals = [r['snr_imp'] for r in results]
    ncc_vals = [r['ncc'] for r in results]
    acc_vals = [r['seg_acc'] * 100 for r in results]
    
    axes[0].bar(range(len(snr_vals)), snr_vals, color='blue', alpha=0.7)
    axes[0].axhline(np.mean(snr_vals), color='red', linestyle='--', label=f'Avg: {np.mean(snr_vals):.2f}')
    axes[0].set_title('SNR Improvement (dB)')
    axes[0].set_xlabel('Sample')
    axes[0].legend()
    
    axes[1].bar(range(len(ncc_vals)), ncc_vals, color='green', alpha=0.7)
    axes[1].axhline(np.mean(ncc_vals), color='red', linestyle='--', label=f'Avg: {np.mean(ncc_vals):.4f}')
    axes[1].set_title('NCC (Shape Correlation)')
    axes[1].set_xlabel('Sample')
    axes[1].legend()
    
    axes[2].bar(range(len(acc_vals)), acc_vals, color='orange', alpha=0.7)
    axes[2].axhline(np.mean(acc_vals), color='red', linestyle='--', label=f'Avg: {np.mean(acc_vals):.1f}%')
    axes[2].set_title('Segmentation Accuracy (%)')
    axes[2].set_xlabel('Sample')
    axes[2].legend()
    
    plt.tight_layout()
    metrics_path = save_dir / 'extreme_samples_metrics.png'
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics chart saved to: {metrics_path}")


def save_model_for_easy_use(model, config):
    """Save model in a simple format for easy loading."""
    SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save complete package
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': {
            'num_classes': NUM_CLASSES,
            'signal_length': config.signal.signal_length,
            'class_names': CLASS_NAMES,
        },
        'training_info': {
            'samples': 50000,
            'epochs': 50,
            'phases': '6-phase curriculum',
            'alpha_ncc': 0.8,
            'alpha_seg': 0.4,
        }
    }
    
    torch.save(save_dict, SAVED_MODEL_PATH)
    print(f"\n✓ Model saved to: {SAVED_MODEL_PATH}")
    print(f"  To load: checkpoint = torch.load('{SAVED_MODEL_PATH}')")
    print(f"           model.load_state_dict(checkpoint['model_state_dict'])")


def main():
    print("="*70)
    print("MR-TAE-FUSION MODEL EVALUATION")
    print("Testing on Extreme Samples (-5dB, Multiple PD)")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # Load model
    model, config = load_model()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Test on extreme samples at different SNR levels
    print("\n" + "="*70)
    print("Testing at -5dB SNR (Extreme Noise)")
    print("="*70)
    results_5db = test_on_extreme_samples(model, config, num_samples=10, snr_db=-5)
    
    print("\n" + "="*70)
    print("Testing at -10dB SNR (Very Hard)")
    print("="*70)
    results_10db = test_on_extreme_samples(model, config, num_samples=10, snr_db=-10)
    
    print("\n" + "="*70)
    print("Testing at -15dB SNR (Extremely Hard)")
    print("="*70)
    results_15db = test_on_extreme_samples(model, config, num_samples=10, snr_db=-15)
    
    # Visualize
    visualize_results(results_5db, OUTPUT_DIR / 'snr_minus5dB', num_show=5)
    visualize_results(results_10db, OUTPUT_DIR / 'snr_minus10dB', num_show=5)
    visualize_results(results_15db, OUTPUT_DIR / 'snr_minus15dB', num_show=5)
    
    # Save model for easy use
    save_model_for_easy_use(model, config)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL COMPARISON ACROSS SNR LEVELS:")
    print("="*70)
    print(f"{'SNR Level':<15} {'SNR Imp (dB)':<15} {'NCC':<15} {'Seg Acc':<15}")
    print("-"*60)
    
    for name, results in [('-5dB', results_5db), ('-10dB', results_10db), ('-15dB', results_15db)]:
        avg_snr = np.mean([r['snr_imp'] for r in results])
        avg_ncc = np.mean([r['ncc'] for r in results])
        avg_acc = np.mean([r['seg_acc'] for r in results]) * 100
        print(f"{name:<15} {avg_snr:<15.2f} {avg_ncc:<15.4f} {avg_acc:<15.1f}%")
    
    print("\n✓ All outputs saved to:", OUTPUT_DIR)
    print("✓ Model saved to:", SAVED_MODEL_PATH)


if __name__ == '__main__':
    main()
