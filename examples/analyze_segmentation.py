#!/usr/bin/env python
"""
analyze_segmentation.py - Comprehensive analysis of segmentation and denoising.

Evaluates:
1. Segmentation mask accuracy (IoU, Dice, per-class accuracy)
2. Denoising quality with spike detection
3. Signal generation verification
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.data.pulse_generators import generate_pd_signal
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path("outputs/extended_training")
BATCH_SIZE = 16

# Class names
CLASS_NAMES = ['Background', 'Corona', 'Surface', 'Internal']
CLASS_COLORS = ['lightgray', 'red', 'blue', 'green']


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    config = get_config()
    model = create_model(config.model).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def detect_spikes(signal, threshold=3.0):
    """Detect abnormal spikes in signal."""
    mean = np.mean(signal)
    std = np.std(signal)
    spikes = np.abs(signal - mean) > threshold * std
    spike_indices = np.where(spikes)[0]
    spike_ratio = len(spike_indices) / len(signal)
    max_spike = np.max(np.abs(signal)) if len(signal) > 0 else 0
    return {
        'has_spikes': spike_ratio > 0.001,
        'spike_ratio': spike_ratio,
        'max_value': max_spike,
        'spike_indices': spike_indices
    }


def calculate_segmentation_metrics(pred_mask, true_mask, num_classes=4):
    """Calculate per-class IoU and Dice scores."""
    results = {}
    
    # Overall accuracy
    correct = (pred_mask == true_mask).sum()
    total = len(true_mask)
    results['accuracy'] = correct / total
    
    # Per-class metrics
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)
        
        intersection = (pred_c & true_c).sum()
        union = (pred_c | true_c).sum()
        
        # IoU
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if not true_c.any() else 0.0
        
        # Dice
        if pred_c.sum() + true_c.sum() > 0:
            dice = 2 * intersection / (pred_c.sum() + true_c.sum())
        else:
            dice = 1.0
        
        # Per-class accuracy
        if true_c.sum() > 0:
            class_acc = (pred_c & true_c).sum() / true_c.sum()
        else:
            class_acc = 1.0
        
        results[f'iou_{c}'] = iou
        results[f'dice_{c}'] = dice
        results[f'acc_{c}'] = class_acc
    
    return results


def visualize_segmentation_examples(model, val_loader, save_path, num_examples=6):
    """Visualize signal with its predicted vs true segmentation mask."""
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 3*num_examples))
    fig.suptitle('Segmentation Analysis: Noisy | Predicted Mask | True Mask', fontsize=14, fontweight='bold')
    
    cmap = ListedColormap(CLASS_COLORS)
    examples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if examples >= num_examples:
                break
            
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            true_mask = batch['mask']
            signal_type = batch['type']
            
            with autocast('cuda'):
                denoised, seg_logits = model(noisy)
            
            pred_mask = seg_logits.argmax(dim=1).cpu().numpy()
            
            for i in range(min(len(noisy), num_examples - examples)):
                noisy_np = noisy[i, 0].cpu().numpy()
                clean_np = clean[i, 0].cpu().numpy()
                denoised_np = denoised[i, 0].cpu().numpy()
                true_m = true_mask[i].numpy()
                pred_m = pred_mask[i]
                
                # Calculate metrics
                seg_metrics = calculate_segmentation_metrics(pred_m, true_m)
                spike_info = detect_spikes(denoised_np)
                
                # Check for NaN values
                has_nan = np.isnan(denoised_np).any()
                
                # Noisy signal with ground truth overlay
                ax = axes[examples, 0]
                ax.plot(noisy_np, 'b-', alpha=0.5, linewidth=0.5, label='Noisy')
                ax.plot(clean_np, 'g-', linewidth=0.8, label='Clean')
                ax.set_title(f'Type {signal_type[i].item()}: Input Signal')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_ylabel(f'Sample {examples+1}')
                ax.grid(True, alpha=0.3)
                
                # Predicted mask with denoised signal
                ax = axes[examples, 1]
                ax.plot(denoised_np, 'k-', linewidth=0.5, alpha=0.5)
                # Color background by predicted class
                for c in range(4):
                    mask_c = (pred_m == c)
                    if mask_c.any():
                        ax.fill_between(range(len(pred_m)), 
                                       ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.5,
                                       ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.5,
                                       where=mask_c, alpha=0.3, 
                                       color=CLASS_COLORS[c], label=CLASS_NAMES[c])
                ax.set_title(f'Predicted | Acc: {seg_metrics["accuracy"]*100:.1f}% | Spikes: {spike_info["has_spikes"]}')
                ax.grid(True, alpha=0.3)
                if has_nan:
                    ax.set_title(f'Predicted | HAS NaN VALUES!', color='red')
                
                # True mask with clean signal
                ax = axes[examples, 2]
                ax.plot(clean_np, 'k-', linewidth=0.5, alpha=0.5)
                for c in range(4):
                    mask_c = (true_m == c)
                    if mask_c.any():
                        ax.fill_between(range(len(true_m)), 
                                       ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.5,
                                       ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.5,
                                       where=mask_c, alpha=0.3, 
                                       color=CLASS_COLORS[c])
                ax.set_title(f'Ground Truth | IoU: {seg_metrics["iou_1"]*100:.1f}%/{seg_metrics["iou_2"]*100:.1f}%/{seg_metrics["iou_3"]*100:.1f}%')
                ax.grid(True, alpha=0.3)
                
                examples += 1
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.3, label=n) 
                       for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def comprehensive_evaluation(model, val_loader, model_name="Model"):
    """Full evaluation with detailed metrics."""
    model.eval()
    
    all_seg_metrics = []
    all_denoising_metrics = []
    spike_count = 0
    nan_count = 0
    total_samples = 0
    
    confusion = np.zeros((4, 4), dtype=np.int64)
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            true_mask = batch['mask']
            
            with autocast('cuda'):
                denoised, seg_logits = model(noisy)
            
            pred_mask = seg_logits.argmax(dim=1).cpu().numpy()
            
            for i in range(len(noisy)):
                noisy_np = noisy[i, 0].cpu().numpy()
                clean_np = clean[i, 0].cpu().numpy()
                denoised_np = denoised[i, 0].cpu().numpy()
                true_m = true_mask[i].numpy()
                pred_m = pred_mask[i]
                
                total_samples += 1
                
                # Check for NaN
                if np.isnan(denoised_np).any():
                    nan_count += 1
                    continue
                
                # Spike detection
                spike_info = detect_spikes(denoised_np)
                if spike_info['has_spikes']:
                    spike_count += 1
                
                # Segmentation metrics
                seg_metrics = calculate_segmentation_metrics(pred_m, true_m)
                all_seg_metrics.append(seg_metrics)
                
                # Confusion matrix update (sample subset)
                for true_class in range(4):
                    for pred_class in range(4):
                        confusion[true_class, pred_class] += ((true_m == true_class) & (pred_m == pred_class)).sum()
                
                # Denoising metrics
                try:
                    snr_imp = calculate_snr_improvement(noisy_np, denoised_np, clean_np)
                    ncc = calculate_ncc(denoised_np, clean_np)
                    rmse = calculate_rmse(denoised_np, clean_np)
                    all_denoising_metrics.append({'snr_imp': snr_imp, 'ncc': ncc, 'rmse': rmse})
                except:
                    pass
    
    # Aggregate results
    results = {
        'total_samples': total_samples,
        'nan_samples': nan_count,
        'spike_samples': spike_count,
        'nan_rate': nan_count / total_samples * 100,
        'spike_rate': spike_count / total_samples * 100,
    }
    
    if all_seg_metrics:
        results['seg_accuracy'] = np.mean([m['accuracy'] for m in all_seg_metrics])
        for c in range(4):
            results[f'iou_{c}'] = np.mean([m[f'iou_{c}'] for m in all_seg_metrics])
            results[f'dice_{c}'] = np.mean([m[f'dice_{c}'] for m in all_seg_metrics])
    
    if all_denoising_metrics:
        results['snr_imp'] = np.mean([m['snr_imp'] for m in all_denoising_metrics])
        results['ncc'] = np.mean([m['ncc'] for m in all_denoising_metrics])
        results['rmse'] = np.mean([m['rmse'] for m in all_denoising_metrics])
    
    results['confusion_matrix'] = confusion
    
    return results


def print_results(results, model_name):
    """Print formatted results."""
    print("\n" + "="*70)
    print(f"                 {model_name} EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n--- DATA QUALITY ---")
    print(f"Total samples: {results['total_samples']}")
    print(f"NaN outputs: {results['nan_samples']} ({results['nan_rate']:.2f}%)")
    print(f"Spike outputs: {results['spike_samples']} ({results['spike_rate']:.2f}%)")
    
    print(f"\n--- SEGMENTATION PERFORMANCE ---")
    print(f"Overall Accuracy: {results['seg_accuracy']*100:.2f}%")
    print(f"\nPer-Class IoU:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: IoU={results[f'iou_{c}']*100:5.1f}% | Dice={results[f'dice_{c}']*100:5.1f}%")
    
    print(f"\n--- DENOISING PERFORMANCE ---")
    print(f"SNR Improvement: {results['snr_imp']:.2f} dB")
    print(f"NCC: {results['ncc']:.4f}")
    print(f"RMSE: {results['rmse']:.6f}")
    
    print("="*70)


def plot_confusion_matrix(confusion, save_path):
    """Plot confusion matrix."""
    # Normalize by row (true labels)
    confusion_norm = confusion.astype(float)
    row_sums = confusion_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_norm = confusion_norm / row_sums * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Segmentation Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_signal_generation():
    """Visualize generated signals to verify labeling."""
    config = get_config()
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Signal Generation Verification: Clean Signals with Ground Truth Masks', 
                 fontsize=14, fontweight='bold')
    
    signal_types = ['A', 'B', 'C', 'D', 'E', 'F']
    type_names = ['A (Corona)', 'B (Surface)', 'C (Internal)', 
                  'D (Internal)', 'E (Internal)', 'F (Internal)']
    
    for idx, (sig_type, name) in enumerate(zip(signal_types, type_names)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Generate signal
        t, signal, mask = generate_pd_signal(sig_type, config.signal)
        
        # Plot signal
        ax.plot(signal, 'k-', linewidth=0.8, label='Clean Signal')
        
        # Overlay mask colors
        for c in range(4):
            mask_c = (mask == c)
            if mask_c.any() and c > 0:  # Skip background for clarity
                ax.fill_between(range(len(mask)), 
                               np.min(signal) - 0.1, np.max(signal) + 0.1,
                               where=mask_c, alpha=0.3, 
                               color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        
        # Stats
        unique, counts = np.unique(mask, return_counts=True)
        mask_info = ", ".join([f"{CLASS_NAMES[u]}:{counts[i]}" for i, u in enumerate(unique)])
        
        ax.set_title(f'Type {name}\nMask: {mask_info}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    save_path = SAVE_DIR / 'signal_generation_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("="*60)
    print("Comprehensive Segmentation & Denoising Analysis")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load SOTA model
    model_sota = load_model(SAVE_DIR / 'sota_best.pth')
    
    # Create validation dataset
    config = get_config()
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=500,
        mode='val',
        epoch=50,
        seed=123
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Comprehensive evaluation
    results = comprehensive_evaluation(model_sota, val_loader, "SOTA")
    print_results(results, "SOTA")
    
    # Confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], SAVE_DIR / 'confusion_matrix.png')
    
    # Segmentation visualization
    visualize_segmentation_examples(model_sota, val_loader, 
                                   SAVE_DIR / 'segmentation_analysis.png', num_examples=8)
    
    # Signal generation verification
    visualize_signal_generation()
    
    print(f"\nAll results saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
