#!/usr/bin/env python
"""
train_extended.py - Extended training for MR-TAE-Fusion with CUDA.

Optimized for RTX 4070 with extended training time (2-4 hours).
Trains both baseline (MSE) and SOTA (Charbonnier) versions for comparison.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json

from mr_tae_fusion.config import Config, get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.training import JointLoss, JointLossWithUncertainty, MultiTaskLoss
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


# =============================================================================
# CONFIGURATION - Extended Training for RTX 4070
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32           # RTX 4070 can handle this with AMP
EPOCHS = 100              # Extended training (~2-3 hours)
TRAIN_SAMPLES = 6000      # Larger dataset
VAL_SAMPLES = 1000        # More validation samples
LEARNING_RATE = 2e-4
USE_AMP = True            # Mixed precision
SAVE_DIR = Path("outputs/extended_training")
SEED = 42


def create_dataloaders(config, train_samples, val_samples, batch_size, seed):
    """Create data loaders."""
    train_dataset = PDSignalDataset(
        config=config,
        num_samples=train_samples,
        mode='train',
        seed=seed
    )
    
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=val_samples,
        mode='val',
        seed=seed + 1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset


def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, total_epochs, use_amp=True):
    """Train for one epoch."""
    model.train()
    train_loader.dataset.update_epoch(epoch)
    
    total_loss = 0
    total_denoise_loss = 0
    total_seg_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for batch in pbar:
        noisy = batch['noisy'].to(DEVICE)
        clean = batch['clean'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        
        optimizer.zero_grad()
        
        if use_amp and DEVICE == 'cuda':
            with autocast():
                denoised, seg_logits = model(noisy)
                loss, loss_dict = criterion(denoised, clean, seg_logits, mask)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            denoised, seg_logits = model(noisy)
            loss, loss_dict = criterion(denoised, clean, seg_logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Handle different loss_dict formats
        if isinstance(loss_dict, dict):
            if 'denoise' in loss_dict:
                total_denoise_loss += loss_dict['denoise'] if isinstance(loss_dict['denoise'], float) else loss_dict['denoise'].item()
            elif 'recon' in loss_dict:
                total_denoise_loss += loss_dict['recon'].item() if hasattr(loss_dict['recon'], 'item') else loss_dict['recon']
            
            if 'seg' in loss_dict:
                total_seg_loss += loss_dict['seg'] if isinstance(loss_dict['seg'], float) else loss_dict['seg'].item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
        })
    
    n = len(train_loader)
    return total_loss/n, total_denoise_loss/n, total_seg_loss/n


def validate(model, val_loader, criterion, use_amp=True):
    """Validate model and calculate metrics."""
    model.eval()
    
    total_loss = 0
    snr_improvements = []
    nccs = []
    rmses = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            if use_amp and DEVICE == 'cuda':
                with autocast():
                    denoised, seg_logits = model(noisy)
                    loss, _ = criterion(denoised, clean, seg_logits, mask)
            else:
                denoised, seg_logits = model(noisy)
                loss, _ = criterion(denoised, clean, seg_logits, mask)
            
            total_loss += loss.item()
            
            # Calculate metrics
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
                except:
                    pass
    
    n = len(val_loader)
    return {
        'loss': total_loss / n,
        'snr_imp': np.mean(snr_improvements) if snr_improvements else 0,
        'ncc': np.mean(nccs) if nccs else 0,
        'rmse': np.mean(rmses) if rmses else 0
    }


def visualize_comparison(model_baseline, model_sota, val_loader, save_path, num_examples=6):
    """Compare baseline vs SOTA denoising results."""
    model_baseline.eval()
    model_sota.eval()
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 3*num_examples))
    fig.suptitle('Comparison: Noisy → Baseline → SOTA → Ground Truth', fontsize=14, fontweight='bold')
    
    columns = ['Noisy Input', 'Baseline (MSE)', 'SOTA (Charbonnier)', 'Ground Truth']
    for ax, col in zip(axes[0], columns):
        ax.set_title(col, fontweight='bold')
    
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
                
                # Noisy
                axes[idx, 0].plot(noisy_np[i], 'b-', alpha=0.7, linewidth=0.5)
                axes[idx, 0].set_ylabel(f'SNR: {snr[i].item():.0f}dB')
                axes[idx, 0].grid(True, alpha=0.3)
                
                # Baseline
                axes[idx, 1].plot(baseline_np[i], 'orange', linewidth=0.8)
                axes[idx, 1].set_title(f'↑{snr_imp_base:.1f}dB, NCC:{ncc_base:.3f}', fontsize=9)
                axes[idx, 1].grid(True, alpha=0.3)
                
                # SOTA
                axes[idx, 2].plot(sota_np[i], 'r-', linewidth=0.8)
                axes[idx, 2].set_title(f'↑{snr_imp_sota:.1f}dB, NCC:{ncc_sota:.3f}', fontsize=9)
                axes[idx, 2].grid(True, alpha=0.3)
                
                # Ground Truth
                axes[idx, 3].plot(clean_np[i], 'g-', linewidth=0.8)
                axes[idx, 3].grid(True, alpha=0.3)
                
                examples_collected += 1
    
    for ax in axes[-1]:
        ax.set_xlabel('Sample')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved: {save_path}")


def train_model(model_name, criterion, epochs=EPOCHS):
    """Train a single model version."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print('='*60)
    
    config = get_config()
    model = create_model(config.model).to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    
    train_loader, val_loader, train_dataset = create_dataloaders(
        config, TRAIN_SAMPLES, VAL_SAMPLES, BATCH_SIZE, SEED
    )
    
    scaler = GradScaler() if USE_AMP else None
    
    history = {'train_loss': [], 'val_loss': [], 'snr_imp': [], 'ncc': [], 'rmse': []}
    best_snr_imp = -float('inf')
    
    for epoch in range(epochs):
        train_loss, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, epochs
        )
        
        val_metrics = validate(model, val_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['snr_imp'].append(val_metrics['snr_imp'])
        history['ncc'].append(val_metrics['ncc'])
        history['rmse'].append(val_metrics['rmse'])
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Val: {val_metrics['loss']:.4f} | SNR↑: {val_metrics['snr_imp']:.2f}dB | "
              f"NCC: {val_metrics['ncc']:.4f}")
        
        if val_metrics['snr_imp'] > best_snr_imp:
            best_snr_imp = val_metrics['snr_imp']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'snr_imp': best_snr_imp,
                'ncc': val_metrics['ncc']
            }, SAVE_DIR / f'{model_name}_best.pth')
    
    return model, history, best_snr_imp


def main():
    """Main training pipeline."""
    start_time = datetime.now()
    
    print("="*60)
    print("MR-TAE-Fusion Extended Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train Samples: {TRAIN_SAMPLES}")
    print(f"Val Samples: {VAL_SAMPLES}")
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Train Baseline (MSE-based loss)
    baseline_criterion = MultiTaskLoss().to(DEVICE)
    model_baseline, hist_baseline, best_baseline = train_model(
        "baseline", baseline_criterion, epochs=EPOCHS
    )
    
    # Train SOTA (Charbonnier + Focal)
    sota_criterion = JointLoss(alpha=1.0, beta=0.5, use_focal=True).to(DEVICE)
    model_sota, hist_sota, best_sota = train_model(
        "sota", sota_criterion, epochs=EPOCHS
    )
    
    # Load best models
    checkpoint = torch.load(SAVE_DIR / 'baseline_best.pth')
    model_baseline.load_state_dict(checkpoint['model_state_dict'])
    
    checkpoint = torch.load(SAVE_DIR / 'sota_best.pth')
    model_sota.load_state_dict(checkpoint['model_state_dict'])
    
    # Create comparison visualizations
    config = get_config()
    _, val_loader, _ = create_dataloaders(config, 100, 500, BATCH_SIZE, SEED+2)
    
    visualize_comparison(
        model_baseline, model_sota, val_loader,
        SAVE_DIR / 'comparison_results.png', num_examples=8
    )
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = range(1, EPOCHS + 1)
    
    axes[0, 0].plot(epochs_range, hist_baseline['snr_imp'], 'b-', label='Baseline')
    axes[0, 0].plot(epochs_range, hist_sota['snr_imp'], 'r-', label='SOTA')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('SNR Improvement (dB)')
    axes[0, 0].set_title('SNR Improvement'); axes[0, 0].legend(); axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs_range, hist_baseline['ncc'], 'b-', label='Baseline')
    axes[0, 1].plot(epochs_range, hist_sota['ncc'], 'r-', label='SOTA')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('NCC')
    axes[0, 1].set_title('Shape Fidelity (NCC)'); axes[0, 1].legend(); axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs_range, hist_baseline['train_loss'], 'b-', label='Baseline')
    axes[1, 0].plot(epochs_range, hist_sota['train_loss'], 'r-', label='SOTA')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss'); axes[1, 0].legend(); axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs_range, hist_baseline['rmse'], 'b-', label='Baseline')
    axes[1, 1].plot(epochs_range, hist_sota['rmse'], 'r-', label='SOTA')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('RMSE'); axes[1, 1].legend(); axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'training_comparison.png', dpi=150)
    plt.close()
    
    # Summary
    elapsed = datetime.now() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Time elapsed: {elapsed}")
    print(f"\nBaseline (MSE) - Best SNR Improvement: {best_baseline:.2f} dB")
    print(f"SOTA (Charbonnier) - Best SNR Improvement: {best_sota:.2f} dB")
    print(f"Improvement: {best_sota - best_baseline:.2f} dB")
    print(f"\nOutputs saved to: {SAVE_DIR}")
    
    # Save summary
    summary = {
        'baseline_snr_imp': float(best_baseline),
        'sota_snr_imp': float(best_sota),
        'improvement': float(best_sota - best_baseline),
        'epochs': EPOCHS,
        'train_samples': TRAIN_SAMPLES,
        'elapsed_time': str(elapsed)
    }
    with open(SAVE_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
