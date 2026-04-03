#!/usr/bin/env python
"""
train_and_visualize.py - Train MR-TAE-Fusion and visualize results.

Optimized for RTX 4070 (8GB VRAM). Should complete in 1-2 hours.
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

from mr_tae_fusion.config import Config, get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.training import JointLoss, JointLossWithUncertainty
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


# =============================================================================
# CONFIGURATION - Optimized for RTX 4070 (8GB VRAM)
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16           # Safe for 8GB VRAM
EPOCHS = 30               # ~1-2 hours on RTX 4070
TRAIN_SAMPLES = 3000      # Training samples per epoch  
VAL_SAMPLES = 500         # Validation samples
LEARNING_RATE = 1e-3
USE_AMP = True            # Mixed precision for speed + memory
SAVE_DIR = Path("outputs/training_run")
SEED = 42


def setup_training():
    """Setup model, optimizer, and data loaders."""
    print("="*60)
    print("MR-TAE-Fusion Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train Samples: {TRAIN_SAMPLES}")
    print(f"Mixed Precision: {USE_AMP}")
    print()
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create output directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = get_config()
    
    # Create model
    print("Creating model...")
    model = create_model(config.model).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function (JointLoss with Charbonnier for robustness)
    criterion = JointLoss(alpha=1.0, beta=0.5, dice_weight=1.0).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Data loaders
    print("Creating datasets...")
    train_dataset = PDSignalDataset(
        config=config,
        num_samples=TRAIN_SAMPLES,
        mode='train',
        seed=SEED
    )
    
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=VAL_SAMPLES,
        mode='val',
        seed=SEED + 1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return model, criterion, optimizer, scheduler, train_loader, val_loader, train_dataset, config


def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, total_epochs):
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
        
        if USE_AMP:
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
        total_denoise_loss += loss_dict['denoise']
        total_seg_loss += loss_dict['seg']
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'denoise': f"{loss_dict['denoise']:.4f}"
        })
    
    n = len(train_loader)
    return total_loss/n, total_denoise_loss/n, total_seg_loss/n


def validate(model, val_loader, criterion):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    snr_improvements = []
    nccs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            if USE_AMP:
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
                    snr_improvements.append(snr_imp)
                    nccs.append(ncc)
                except:
                    pass
    
    n = len(val_loader)
    avg_snr_imp = np.mean(snr_improvements) if snr_improvements else 0
    avg_ncc = np.mean(nccs) if nccs else 0
    
    return total_loss/n, avg_snr_imp, avg_ncc


def visualize_results(model, val_loader, save_path, num_examples=6):
    """Generate visualization of denoising results."""
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 3*num_examples))
    fig.suptitle('MR-TAE-Fusion Denoising Results', fontsize=14, fontweight='bold')
    
    examples_collected = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if examples_collected >= num_examples:
                break
            
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            snr = batch['snr']
            
            if USE_AMP:
                with autocast():
                    denoised, _ = model(noisy)
            else:
                denoised, _ = model(noisy)
            
            noisy_np = noisy.squeeze(1).cpu().numpy()
            clean_np = clean.squeeze(1).cpu().numpy()
            denoised_np = denoised.squeeze(1).cpu().numpy()
            
            for i in range(min(len(noisy_np), num_examples - examples_collected)):
                idx = examples_collected
                
                # Calculate metrics for this sample
                try:
                    snr_imp = calculate_snr_improvement(noisy_np[i], denoised_np[i], clean_np[i])
                    ncc = calculate_ncc(denoised_np[i], clean_np[i])
                except:
                    snr_imp = 0
                    ncc = 0
                
                # Noisy signal
                axes[idx, 0].plot(noisy_np[i], 'b-', alpha=0.7, linewidth=0.5)
                axes[idx, 0].set_title(f'Noisy (SNR: {snr[i].item():.1f} dB)')
                axes[idx, 0].set_ylabel(f'Sample {idx+1}')
                axes[idx, 0].grid(True, alpha=0.3)
                
                # Clean (Ground Truth)
                axes[idx, 1].plot(clean_np[i], 'g-', linewidth=0.8)
                axes[idx, 1].set_title('Clean (Ground Truth)')
                axes[idx, 1].grid(True, alpha=0.3)
                
                # Denoised
                axes[idx, 2].plot(denoised_np[i], 'r-', linewidth=0.8)
                axes[idx, 2].set_title(f'Denoised (SNR Imp: {snr_imp:.1f} dB, NCC: {ncc:.3f})')
                axes[idx, 2].grid(True, alpha=0.3)
                
                examples_collected += 1
    
    # Add common labels
    for ax in axes[-1]:
        ax.set_xlabel('Sample')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {save_path}")


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Denoise Loss
    axes[0, 1].plot(epochs, history['train_denoise'], 'b-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Denoise Loss')
    axes[0, 1].set_title('Denoising Loss (Charbonnier)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SNR Improvement
    axes[1, 0].plot(epochs, history['val_snr_imp'], 'g-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SNR Improvement (dB)')
    axes[1, 0].set_title('Validation SNR Improvement')
    axes[1, 0].grid(True, alpha=0.3)
    
    # NCC
    axes[1, 1].plot(epochs, history['val_ncc'], 'm-')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('NCC')
    axes[1, 1].set_title('Validation NCC (Shape Fidelity)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved: {save_path}")


def main():
    """Main training loop."""
    start_time = datetime.now()
    
    # Setup
    model, criterion, optimizer, scheduler, train_loader, val_loader, train_dataset, config = setup_training()
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_AMP else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_denoise': [],
        'train_seg': [],
        'val_loss': [],
        'val_snr_imp': [],
        'val_ncc': []
    }
    
    best_snr_imp = -float('inf')
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_denoise, train_seg = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, EPOCHS
        )
        
        # Validate
        val_loss, val_snr_imp, val_ncc = validate(model, val_loader, criterion)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_denoise'].append(train_denoise)
        history['train_seg'].append(train_seg)
        history['val_loss'].append(val_loss)
        history['val_snr_imp'].append(val_snr_imp)
        history['val_ncc'].append(val_ncc)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Denoise: {train_denoise:.4f} | Seg: {train_seg:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | SNR Imp: {val_snr_imp:.2f} dB | NCC: {val_ncc:.4f}")
        
        # Save best model
        if val_snr_imp > best_snr_imp:
            best_snr_imp = val_snr_imp
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_snr_imp': val_snr_imp,
                'val_ncc': val_ncc,
            }, SAVE_DIR / 'best_model.pth')
            print(f"  >> New best model saved! SNR Imp: {best_snr_imp:.2f} dB")
        
        # Generate visualizations every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            visualize_results(
                model, val_loader, 
                SAVE_DIR / f'results_epoch_{epoch+1}.png'
            )
    
    # Final visualizations
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Load best model for final visualization
    checkpoint = torch.load(SAVE_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final results visualization
    visualize_results(model, val_loader, SAVE_DIR / 'final_results.png', num_examples=8)
    
    # Training history plot
    plot_training_history(history, SAVE_DIR / 'training_history.png')
    
    # Summary
    elapsed = datetime.now() - start_time
    print(f"\nTotal training time: {elapsed}")
    print(f"Best SNR Improvement: {best_snr_imp:.2f} dB")
    print(f"Best NCC: {checkpoint['val_ncc']:.4f}")
    print(f"\nOutputs saved to: {SAVE_DIR}")
    print(f"  - best_model.pth")
    print(f"  - final_results.png")
    print(f"  - training_history.png")
    
    return model, history


if __name__ == '__main__':
    main()
