#!/usr/bin/env python
"""
train_improved.py - Improved training with class balancing and curriculum learning.

Key improvements:
1. Class-weighted loss for handling extreme imbalance
2. Proper curriculum learning (easy SNR -> hard SNR)
3. Shape preservation loss (NCC-based)
4. Gradient clipping and stability improvements
5. Extended training with proper early stopping
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.training import GeneralizedDiceLoss
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


# =============================================================================
# CONFIGURATION - Optimized for better convergence
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 24           # Slightly smaller for stability
EPOCHS = 150              # Extended training
TRAIN_SAMPLES = 8000      # Larger dataset
VAL_SAMPLES = 1000
LEARNING_RATE = 5e-4      # Lower for stability
USE_AMP = True
SAVE_DIR = Path("outputs/improved_training")
SEED = 42


# =============================================================================
# IMPROVED LOSS FUNCTIONS
# =============================================================================

class CharbonnierLoss(nn.Module):
    """Robust loss for denoising."""
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.eps_sq = epsilon ** 2
    
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class NCCLoss(nn.Module):
    """Normalized Cross-Correlation loss for shape preservation."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Mean center
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        
        # NCC
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1) + 1e-8)
        ncc = numerator / denominator
        
        # Return 1 - NCC so minimizing loss maximizes NCC
        return 1 - ncc.mean()


class ClassWeightedCELoss(nn.Module):
    """Cross-entropy with class weights computed from batch."""
    def __init__(self, num_classes=4, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # Compute class weights from target distribution
        with torch.no_grad():
            class_counts = torch.bincount(target.flatten(), minlength=self.num_classes).float()
            class_counts = class_counts + 1  # Add smoothing
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * self.num_classes
            class_weights = class_weights.to(pred.device)
        
        return F.cross_entropy(pred, target, weight=class_weights, label_smoothing=self.smoothing)


class FocalDiceLoss(nn.Module):
    """Focal loss + Dice loss for class imbalance."""
    def __init__(self, gamma=2.0, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        # Focal Loss
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        
        # Dice Loss (per-class)
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 2, 1).float()
        
        intersection = (pred_soft * target_onehot).sum(dim=2)
        union = pred_soft.sum(dim=2) + target_onehot.sum(dim=2)
        
        # Class weights (inverse frequency)
        class_counts = target_onehot.sum(dim=(0, 2)) + 1
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        
        dice = (2 * intersection + 1) / (union + 1)
        weighted_dice = (dice * weights.unsqueeze(0)).sum(dim=1)
        dice_loss = 1 - weighted_dice.mean()
        
        return focal_loss + dice_loss


class ImprovedMultiTaskLoss(nn.Module):
    """
    Improved multi-task loss with:
    - Charbonnier for robust denoising
    - NCC loss for shape preservation
    - Focal-Dice for class imbalance
    - Learnable uncertainty weighting
    """
    def __init__(self, alpha_denoise=1.0, alpha_ncc=0.5, alpha_seg=0.3):
        super().__init__()
        
        # Fixed weights (can be made learnable)
        self.alpha_denoise = alpha_denoise
        self.alpha_ncc = alpha_ncc
        self.alpha_seg = alpha_seg
        
        # Learnable uncertainty
        self.log_var_denoise = nn.Parameter(torch.tensor(0.0))
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        
        # Loss functions
        self.charbonnier = CharbonnierLoss()
        self.ncc_loss = NCCLoss()
        self.focal_dice = FocalDiceLoss(gamma=2.0)
    
    def forward(self, denoised, clean, seg_pred, seg_target):
        # Denoising losses
        l_charb = self.charbonnier(denoised, clean)
        l_ncc = self.ncc_loss(denoised, clean)
        l_denoise = self.alpha_denoise * l_charb + self.alpha_ncc * l_ncc
        
        # Segmentation loss
        l_seg = self.focal_dice(seg_pred, seg_target)
        
        # Uncertainty weighting
        var_denoise = torch.exp(self.log_var_denoise)
        var_seg = torch.exp(self.log_var_seg)
        
        loss = (l_denoise / (2 * var_denoise) + 0.5 * self.log_var_denoise +
                self.alpha_seg * l_seg / (2 * var_seg) + 0.5 * self.log_var_seg)
        
        return loss, {
            'total': loss.item(),
            'charb': l_charb.item(),
            'ncc': l_ncc.item(),
            'denoise': l_denoise.item(),
            'seg': l_seg.item(),
            'sigma_d': torch.sqrt(var_denoise).item(),
            'sigma_s': torch.sqrt(var_seg).item()
        }


# =============================================================================
# CURRICULUM LEARNING
# =============================================================================

def get_curriculum_snr_range(epoch, total_epochs):
    """Progressive SNR curriculum: easy -> hard."""
    progress = epoch / total_epochs
    
    if progress < 0.2:  # Phase 1: Warm-up (0-20%)
        return (5.0, 10.0)  # Easy: +5 to +10 dB
    elif progress < 0.4:  # Phase 2: Medium (20-40%)
        return (0.0, 5.0)   # Medium: 0 to +5 dB
    elif progress < 0.6:  # Phase 3: Hard (40-60%)
        return (-5.0, 0.0)  # Hard: -5 to 0 dB
    elif progress < 0.8:  # Phase 4: Harder (60-80%)
        return (-10.0, -5.0)  # Harder: -10 to -5 dB
    else:  # Phase 5: Extreme (80-100%)
        return (-20.0, -10.0)  # Extreme: -20 to -10 dB


def get_curriculum_noise_type(epoch, total_epochs):
    """Progressive noise complexity."""
    progress = epoch / total_epochs
    
    if progress < 0.3:
        return 'wgn'
    elif progress < 0.6:
        return 'wgn_impulsive'
    else:
        return 'composite'


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, total_epochs, config):
    """Train for one epoch with curriculum."""
    model.train()
    
    # Update curriculum
    snr_range = get_curriculum_snr_range(epoch, total_epochs)
    noise_type = get_curriculum_noise_type(epoch, total_epochs)
    
    # Update dataset curriculum parameters
    train_loader.dataset.update_epoch(epoch)
    
    total_loss = 0
    metrics = {'charb': 0, 'ncc': 0, 'seg': 0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for batch in pbar:
        noisy = batch['noisy'].to(DEVICE)
        clean = batch['clean'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=USE_AMP):
            denoised, seg_logits = model(noisy)
            loss, loss_dict = criterion(denoised, clean, seg_logits, mask)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k in metrics:
            if k in loss_dict:
                metrics[k] += loss_dict[k]
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ncc': f"{1-loss_dict['ncc']:.3f}",
            'phase': f"SNR{snr_range[1]:.0f}"
        })
    
    n = len(train_loader)
    return total_loss/n, {k: v/n for k, v in metrics.items()}


def validate(model, val_loader, criterion):
    """Validate with comprehensive metrics."""
    model.eval()
    
    total_loss = 0
    all_metrics = {'snr_imp': [], 'ncc': [], 'rmse': []}
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            with autocast('cuda', enabled=USE_AMP):
                denoised, seg_logits = model(noisy)
                loss, _ = criterion(denoised, clean, seg_logits, mask)
            
            total_loss += loss.item()
            
            # Segmentation accuracy
            pred_mask = seg_logits.argmax(dim=1).cpu()
            true_mask = mask.cpu()
            
            for c in range(4):
                mask_c = (true_mask == c)
                class_total[c] += mask_c.sum().item()
                class_correct[c] += ((pred_mask == c) & mask_c).sum().item()
            
            # Denoising metrics
            noisy_np = noisy.squeeze(1).cpu().numpy()
            clean_np = clean.squeeze(1).cpu().numpy()
            denoised_np = denoised.squeeze(1).cpu().numpy()
            
            for i in range(len(noisy_np)):
                try:
                    all_metrics['snr_imp'].append(
                        calculate_snr_improvement(noisy_np[i], denoised_np[i], clean_np[i])
                    )
                    all_metrics['ncc'].append(calculate_ncc(denoised_np[i], clean_np[i]))
                    all_metrics['rmse'].append(calculate_rmse(denoised_np[i], clean_np[i]))
                except:
                    pass
    
    n = len(val_loader)
    
    # Per-class accuracy
    class_acc = [class_correct[c] / max(class_total[c], 1) for c in range(4)]
    overall_acc = sum(class_correct) / max(sum(class_total), 1)
    
    return {
        'loss': total_loss / n,
        'snr_imp': np.mean(all_metrics['snr_imp']) if all_metrics['snr_imp'] else 0,
        'ncc': np.mean(all_metrics['ncc']) if all_metrics['ncc'] else 0,
        'rmse': np.mean(all_metrics['rmse']) if all_metrics['rmse'] else 0,
        'class_acc': class_acc,
        'overall_acc': overall_acc
    }


def visualize_results(model, val_loader, save_path, num_examples=8):
    """Generate comprehensive visualization."""
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 3*num_examples))
    fig.suptitle('Improved Model Results: Noisy | Denoised | Segmentation | Ground Truth', 
                 fontsize=14, fontweight='bold')
    
    CLASS_COLORS = ['lightgray', 'red', 'blue', 'green']
    CLASS_NAMES = ['BG', 'Corona', 'Surface', 'Internal']
    
    examples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if examples >= num_examples:
                break
            
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            true_mask = batch['mask']
            snr = batch['snr']
            sig_type = batch['type']
            
            with autocast('cuda', enabled=USE_AMP):
                denoised, seg_logits = model(noisy)
            
            pred_mask = seg_logits.argmax(dim=1).cpu().numpy()
            
            for i in range(min(len(noisy), num_examples - examples)):
                noisy_np = noisy[i, 0].cpu().numpy()
                clean_np = clean[i, 0].cpu().numpy()
                denoised_np = denoised[i, 0].cpu().numpy()
                true_m = true_mask[i].numpy()
                pred_m = pred_mask[i]
                
                # Metrics
                try:
                    snr_imp = calculate_snr_improvement(noisy_np, denoised_np, clean_np)
                    ncc = calculate_ncc(denoised_np, clean_np)
                except:
                    snr_imp, ncc = 0, 0
                
                # Noisy
                axes[examples, 0].plot(noisy_np, 'b-', alpha=0.7, linewidth=0.5)
                axes[examples, 0].set_ylabel(f'SNR:{snr[i].item():.0f}dB')
                axes[examples, 0].set_title(f'Type {sig_type[i].item()}' if examples == 0 else '')
                axes[examples, 0].grid(True, alpha=0.3)
                
                # Denoised
                axes[examples, 1].plot(denoised_np, 'r-', linewidth=0.8)
                axes[examples, 1].plot(clean_np, 'g--', alpha=0.5, linewidth=0.5)
                axes[examples, 1].set_title(f'SNR↑:{snr_imp:.1f}dB NCC:{ncc:.3f}' if examples == 0 else f'{snr_imp:.1f}dB/{ncc:.3f}')
                axes[examples, 1].grid(True, alpha=0.3)
                
                # Predicted Segmentation
                for c in range(4):
                    mask_c = (pred_m == c)
                    if mask_c.any() and c > 0:
                        axes[examples, 2].fill_between(range(len(pred_m)), -1, 1,
                                                       where=mask_c, alpha=0.4, color=CLASS_COLORS[c])
                axes[examples, 2].plot(denoised_np, 'k-', linewidth=0.5, alpha=0.7)
                axes[examples, 2].set_ylim(-1.2, 1.2)
                axes[examples, 2].grid(True, alpha=0.3)
                
                # Ground Truth
                for c in range(4):
                    mask_c = (true_m == c)
                    if mask_c.any() and c > 0:
                        axes[examples, 3].fill_between(range(len(true_m)), -1, 1,
                                                       where=mask_c, alpha=0.4, color=CLASS_COLORS[c])
                axes[examples, 3].plot(clean_np, 'k-', linewidth=0.5, alpha=0.7)
                axes[examples, 3].set_ylim(-1.2, 1.2)
                axes[examples, 3].grid(True, alpha=0.3)
                
                examples += 1
    
    # Column titles
    for ax, title in zip(axes[0], ['Noisy Input', 'Denoised (red) vs Clean (green)', 
                                    'Predicted Mask', 'Ground Truth Mask']):
        ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main training pipeline with improvements."""
    start_time = datetime.now()
    
    print("="*70)
    print("MR-TAE-Fusion IMPROVED Training")
    print("="*70)
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train Samples: {TRAIN_SAMPLES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print()
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Model
    config = get_config()
    model = create_model(config.model).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Improved loss
    criterion = ImprovedMultiTaskLoss(
        alpha_denoise=1.0,
        alpha_ncc=0.5,  # Weight for shape preservation
        alpha_seg=0.3   # Lower initial weight for segmentation
    ).to(DEVICE)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Scheduler: Warm-up + Cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_SAMPLES // BATCH_SIZE,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Datasets
    train_dataset = PDSignalDataset(
        config=config,
        num_samples=TRAIN_SAMPLES,
        mode='train',
        seed=SEED,
        total_epochs=EPOCHS
    )
    
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=VAL_SAMPLES,
        mode='val',
        epoch=EPOCHS-1,  # Validate at hardest difficulty
        seed=SEED+1,
        total_epochs=EPOCHS
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    scaler = GradScaler('cuda')
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'snr_imp': [], 'ncc': [], 'class_acc': [], 'overall_acc': []
    }
    
    best_score = -float('inf')
    patience = 20
    patience_counter = 0
    
    print("\nStarting training with curriculum learning...\n")
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, EPOCHS, config
        )
        
        # Step scheduler (per-batch, but we approximate)
        for _ in range(len(train_loader)):
            scheduler.step()
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_metrics = validate(model, val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['snr_imp'].append(val_metrics['snr_imp'])
            history['ncc'].append(val_metrics['ncc'])
            history['class_acc'].append(val_metrics['class_acc'])
            history['overall_acc'].append(val_metrics['overall_acc'])
            
            curriculum_phase = get_curriculum_snr_range(epoch, EPOCHS)
            
            print(f"\nEpoch {epoch+1}/{EPOCHS} [Phase: SNR {curriculum_phase[0]:.0f} to {curriculum_phase[1]:.0f} dB]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  SNR Improvement: {val_metrics['snr_imp']:.2f} dB")
            print(f"  NCC: {val_metrics['ncc']:.4f}")
            print(f"  Seg Accuracy: BG={val_metrics['class_acc'][0]*100:.1f}% | "
                  f"Corona={val_metrics['class_acc'][1]*100:.1f}% | "
                  f"Surface={val_metrics['class_acc'][2]*100:.1f}% | "
                  f"Internal={val_metrics['class_acc'][3]*100:.1f}%")
            print(f"  Overall: {val_metrics['overall_acc']*100:.1f}%")
            
            # Save best model (weighted score)
            score = val_metrics['ncc'] + 0.3 * val_metrics['overall_acc']
            if score > best_score:
                best_score = score
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    **val_metrics
                }, SAVE_DIR / 'best_model.pth')
                print(f"  >> New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            visualize_results(model, val_loader, 
                             SAVE_DIR / f'results_epoch_{epoch+1}.png')
    
    # Load best and final visualization
    checkpoint = torch.load(SAVE_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    visualize_results(model, val_loader, SAVE_DIR / 'final_results.png', num_examples=10)
    
    # Save history
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Time: {elapsed}")
    print(f"Best NCC: {checkpoint['ncc']:.4f}")
    print(f"Best SNR Improvement: {checkpoint['snr_imp']:.2f} dB")
    print(f"Best Segmentation: {checkpoint['overall_acc']*100:.1f}%")
    print(f"\nSaved to: {SAVE_DIR}")
    
    # Save summary
    with open(SAVE_DIR / 'summary.json', 'w') as f:
        json.dump({
            'best_ncc': float(checkpoint['ncc']),
            'best_snr_imp': float(checkpoint['snr_imp']),
            'best_overall_acc': float(checkpoint['overall_acc']),
            'class_acc': [float(x) for x in checkpoint['class_acc']],
            'epochs': EPOCHS,
            'elapsed': str(elapsed)
        }, f, indent=2)


if __name__ == '__main__':
    main()
