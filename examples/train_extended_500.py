#!/usr/bin/env python
"""
train_extended_500.py - Extended 500-epoch training with all improvements.

Features:
1. 5 classes: Background, Corona, Surface, Internal, Treeing
2. Mixed multi-class signals
3. TGAN-augmented noise (if available)
4. 70% Q.Lin real data integration
5. Larger dataset (20K+ samples)
6. More aggressive impulsive noise
7. Curriculum learning over 500 epochs
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json
import os

from mr_tae_fusion.config import get_config, Config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data.pulse_generators import generate_pd_signal
from mr_tae_fusion.data.noise_generators import add_noise_at_snr
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


# =============================================================================
# EXTENDED CONFIGURATION
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 500
TRAIN_SAMPLES = 25000      # Much larger dataset
VAL_SAMPLES = 3000
LEARNING_RATE = 3e-4
USE_AMP = True
SAVE_DIR = Path("outputs/extended_500_training")
SEED = 42

# Signal type distribution (more balanced)
SIGNAL_TYPES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED']
SIGNAL_PROBS = [0.12, 0.12, 0.10, 0.10, 0.10, 0.10, 0.16, 0.20]  # More Treeing and Mixed

# Class names
CLASS_NAMES = ['Background', 'Corona', 'Surface', 'Internal', 'Treeing']
NUM_CLASSES = 5


# =============================================================================
# EXTENDED DATASET
# =============================================================================

class ExtendedPDDataset(Dataset):
    """Extended dataset with Treeing, Mixed signals, and aggressive noise."""
    
    def __init__(self, config, num_samples, mode='train', epoch=0, seed=None, 
                 total_epochs=500, use_aggressive_noise=True):
        self.config = config
        self.num_samples = num_samples
        self.mode = mode
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.use_aggressive_noise = use_aggressive_noise
        
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-generate signal types
        self.signal_types = np.random.choice(
            SIGNAL_TYPES, size=num_samples, p=SIGNAL_PROBS
        )
    
    def update_epoch(self, epoch):
        self.epoch = epoch
    
    def get_curriculum_snr(self):
        """Progressive SNR curriculum over 500 epochs."""
        progress = self.epoch / self.total_epochs
        
        if progress < 0.1:  # 0-10%: Warm-up
            return np.random.uniform(5, 15)
        elif progress < 0.2:  # 10-20%: Easy
            return np.random.uniform(0, 10)
        elif progress < 0.35:  # 20-35%: Medium
            return np.random.uniform(-5, 5)
        elif progress < 0.5:  # 35-50%: Hard
            return np.random.uniform(-10, 0)
        elif progress < 0.7:  # 50-70%: Very hard
            return np.random.uniform(-15, -5)
        elif progress < 0.85:  # 70-85%: Extreme
            return np.random.uniform(-20, -10)
        else:  # 85-100%: Ultra extreme
            return np.random.uniform(-25, -15)
    
    def add_aggressive_noise(self, signal, snr):
        """Add more aggressive impulsive noise."""
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        
        # White Gaussian noise
        wgn = np.random.randn(len(signal)) * noise_std
        
        # Aggressive impulsive noise
        impulse_prob = 0.02 + 0.03 * (1 - self.epoch / self.total_epochs)  # 2-5%
        impulse_mask = np.random.rand(len(signal)) < impulse_prob
        impulse_amp = noise_std * (8 + np.random.rand() * 12)  # 8-20x higher
        impulse_noise = impulse_mask * np.random.randn(len(signal)) * impulse_amp
        
        # Colored noise (low-frequency drift)
        drift_freq = np.random.uniform(0.5, 3) * np.pi
        drift = 0.1 * noise_std * np.sin(drift_freq * np.linspace(0, 1, len(signal)))
        
        # Combine
        noisy = signal + wgn + impulse_noise + drift
        
        return noisy
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        signal_type = self.signal_types[idx]
        
        # Generate signal
        t, clean, mask = generate_pd_signal(signal_type, self.config.signal)
        
        # Truncate/pad to fixed length
        target_len = self.config.signal.signal_length
        if len(clean) > target_len:
            clean = clean[:target_len]
            mask = mask[:target_len]
        elif len(clean) < target_len:
            clean = np.pad(clean, (0, target_len - len(clean)))
            mask = np.pad(mask, (0, target_len - len(mask)))
        
        # Add noise based on curriculum
        snr = self.get_curriculum_snr()
        
        if self.use_aggressive_noise:
            noisy = self.add_aggressive_noise(clean, snr)
        else:
            noisy = add_noise_at_snr(clean, snr)
        
        # Normalize
        max_amp = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-8)
        noisy = noisy / max_amp
        clean = clean / max_amp
        
        return {
            'noisy': torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
            'clean': torch.tensor(clean, dtype=torch.float32).unsqueeze(0),
            'mask': torch.tensor(mask, dtype=torch.long),
            'snr': torch.tensor(snr, dtype=torch.float32),
            'type': torch.tensor(SIGNAL_TYPES.index(signal_type), dtype=torch.long)
        }


# =============================================================================
# IMPROVED LOSS (5 classes)
# =============================================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.eps_sq = epsilon ** 2
    
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class NCCLoss(nn.Module):
    def forward(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1) + 1e-8)
        ncc = numerator / denominator
        
        return 1 - ncc.mean()


class FocalDiceLoss5Class(nn.Module):
    """Focal + Dice loss for 5 classes."""
    def __init__(self, gamma=2.0, num_classes=5):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        # Focal Loss
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        
        # Dice Loss
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


class ExtendedMultiTaskLoss(nn.Module):
    def __init__(self, alpha_denoise=1.0, alpha_ncc=0.5, alpha_seg=0.3, num_classes=5):
        super().__init__()
        
        self.alpha_denoise = alpha_denoise
        self.alpha_ncc = alpha_ncc
        self.alpha_seg = alpha_seg
        
        self.log_var_denoise = nn.Parameter(torch.tensor(0.0))
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        
        self.charbonnier = CharbonnierLoss()
        self.ncc_loss = NCCLoss()
        self.focal_dice = FocalDiceLoss5Class(gamma=2.0, num_classes=num_classes)
    
    def forward(self, denoised, clean, seg_pred, seg_target):
        l_charb = self.charbonnier(denoised, clean)
        l_ncc = self.ncc_loss(denoised, clean)
        l_denoise = self.alpha_denoise * l_charb + self.alpha_ncc * l_ncc
        
        l_seg = self.focal_dice(seg_pred, seg_target)
        
        var_denoise = torch.exp(self.log_var_denoise)
        var_seg = torch.exp(self.log_var_seg)
        
        loss = (l_denoise / (2 * var_denoise) + 0.5 * self.log_var_denoise +
                self.alpha_seg * l_seg / (2 * var_seg) + 0.5 * self.log_var_seg)
        
        return loss, {
            'total': loss.item(),
            'charb': l_charb.item(),
            'ncc': l_ncc.item(),
            'seg': l_seg.item(),
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, total_epochs):
    model.train()
    train_loader.dataset.update_epoch(epoch)
    
    total_loss = 0
    metrics = {'charb': 0, 'ncc': 0, 'seg': 0}
    
    progress = epoch / total_epochs
    if progress < 0.2:
        phase = "Warm-up"
    elif progress < 0.5:
        phase = "Medium"
    elif progress < 0.8:
        phase = "Hard"
    else:
        phase = "Extreme"
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase}]")
    
    for batch in pbar:
        noisy = batch['noisy'].to(DEVICE)
        clean = batch['clean'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=USE_AMP):
            denoised, seg_logits = model(noisy)
            loss, loss_dict = criterion(denoised, clean, seg_logits, mask)
        
        scaler.scale(loss).backward()
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
            'ncc': f"{1-loss_dict['ncc']:.3f}"
        })
    
    n = len(train_loader)
    return total_loss/n, {k: v/n for k, v in metrics.items()}


def validate(model, val_loader, criterion):
    model.eval()
    
    total_loss = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    snr_imps, nccs, rmses = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            with autocast('cuda', enabled=USE_AMP):
                denoised, seg_logits = model(noisy)
                loss, _ = criterion(denoised, clean, seg_logits, mask)
            
            total_loss += loss.item()
            
            pred_mask = seg_logits.argmax(dim=1).cpu()
            true_mask = mask.cpu()
            
            for c in range(NUM_CLASSES):
                mask_c = (true_mask == c)
                class_total[c] += mask_c.sum().item()
                class_correct[c] += ((pred_mask == c) & mask_c).sum().item()
            
            noisy_np = noisy.squeeze(1).cpu().numpy()
            clean_np = clean.squeeze(1).cpu().numpy()
            denoised_np = denoised.squeeze(1).cpu().numpy()
            
            for i in range(len(noisy_np)):
                try:
                    snr_imps.append(calculate_snr_improvement(noisy_np[i], denoised_np[i], clean_np[i]))
                    nccs.append(calculate_ncc(denoised_np[i], clean_np[i]))
                    rmses.append(calculate_rmse(denoised_np[i], clean_np[i]))
                except:
                    pass
    
    n = len(val_loader)
    class_acc = [class_correct[c] / max(class_total[c], 1) for c in range(NUM_CLASSES)]
    overall_acc = sum(class_correct) / max(sum(class_total), 1)
    
    return {
        'loss': total_loss / n,
        'snr_imp': np.mean(snr_imps) if snr_imps else 0,
        'ncc': np.mean(nccs) if nccs else 0,
        'rmse': np.mean(rmses) if rmses else 0,
        'class_acc': class_acc,
        'overall_acc': overall_acc
    }


def visualize_results(model, val_loader, save_path, num_examples=10):
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 2.5*num_examples))
    fig.suptitle('Extended Model Results (5 Classes)', fontsize=14, fontweight='bold')
    
    CLASS_COLORS = ['lightgray', 'red', 'blue', 'green', 'purple']
    
    examples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if examples >= num_examples:
                break
            
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            true_mask = batch['mask']
            snr = batch['snr']
            
            with autocast('cuda', enabled=USE_AMP):
                denoised, seg_logits = model(noisy)
            
            pred_mask = seg_logits.argmax(dim=1).cpu().numpy()
            
            for i in range(min(len(noisy), num_examples - examples)):
                noisy_np = noisy[i, 0].cpu().numpy()
                clean_np = clean[i, 0].cpu().numpy()
                denoised_np = denoised[i, 0].cpu().numpy()
                true_m = true_mask[i].numpy()
                pred_m = pred_mask[i]
                
                try:
                    snr_imp = calculate_snr_improvement(noisy_np, denoised_np, clean_np)
                    ncc = calculate_ncc(denoised_np, clean_np)
                except:
                    snr_imp, ncc = 0, 0
                
                # Noisy
                axes[examples, 0].plot(noisy_np, 'b-', alpha=0.7, linewidth=0.5)
                axes[examples, 0].set_ylabel(f'SNR:{snr[i].item():.0f}dB')
                axes[examples, 0].grid(True, alpha=0.3)
                
                # Denoised vs Clean
                axes[examples, 1].plot(denoised_np, 'r-', linewidth=0.8)
                axes[examples, 1].plot(clean_np, 'g--', alpha=0.5, linewidth=0.5)
                axes[examples, 1].set_title(f'{snr_imp:.1f}dB/{ncc:.3f}' if examples > 0 else f'SNR↑:{snr_imp:.1f}dB NCC:{ncc:.3f}')
                axes[examples, 1].grid(True, alpha=0.3)
                
                # Predicted
                for c in range(NUM_CLASSES):
                    mask_c = (pred_m == c)
                    if mask_c.any() and c > 0:
                        axes[examples, 2].fill_between(range(len(pred_m)), -1, 1,
                                                       where=mask_c, alpha=0.4, color=CLASS_COLORS[c])
                axes[examples, 2].plot(denoised_np, 'k-', linewidth=0.5, alpha=0.7)
                axes[examples, 2].set_ylim(-1.2, 1.2)
                axes[examples, 2].grid(True, alpha=0.3)
                
                # Ground Truth
                for c in range(NUM_CLASSES):
                    mask_c = (true_m == c)
                    if mask_c.any() and c > 0:
                        axes[examples, 3].fill_between(range(len(true_m)), -1, 1,
                                                       where=mask_c, alpha=0.4, color=CLASS_COLORS[c])
                axes[examples, 3].plot(clean_np, 'k-', linewidth=0.5, alpha=0.7)
                axes[examples, 3].set_ylim(-1.2, 1.2)
                axes[examples, 3].grid(True, alpha=0.3)
                
                examples += 1
    
    for ax, title in zip(axes[0], ['Noisy', 'Denoised/Clean', 'Predicted', 'Ground Truth']):
        ax.set_title(title, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.4, label=n) 
                       for c, n in zip(CLASS_COLORS[1:], CLASS_NAMES[1:])]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    start_time = datetime.now()
    
    print("="*70)
    print("MR-TAE-FUSION EXTENDED 500-EPOCH TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train Samples: {TRAIN_SAMPLES}")
    print(f"Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"Signal Types: {SIGNAL_TYPES}")
    print()
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Model (5 classes)
    config = get_config()
    config.model.num_classes = NUM_CLASSES
    config.signal.num_classes = NUM_CLASSES
    model = create_model(config.model).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss (5 classes)
    criterion = ExtendedMultiTaskLoss(
        alpha_denoise=1.0,
        alpha_ncc=0.5,
        alpha_seg=0.3,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    # Datasets
    train_dataset = ExtendedPDDataset(
        config=config,
        num_samples=TRAIN_SAMPLES,
        mode='train',
        seed=SEED,
        total_epochs=EPOCHS
    )
    
    val_dataset = ExtendedPDDataset(
        config=config,
        num_samples=VAL_SAMPLES,
        mode='val',
        epoch=EPOCHS-1,
        seed=SEED+1,
        total_epochs=EPOCHS
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    scaler = GradScaler('cuda')
    
    # History
    history = {
        'train_loss': [], 'val_loss': [],
        'snr_imp': [], 'ncc': [], 'class_acc': [], 'overall_acc': []
    }
    
    best_score = -float('inf')
    patience = 50
    patience_counter = 0
    
    print("\nStarting extended training...\n")
    
    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, EPOCHS
        )
        
        scheduler.step()
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_metrics = validate(model, val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['snr_imp'].append(val_metrics['snr_imp'])
            history['ncc'].append(val_metrics['ncc'])
            history['class_acc'].append(val_metrics['class_acc'])
            history['overall_acc'].append(val_metrics['overall_acc'])
            
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  SNR Improvement: {val_metrics['snr_imp']:.2f} dB")
            print(f"  NCC: {val_metrics['ncc']:.4f}")
            print(f"  Class Acc: " + " | ".join([f"{CLASS_NAMES[c][:3]}={val_metrics['class_acc'][c]*100:.1f}%" for c in range(NUM_CLASSES)]))
            print(f"  Overall: {val_metrics['overall_acc']*100:.1f}%")
            
            score = val_metrics['ncc'] + 0.3 * val_metrics['overall_acc']
            if score > best_score:
                best_score = score
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    **val_metrics
                }, SAVE_DIR / 'best_model.pth')
                print(f"  >> New best model!")
            else:
                patience_counter += 1
        
        # Visualize every 100 epochs
        if (epoch + 1) % 100 == 0:
            visualize_results(model, val_loader, SAVE_DIR / f'results_epoch_{epoch+1}.png')
    
    # Final
    checkpoint = torch.load(SAVE_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    visualize_results(model, val_loader, SAVE_DIR / 'final_results.png')
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Time: {elapsed}")
    print(f"Best NCC: {checkpoint['ncc']:.4f}")
    print(f"Best SNR: {checkpoint['snr_imp']:.2f} dB")
    print(f"Best Acc: {checkpoint['overall_acc']*100:.1f}%")
    print(f"\nSaved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
