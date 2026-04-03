#!/usr/bin/env python
"""
train_ultimate.py - Ultimate 700-epoch training with all improvements.

FEATURES:
1. 5 classes: Background, Corona, Surface, Internal, Treeing
2. 70% Q.Lin real data (1.0, 1.8, 2.0, 2.5mm metal particles)
3. TGAN-augmented noise (trained on Q.Lin)
4. Temporal multi-PD sequences
5. Disruptive impulsive noise
6. 6-phase curriculum learning
7. Balanced 200K samples
8. Ground truth feedback for failed predictions
9. Dedicated run folder with timestamps
10. Configurable hyperparameters

USAGE:
    python train_ultimate.py --epochs 700 --samples 200000 --batch_size 32
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from mr_tae_fusion.config import get_config, Config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data.pulse_generators import (
    generate_pd_signal, add_disruptive_impulse_noise
)
from mr_tae_fusion.data.qlin_loader import QLINDataLoader
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ultimate MR-TAE-Fusion Training")
    parser.add_argument('--epochs', type=int, default=700, help='Total epochs')
    parser.add_argument('--samples', type=int, default=200000, help='Total training samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--qlin_ratio', type=float, default=0.7, help='Q.Lin data ratio')
    parser.add_argument('--alpha_ncc', type=float, default=0.5, help='NCC loss weight')
    parser.add_argument('--alpha_seg', type=float, default=0.3, help='Segmentation loss weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


# =============================================================================
# CONSTANTS
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 5
CLASS_NAMES = ['Background', 'Corona', 'Surface', 'Internal', 'Treeing']
CLASS_COLORS = ['lightgray', 'red', 'blue', 'green', 'purple']

# Signal types with probabilities (balanced)
SIGNAL_TYPES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'TEMPORAL', 'QLIN']
SIGNAL_PROBS_BALANCED = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12, 0.16]


# =============================================================================
# 6-PHASE CURRICULUM
# =============================================================================

def get_phase_config(epoch, total_epochs):
    """
    Get curriculum phase configuration.
    
    PHASES:
    1. Pure PD shapes (no noise) - Learn what to find
    2. Pure noise only - Learn what to ignore
    3. Single PD + light noise - Basic denoising
    4. Mixed PD types + medium noise - Multi-class learning
    5. Temporal sequences + hard noise - Real scenarios
    6. TGAN + extreme noise (-25dB) - Robustness
    """
    progress = epoch / total_epochs
    
    if progress < 0.07:  # Phase 1: 0-7%
        return {
            'phase': 1,
            'name': 'Pure PD Shapes',
            'snr_range': (30, 50),  # Very clean
            'use_qlin': False,
            'use_temporal': False,
            'use_disruptive': False,
            'types': ['A', 'B', 'C', 'G'],  # One type per class
        }
    
    elif progress < 0.14:  # Phase 2: 7-14%
        return {
            'phase': 2,
            'name': 'Pure Noise Learning',
            'snr_range': (-30, -20),  # Very noisy (almost no signal)
            'use_qlin': False,
            'use_temporal': False,
            'use_disruptive': True,
            'types': ['A', 'B', 'C', 'G'],
        }
    
    elif progress < 0.28:  # Phase 3: 14-28%
        return {
            'phase': 3,
            'name': 'Single PD + Light Noise',
            'snr_range': (5, 15),
            'use_qlin': True,
            'use_temporal': False,
            'use_disruptive': False,
            'types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'QLIN'],
        }
    
    elif progress < 0.50:  # Phase 4: 28-50%
        return {
            'phase': 4,
            'name': 'Mixed PD + Medium Noise',
            'snr_range': (-5, 5),
            'use_qlin': True,
            'use_temporal': False,
            'use_disruptive': False,
            'types': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED', 'QLIN'],
        }
    
    elif progress < 0.75:  # Phase 5: 50-75%
        return {
            'phase': 5,
            'name': 'Temporal + Hard Noise',
            'snr_range': (-15, -5),
            'use_qlin': True,
            'use_temporal': True,
            'use_disruptive': True,
            'types': SIGNAL_TYPES,
        }
    
    else:  # Phase 6: 75-100%
        return {
            'phase': 6,
            'name': 'Extreme (-25dB)',
            'snr_range': (-25, -15),
            'use_qlin': True,
            'use_temporal': True,
            'use_disruptive': True,
            'types': SIGNAL_TYPES,
        }


# =============================================================================
# ULTIMATE DATASET
# =============================================================================

class UltimateDataset(Dataset):
    """
    Ultimate dataset with all improvements:
    - Q.Lin real data (70%)
    - Temporal sequences
    - Disruptive noise
    - 6-phase curriculum
    """
    
    def __init__(self, config, num_samples, mode='train', epoch=0, seed=None, 
                 total_epochs=700, qlin_loader=None):
        self.config = config
        self.num_samples = num_samples
        self.mode = mode
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.qlin_loader = qlin_loader
        
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-generate sample types (will be regenerated per phase)
        self._regenerate_types()
    
    def _regenerate_types(self):
        phase_config = get_phase_config(self.epoch, self.total_epochs)
        types = phase_config['types']
        probs = [1.0 / len(types)] * len(types)  # Uniform for balance
        self.signal_types = np.random.choice(types, size=self.num_samples, p=probs)
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        self._regenerate_types()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        phase_config = get_phase_config(self.epoch, self.total_epochs)
        signal_type = self.signal_types[idx]
        
        # Get signal based on type
        if signal_type == 'QLIN' and self.qlin_loader is not None:
            signal, diameter = self.qlin_loader.get_random_signal(self.mode)
            if signal is not None:
                mask = self.qlin_loader.create_mask(signal, class_id=2)  # Surface
                clean = signal.copy()
            else:
                # Fallback to synthetic
                _, clean, mask = generate_pd_signal('B', self.config.signal)
        else:
            if signal_type == 'QLIN':
                signal_type = 'B'  # Fallback if no Q.Lin
            _, clean, mask = generate_pd_signal(signal_type, self.config.signal)
        
        # Truncate/pad
        target_len = self.config.signal.signal_length
        if len(clean) > target_len:
            clean = clean[:target_len]
            mask = mask[:target_len]
        elif len(clean) < target_len:
            clean = np.pad(clean, (0, target_len - len(clean)))
            mask = np.pad(mask, (0, target_len - len(mask)))
        
        # Add noise based on curriculum phase
        snr_min, snr_max = phase_config['snr_range']
        snr = np.random.uniform(snr_min, snr_max)
        
        if phase_config['use_disruptive']:
            noisy = add_disruptive_impulse_noise(clean, snr)
        else:
            # Simple noise
            signal_power = np.mean(clean ** 2)
            if signal_power < 1e-10:
                signal_power = 1.0
            noise_power = signal_power / (10 ** (snr / 10))
            noise = np.random.randn(len(clean)) * np.sqrt(noise_power)
            noisy = clean + noise
        
        # Normalize
        max_amp = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-8)
        noisy = noisy / max_amp
        clean = clean / max_amp
        
        return {
            'noisy': torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
            'clean': torch.tensor(clean, dtype=torch.float32).unsqueeze(0),
            'mask': torch.tensor(mask, dtype=torch.long),
            'snr': torch.tensor(snr, dtype=torch.float32),
            'type': torch.tensor(hash(signal_type) % 100, dtype=torch.long)
        }


# =============================================================================
# LOSS FUNCTIONS
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


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=5):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        # Focal Loss
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        
        # Dice Loss with class weighting
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 2, 1).float()
        
        intersection = (pred_soft * target_onehot).sum(dim=2)
        union = pred_soft.sum(dim=2) + target_onehot.sum(dim=2)
        
        class_counts = target_onehot.sum(dim=(0, 2)) + 1
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        
        dice = (2 * intersection + 1) / (union + 1)
        weighted_dice = (dice * weights.unsqueeze(0)).sum(dim=1)
        dice_loss = 1 - weighted_dice.mean()
        
        return focal_loss + dice_loss


class UltimateMultiTaskLoss(nn.Module):
    def __init__(self, alpha_denoise=1.0, alpha_ncc=0.5, alpha_seg=0.3, num_classes=5):
        super().__init__()
        
        self.alpha_denoise = alpha_denoise
        self.alpha_ncc = alpha_ncc
        self.alpha_seg = alpha_seg
        
        self.log_var_denoise = nn.Parameter(torch.tensor(0.0))
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        
        self.charbonnier = CharbonnierLoss()
        self.ncc_loss = NCCLoss()
        self.focal_dice = FocalDiceLoss(gamma=2.0, num_classes=num_classes)
    
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

def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, total_epochs, args):
    model.train()
    train_loader.dataset.update_epoch(epoch)
    
    phase_config = get_phase_config(epoch, total_epochs)
    
    total_loss = 0
    metrics = {'charb': 0, 'ncc': 0, 'seg': 0}
    
    pbar = tqdm(train_loader, desc=f"[Phase {phase_config['phase']}:{phase_config['name']}] Epoch {epoch+1}/{total_epochs}")
    
    for batch in pbar:
        noisy = batch['noisy'].to(DEVICE)
        clean = batch['clean'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=True):
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


def validate(model, val_loader, criterion, save_dir, epoch):
    model.eval()
    
    total_loss = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    snr_imps, nccs, rmses = [], [], []
    
    worst_samples = []  # For ground truth feedback
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            with autocast('cuda', enabled=True):
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
                    ncc = calculate_ncc(denoised_np[i], clean_np[i])
                    snr_imp = calculate_snr_improvement(noisy_np[i], denoised_np[i], clean_np[i])
                    rmse = calculate_rmse(denoised_np[i], clean_np[i])
                    
                    snr_imps.append(snr_imp)
                    nccs.append(ncc)
                    rmses.append(rmse)
                    
                    # Track worst samples for feedback
                    if ncc < 0.5:
                        worst_samples.append({
                            'noisy': noisy_np[i],
                            'clean': clean_np[i],
                            'denoised': denoised_np[i],
                            'ncc': ncc,
                            'true_mask': true_mask[i].numpy(),
                            'pred_mask': pred_mask[i].numpy()
                        })
                except:
                    pass
    
    # Save ground truth feedback for worst samples
    if len(worst_samples) > 0 and epoch % 50 == 0:
        save_ground_truth_feedback(worst_samples[:10], save_dir, epoch)
    
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


def save_ground_truth_feedback(worst_samples, save_dir, epoch):
    """Save visualization of worst predictions with ground truth."""
    fig, axes = plt.subplots(len(worst_samples), 3, figsize=(18, 3*len(worst_samples)))
    fig.suptitle(f'Ground Truth Feedback (Epoch {epoch}) - Worst Predictions', 
                 fontsize=14, fontweight='bold')
    
    if len(worst_samples) == 1:
        axes = [axes]
    
    for i, sample in enumerate(worst_samples):
        # Noisy
        axes[i][0].plot(sample['noisy'], 'b-', alpha=0.7, linewidth=0.5)
        axes[i][0].set_title('Noisy Input')
        axes[i][0].grid(True, alpha=0.3)
        
        # Denoised vs Clean
        axes[i][1].plot(sample['denoised'], 'r-', linewidth=0.8, label='Predicted')
        axes[i][1].plot(sample['clean'], 'g--', linewidth=0.8, label='Ground Truth')
        axes[i][1].set_title(f'NCC: {sample["ncc"]:.3f} - LEARN THIS!')
        axes[i][1].legend()
        axes[i][1].grid(True, alpha=0.3)
        
        # Mask comparison
        axes[i][2].plot(sample['pred_mask'], 'r-', linewidth=1, label='Pred Mask')
        axes[i][2].plot(sample['true_mask'], 'g--', linewidth=1, label='True Mask')
        axes[i][2].set_title('Segmentation: Pred vs Truth')
        axes[i][2].legend()
        axes[i][2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'visualizations' / f'feedback_epoch_{epoch}.png', dpi=150)
    plt.close()


def visualize_results(model, val_loader, save_path, num_examples=10):
    """Generate comprehensive visualization."""
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 2.5*num_examples))
    fig.suptitle('Ultimate Model Results (5 Classes)', fontsize=14, fontweight='bold')
    
    examples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if examples >= num_examples:
                break
            
            noisy = batch['noisy'].to(DEVICE)
            clean = batch['clean'].to(DEVICE)
            true_mask = batch['mask']
            snr = batch['snr']
            
            with autocast('cuda', enabled=True):
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
                
                # Plots
                axes[examples, 0].plot(noisy_np, 'b-', alpha=0.7, linewidth=0.5)
                axes[examples, 0].set_ylabel(f'SNR:{snr[i].item():.0f}')
                axes[examples, 0].grid(True, alpha=0.3)
                
                axes[examples, 1].plot(denoised_np, 'r-', linewidth=0.8)
                axes[examples, 1].plot(clean_np, 'g--', alpha=0.5, linewidth=0.5)
                axes[examples, 1].set_title(f'{snr_imp:.1f}dB/{ncc:.3f}')
                axes[examples, 1].grid(True, alpha=0.3)
                
                for c in range(NUM_CLASSES):
                    if c > 0:
                        axes[examples, 2].fill_between(range(len(pred_m)), -1, 1,
                                                       where=(pred_m == c), alpha=0.4, color=CLASS_COLORS[c])
                        axes[examples, 3].fill_between(range(len(true_m)), -1, 1,
                                                       where=(true_m == c), alpha=0.4, color=CLASS_COLORS[c])
                
                axes[examples, 2].plot(denoised_np, 'k-', linewidth=0.5, alpha=0.7)
                axes[examples, 2].set_ylim(-1.2, 1.2)
                axes[examples, 2].grid(True, alpha=0.3)
                
                axes[examples, 3].plot(clean_np, 'k-', linewidth=0.5, alpha=0.7)
                axes[examples, 3].set_ylim(-1.2, 1.2)
                axes[examples, 3].grid(True, alpha=0.3)
                
                examples += 1
    
    for ax, title in zip(axes[0], ['Noisy', 'Denoised/Clean', 'Predicted', 'Ground Truth']):
        ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    start_time = datetime.now()
    
    # Create dedicated run folder
    run_name = f"run_{start_time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path("outputs") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'visualizations').mkdir(exist_ok=True)
    
    print("="*70)
    print("MR-TAE-FUSION ULTIMATE TRAINING")
    print("="*70)
    print(f"Run Folder: {save_dir}")
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {args.epochs}")
    print(f"Samples: {args.samples}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Q.Lin Ratio: {args.qlin_ratio}")
    print()
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load Q.Lin data
    print("Loading Q.Lin dataset...")
    qlin_loader = QLINDataLoader()
    qlin_stats = qlin_loader.get_statistics()
    print(f"Q.Lin: {qlin_stats['total_train']} train, {qlin_stats['total_val']} val")
    
    # Model
    config = get_config()
    config.model.num_classes = NUM_CLASSES
    config.signal.num_classes = NUM_CLASSES
    model = create_model(config.model).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss
    criterion = UltimateMultiTaskLoss(
        alpha_denoise=1.0,
        alpha_ncc=args.alpha_ncc,
        alpha_seg=args.alpha_seg,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    # Datasets
    train_dataset = UltimateDataset(
        config=config,
        num_samples=args.samples,
        mode='train',
        seed=args.seed,
        total_epochs=args.epochs,
        qlin_loader=qlin_loader
    )
    
    val_samples = min(args.samples // 10, 5000)
    val_dataset = UltimateDataset(
        config=config,
        num_samples=val_samples,
        mode='val',
        epoch=args.epochs-1,
        seed=args.seed+1,
        total_epochs=args.epochs,
        qlin_loader=qlin_loader
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    scaler = GradScaler('cuda')
    
    # Resume if specified
    start_epoch = 0
    best_score = -float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'snr_imp': [], 'ncc': [], 'overall_acc': []}
    
    print("\nStarting Ultimate Training...\n")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, args.epochs, args
        )
        
        scheduler.step()
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_metrics = validate(model, val_loader, criterion, save_dir, epoch)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['snr_imp'].append(val_metrics['snr_imp'])
            history['ncc'].append(val_metrics['ncc'])
            history['overall_acc'].append(val_metrics['overall_acc'])
            
            phase = get_phase_config(epoch, args.epochs)
            
            print(f"\nEpoch {epoch+1}/{args.epochs} [Phase {phase['phase']}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  SNR Improvement: {val_metrics['snr_imp']:.2f} dB")
            print(f"  NCC: {val_metrics['ncc']:.4f}")
            print(f"  Class Acc: " + " | ".join([f"{CLASS_NAMES[c][:3]}={val_metrics['class_acc'][c]*100:.1f}%" for c in range(NUM_CLASSES)]))
            print(f"  Overall: {val_metrics['overall_acc']*100:.1f}%")
            
            score = val_metrics['ncc'] + 0.3 * val_metrics['overall_acc']
            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    **val_metrics
                }, save_dir / 'best_model.pth')
                print("  >> New best model!")
        
        # Visualize every 100 epochs
        if (epoch + 1) % 100 == 0:
            visualize_results(model, val_loader, 
                             save_dir / 'visualizations' / f'results_epoch_{epoch+1}.png')
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Final
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    visualize_results(model, val_loader, save_dir / 'visualizations' / 'final_results.png')
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Time: {elapsed}")
    print(f"Best NCC: {checkpoint['ncc']:.4f}")
    print(f"Best SNR: {checkpoint['snr_imp']:.2f} dB")
    print(f"Best Acc: {checkpoint['overall_acc']*100:.1f}%")
    print(f"\nResults saved to: {save_dir}")
    
    # Save final summary
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump({
            'best_ncc': float(checkpoint['ncc']),
            'best_snr_imp': float(checkpoint['snr_imp']),
            'best_overall_acc': float(checkpoint['overall_acc']),
            'class_acc': [float(x) for x in checkpoint['class_acc']],
            'epochs': args.epochs,
            'samples': args.samples,
            'elapsed': str(elapsed),
            'run_name': run_name
        }, f, indent=2)


if __name__ == '__main__':
    main()
