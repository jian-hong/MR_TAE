#!/usr/bin/env python
"""
train_mr_tae_gem.py - MR-TAE-GEM: Gradient Episodic Memory Enhanced Training

FIXES FOR CRITICAL ISSUES IN train_comprehensive.py:

1. VALIDATION MIRAGE FIX:
   - Validation dataset now syncs with current training epoch
   - val_dataset.update_epoch(epoch) called each epoch

2. CATASTROPHIC FORGETTING FIX:
   - Overlapping curriculum phases (no hard transitions)
   - No "pure noise" phase (always includes PD signals)
   - Replay buffer mixes samples from previous phases
   - Optional EWC regularization for weight protection

3. PHANTOM TGAN FIX:
   - Uses colored noise (1/f flicker, substation)
   - TGAN integration if pre-trained weights available

USAGE:
    python mr_tae_gem/train_mr_tae_gem.py --epochs 700 --samples 200000
    python mr_tae_gem/train_mr_tae_gem.py --quick_test  # Quick validation test
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

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

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data.qlin_loader import QLINDataLoader
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse

# Import GEM components - add gem directory to path for standalone execution
gem_dir = Path(__file__).parent
sys.path.insert(0, str(gem_dir))
from gem_dataset import GEMDataset, EWCRegularizer, get_overlapping_curriculum_phase
from colored_noise import add_colored_noise_at_snr


# =============================================================================
# CONFIGURATION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MR-TAE-GEM: Gradient Episodic Memory Training")
    
    # Training params
    parser.add_argument('--epochs', type=int, default=700, help='Total epochs')
    parser.add_argument('--samples', type=int, default=200000, help='Total training samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    # Loss weights
    parser.add_argument('--alpha_ncc', type=float, default=0.8, help='NCC loss weight')
    parser.add_argument('--alpha_seg', type=float, default=0.3, help='Segmentation loss weight')
    parser.add_argument('--alpha_rmse', type=float, default=0.3, help='RMSE weight')
    
    # GEM-specific params
    parser.add_argument('--use_ewc', action='store_true', help='Enable EWC regularization')
    parser.add_argument('--lambda_ewc', type=float, default=0.1, help='EWC penalty strength')
    parser.add_argument('--replay_ratio', type=float, default=0.2, help='Replay buffer mix ratio')
    parser.add_argument('--tgan_weights', type=str, default=None, help='Path to TGAN weights')
    
    # Data params
    parser.add_argument('--qlin_ratio', type=float, default=0.7, help='Q.Lin data ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 10 epochs, 1000 samples')
    
    return parser.parse_args()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 5
CLASS_NAMES = ['Background', 'Corona', 'Surface', 'Internal', 'Treeing']


# =============================================================================
# LOSS FUNCTIONS (Same as original, with NaN protection)
# =============================================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.eps_sq = epsilon ** 2
    
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class NCCLoss(nn.Module):
    """NCC Loss with NaN protection."""
    def forward(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_c = target_flat - target_flat.mean(dim=1, keepdim=True)
        
        pred_var = (pred_c ** 2).sum(dim=1)
        target_var = (target_c ** 2).sum(dim=1)
        
        valid_mask = (pred_var > 1e-6) & (target_var > 1e-6)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        numerator = (pred_c * target_c).sum(dim=1)
        denominator = torch.sqrt(pred_var * target_var + 1e-8)
        
        ncc = numerator / denominator
        ncc = ncc[valid_mask]
        ncc = torch.clamp(ncc, -1.0, 1.0)
        
        return 1 - ncc.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=5):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        
        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 2, 1).float()
        
        intersection = (pred_soft * target_oh).sum(dim=2)
        union = pred_soft.sum(dim=2) + target_oh.sum(dim=2)
        
        class_counts = target_oh.sum(dim=(0, 2)) + 1
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        
        dice = (2 * intersection + 1) / (union + 1)
        weighted_dice = (dice * weights.unsqueeze(0)).sum(dim=1)
        
        return focal_loss + (1 - weighted_dice.mean())


class GEMMultiTaskLoss(nn.Module):
    """Multi-task loss with configurable weights."""
    
    def __init__(self, alpha_charb=1.0, alpha_ncc=0.8, alpha_rmse=0.3, 
                 alpha_seg=0.3, num_classes=5):
        super().__init__()
        
        self.alpha_charb = alpha_charb
        self.alpha_ncc = alpha_ncc
        self.alpha_rmse = alpha_rmse
        self.alpha_seg = alpha_seg
        
        self.log_var_denoise = nn.Parameter(torch.tensor(0.0))
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        
        self.charbonnier = CharbonnierLoss()
        self.ncc_loss = NCCLoss()
        self.focal_dice = FocalDiceLoss(gamma=2.0, num_classes=num_classes)
    
    def forward(self, denoised, clean, seg_pred, seg_target):
        l_charb = self.charbonnier(denoised, clean)
        l_ncc = self.ncc_loss(denoised, clean)
        l_rmse = torch.sqrt(F.mse_loss(denoised, clean) + 1e-8)
        
        l_denoise = (self.alpha_charb * l_charb + 
                     self.alpha_ncc * l_ncc + 
                     self.alpha_rmse * l_rmse)
        
        l_seg = self.focal_dice(seg_pred, seg_target)
        
        var_denoise = torch.exp(torch.clamp(self.log_var_denoise, -10, 10))
        var_seg = torch.exp(torch.clamp(self.log_var_seg, -10, 10))
        
        total_loss = (l_denoise / (2 * var_denoise) + 0.5 * self.log_var_denoise +
                      self.alpha_seg * l_seg / (2 * var_seg) + 0.5 * self.log_var_seg)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = l_charb + l_seg
        
        return total_loss, {
            'total': total_loss.item() if not torch.isnan(total_loss) else 0.0,
            'charb': l_charb.item() if not torch.isnan(l_charb) else 0.0,
            'ncc': l_ncc.item() if not torch.isnan(l_ncc) else 0.0,
            'rmse': l_rmse.item() if not torch.isnan(l_rmse) else 0.0,
            'seg': l_seg.item() if not torch.isnan(l_seg) else 0.0,
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, 
                total_epochs, device, ewc_regularizer=None):
    model.train()
    train_loader.dataset.update_epoch(epoch)  # FIX: Sync epoch
    
    phase = get_overlapping_curriculum_phase(epoch, total_epochs)
    
    total_loss = 0
    metrics = {'charb': 0, 'ncc': 0, 'rmse': 0, 'seg': 0, 'ewc': 0}
    
    pbar = tqdm(train_loader, desc=f"[Phase {phase['phase']}:{phase['name']}] Ep{epoch+1}/{total_epochs}")
    
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=True):
            denoised, seg_logits = model(noisy)
            loss, loss_dict = criterion(denoised, clean, seg_logits, mask)
            
            # Add EWC penalty if enabled
            if ewc_regularizer is not None and ewc_regularizer.initialized:
                ewc_loss = ewc_regularizer.penalty()
                loss = loss + ewc_loss
                metrics['ewc'] += ewc_loss.item()
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k in ['charb', 'ncc', 'rmse', 'seg']:
            if k in loss_dict and not np.isnan(loss_dict[k]):
                metrics[k] += loss_dict[k]
        
        ncc_val = 1 - loss_dict.get('ncc', 0)
        if np.isnan(ncc_val):
            ncc_val = 0.0
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ncc': f"{ncc_val:.3f}"})
    
    n = len(train_loader)
    return total_loss/max(n, 1), {k: v/max(n, 1) for k, v in metrics.items()}


def validate(model, val_loader, criterion, epoch, total_epochs, device):
    """
    FIX FOR VALIDATION MIRAGE: Sync validation with current epoch.
    """
    model.eval()
    
    # FIX: Update validation dataset to current epoch
    val_loader.dataset.update_epoch(epoch)
    
    phase = get_overlapping_curriculum_phase(epoch, total_epochs)
    
    total_loss = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    snr_imps, nccs, rmses = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val [Phase {phase['phase']}]", leave=False):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)
            
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
                    snr_imps.append(calculate_snr_improvement(noisy_np[i], denoised_np[i], clean_np[i]))
                    nccs.append(calculate_ncc(denoised_np[i], clean_np[i]))
                    rmses.append(calculate_rmse(denoised_np[i], clean_np[i]))
                except:
                    pass
    
    n = len(val_loader)
    class_acc = [class_correct[c] / max(class_total[c], 1) for c in range(NUM_CLASSES)]
    overall_acc = sum(class_correct) / max(sum(class_total), 1)
    
    return {
        'loss': total_loss / max(n, 1),
        'snr_imp': np.mean(snr_imps) if snr_imps else 0,
        'ncc': np.mean(nccs) if nccs else 0,
        'rmse': np.mean(rmses) if rmses else 0,
        'class_acc': class_acc,
        'overall_acc': overall_acc,
        'phase': phase['phase'],
        'phase_name': phase['name']
    }


def save_comparison_visualization(model, val_loader, save_dir, epoch, device):
    """Save visualization comparing noisy/denoised/clean."""
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            
            with autocast('cuda', enabled=True):
                denoised, seg_logits = model(noisy)
            
            # Take first 4 samples
            n_samples = min(4, len(noisy))
            fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
            
            for i in range(n_samples):
                noisy_np = noisy[i, 0].cpu().numpy()
                clean_np = clean[i, 0].cpu().numpy()
                denoised_np = denoised[i, 0].cpu().numpy()
                
                axes[i, 0].plot(noisy_np, 'b-', alpha=0.7, linewidth=0.5)
                axes[i, 0].set_title(f'Noisy (SNR={batch["snr"][i]:.1f}dB)')
                axes[i, 0].set_ylabel(f'Sample {i+1}')
                
                axes[i, 1].plot(denoised_np, 'r-', linewidth=1)
                axes[i, 1].set_title('Denoised')
                
                axes[i, 2].plot(clean_np, 'g-', linewidth=1)
                axes[i, 2].set_title('Ground Truth')
            
            plt.tight_layout()
            phase = get_overlapping_curriculum_phase(epoch, 700)
            fig.suptitle(f'Epoch {epoch+1} - Phase {phase["phase"]}: {phase["name"]}', y=1.02)
            
            save_path = save_dir / 'visualizations' / f'comparison_epoch_{epoch+1}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    # Quick test mode overrides
    if args.quick_test:
        args.epochs = 10
        args.samples = 1000
        print("=" * 70)
        print("QUICK TEST MODE: 10 epochs, 1000 samples")
        print("=" * 70)
    
    start_time = datetime.now()
    
    # Create dedicated run folder
    run_name = f"gem_{start_time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path("outputs") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'visualizations').mkdir(exist_ok=True)
    
    print("=" * 70)
    print("MR-TAE-GEM: GRADIENT EPISODIC MEMORY TRAINING")
    print("=" * 70)
    print(f"FIXES APPLIED:")
    print(f"  1. Validation Mirage: Validation syncs with current epoch")
    print(f"  2. Catastrophic Forgetting: Overlapping curriculum + replay buffer")
    print(f"  3. Phantom TGAN: Using colored noise (1/f, substation)")
    print(f"")
    print(f"Run Folder: {save_dir}")
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {args.epochs}")
    print(f"Samples: {args.samples}")
    print(f"Replay Ratio: {args.replay_ratio}")
    print(f"EWC Enabled: {args.use_ewc}")
    print()
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load Q.Lin data
    print("Loading Q.Lin real data...")
    qlin_loader = QLINDataLoader()
    
    # Model
    config = get_config()
    config.model.num_classes = NUM_CLASSES
    config.signal.num_classes = NUM_CLASSES
    model = create_model(config.model).to(DEVICE)
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Loss
    criterion = GEMMultiTaskLoss(
        alpha_charb=1.0,
        alpha_ncc=args.alpha_ncc,
        alpha_rmse=args.alpha_rmse,
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
    
    # GEM Datasets (FIX: Initialize at epoch 0, not epochs-1)
    train_dataset = GEMDataset(
        config=config,
        num_samples=args.samples,
        mode='train',
        epoch=0,  # FIX: Start at epoch 0
        total_epochs=args.epochs,
        qlin_loader=qlin_loader,
        seed=args.seed,
        use_replay=True,
        replay_ratio=args.replay_ratio,
        tgan_weights_path=args.tgan_weights
    )
    
    val_samples = min(args.samples // 10, 5000)
    val_dataset = GEMDataset(
        config=config,
        num_samples=val_samples,
        mode='val',
        epoch=0,  # FIX: Start at epoch 0 (will be updated each epoch)
        total_epochs=args.epochs,
        qlin_loader=qlin_loader,
        seed=args.seed + 1,
        use_replay=False  # No replay for validation
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    scaler = GradScaler('cuda')
    
    # EWC Regularizer (optional)
    ewc_regularizer = None
    if args.use_ewc:
        ewc_regularizer = EWCRegularizer(model, lambda_ewc=args.lambda_ewc)
        print("EWC Regularization enabled")
    
    # Resume if specified
    start_epoch = 0
    best_score = -float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Tracking for phase transitions
    last_phase = 0
    validation_history = []
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting GEM Training with Overlapping Curriculum...")
    print("=" * 70 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            epoch, args.epochs, DEVICE, ewc_regularizer
        )
        
        scheduler.step()
        
        # Check for phase transition (for EWC consolidation)
        current_phase = get_overlapping_curriculum_phase(epoch, args.epochs)['phase']
        if current_phase != last_phase and ewc_regularizer is not None:
            print(f"\n  >> Phase transition {last_phase} -> {current_phase}, computing Fisher Information...")
            ewc_regularizer.compute_fisher(train_loader, criterion, DEVICE)
            last_phase = current_phase
        
        # Validate every 10 epochs (or every epoch for quick test)
        validate_freq = 1 if args.quick_test else 10
        if (epoch + 1) % validate_freq == 0 or epoch == 0:
            val_metrics = validate(model, val_loader, criterion, epoch, args.epochs, DEVICE)
            
            phase = get_overlapping_curriculum_phase(epoch, args.epochs)
            
            print(f"\nEpoch {epoch+1}/{args.epochs} [Phase {phase['phase']}:{phase['name']}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} (Phase-matched)")
            print(f"  SNR Imp: {val_metrics['snr_imp']:.2f} dB | NCC: {val_metrics['ncc']:.4f} | RMSE: {val_metrics['rmse']:.6f}")
            print(f"  Class: " + " | ".join([f"{CLASS_NAMES[c][:3]}={val_metrics['class_acc'][c]*100:.1f}%" for c in range(NUM_CLASSES)]))
            print(f"  Overall: {val_metrics['overall_acc']*100:.1f}%")
            
            # Track validation history (FIX: should now show progression)
            validation_history.append({
                'epoch': epoch + 1,
                'phase': phase['phase'],
                'loss': val_metrics['loss'],
                'ncc': val_metrics['ncc'],
                'snr_imp': val_metrics['snr_imp']
            })
            
            # Save best model
            score = val_metrics['ncc'] + 0.3 * val_metrics['overall_acc']
            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    **val_metrics
                }, save_dir / 'best_model.pth')
                print("  >> New best model saved!")
        
        # Save visualization every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_comparison_visualization(model, val_loader, save_dir, epoch, DEVICE)
            print("  >> Visualization saved")
        
        # Checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FINAL EVALUATION")
    print("=" * 70)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = validate(model, val_loader, criterion, args.epochs - 1, args.epochs, DEVICE)
    
    elapsed = datetime.now() - start_time
    
    print(f"\nBest Results:")
    print(f"  NCC: {final_metrics['ncc']:.4f}")
    print(f"  SNR Improvement: {final_metrics['snr_imp']:.2f} dB")
    print(f"  RMSE: {final_metrics['rmse']:.6f}")
    print(f"  Overall Accuracy: {final_metrics['overall_acc']*100:.1f}%")
    print(f"  Time: {elapsed}")
    
    # Save validation history (proves Validation Mirage is fixed)
    with open(save_dir / 'validation_history.json', 'w') as f:
        json.dump(validation_history, f, indent=2)
    
    # Save final summary
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump({
            'fixes_applied': [
                'Validation Mirage: val_dataset synced with training epoch',
                'Catastrophic Forgetting: Overlapping curriculum + replay',
                'Phantom TGAN: Colored noise (flicker + substation)'
            ],
            'best_ncc': float(final_metrics['ncc']),
            'best_snr_imp': float(final_metrics['snr_imp']),
            'best_rmse': float(final_metrics['rmse']),
            'best_overall_acc': float(final_metrics['overall_acc']),
            'class_acc': [float(x) for x in final_metrics['class_acc']],
            'epochs': args.epochs,
            'samples': args.samples,
            'replay_ratio': args.replay_ratio,
            'ewc_enabled': args.use_ewc,
            'elapsed': str(elapsed),
            'run_name': run_name
        }, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    print(f"Validation history: {save_dir / 'validation_history.json'}")
    print(f"  ^ Check this to verify NCC/loss changes with epoch (not flatlined)")


if __name__ == '__main__':
    main()
