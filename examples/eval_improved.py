#!/usr/bin/env python
"""Quick evaluation of improved model results."""

import sys
sys.path.insert(0, 'D:/New folder (2)')
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse

DEVICE = 'cuda'
SAVE_DIR = Path('D:/New folder (2)/outputs/improved_training')

# Load model
config = get_config()
model = create_model(config.model).to(DEVICE)
ckpt = torch.load(SAVE_DIR / 'best_model.pth', map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print("="*70)
print("IMPROVED MODEL EVALUATION")
print("="*70)
print(f"Best Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Saved NCC: {ckpt.get('ncc', 'N/A'):.4f}")
print(f"Saved SNR Imp: {ckpt.get('snr_imp', 'N/A'):.2f} dB")
print(f"Saved Overall Acc: {ckpt.get('overall_acc', 'N/A')*100:.1f}%")
if 'class_acc' in ckpt:
    print(f"Class Acc: BG={ckpt['class_acc'][0]*100:.1f}% | Corona={ckpt['class_acc'][1]*100:.1f}% | Surface={ckpt['class_acc'][2]*100:.1f}% | Internal={ckpt['class_acc'][3]*100:.1f}%")

# Re-evaluate on fresh data
val_dataset = PDSignalDataset(config=config, num_samples=500, mode='val', epoch=100, seed=999)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

CLASS_NAMES = ['Background', 'Corona', 'Surface', 'Internal']
class_correct = [0, 0, 0, 0]
class_total = [0, 0, 0, 0]
snr_imps = []
nccs = []
rmses = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        noisy = batch['noisy'].to(DEVICE)
        clean = batch['clean'].to(DEVICE)
        true_mask = batch['mask'].numpy()
        
        with autocast('cuda'):
            denoised, seg_logits = model(noisy)
        
        pred_mask = seg_logits.argmax(dim=1).cpu().numpy()
        
        for i in range(len(true_mask)):
            for c in range(4):
                mask_c = (true_mask[i] == c)
                class_total[c] += mask_c.sum()
                class_correct[c] += ((pred_mask[i] == c) & mask_c).sum()
            
            noisy_np = noisy[i, 0].cpu().numpy()
            clean_np = clean[i, 0].cpu().numpy()
            denoised_np = denoised[i, 0].cpu().numpy()
            
            try:
                snr_imps.append(calculate_snr_improvement(noisy_np, denoised_np, clean_np))
                nccs.append(calculate_ncc(denoised_np, clean_np))
                rmses.append(calculate_rmse(denoised_np, clean_np))
            except:
                pass

print("\n" + "="*70)
print("FRESH EVALUATION RESULTS")
print("="*70)
print("\nSEGMENTATION ACCURACY:")
for c in range(4):
    if class_total[c] > 0:
        acc = class_correct[c] / class_total[c] * 100
        print(f"  {CLASS_NAMES[c]:12s}: {acc:6.2f}% ({class_correct[c]:,} / {class_total[c]:,})")

overall = sum(class_correct) / sum(class_total) * 100
print(f"  {'OVERALL':12s}: {overall:6.2f}%")

print("\nDENOISING QUALITY:")
print(f"  SNR Improvement: {np.mean(snr_imps):.2f} +/- {np.std(snr_imps):.2f} dB")
print(f"  NCC (Shape):     {np.mean(nccs):.4f} +/- {np.std(nccs):.4f}")
print(f"  RMSE:            {np.mean(rmses):.6f} +/- {np.std(rmses):.6f}")
print("="*70)

# Save results
with open(SAVE_DIR / 'final_eval_results.txt', 'w') as f:
    f.write("IMPROVED MODEL EVALUATION RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write("SEGMENTATION ACCURACY:\n")
    for c in range(4):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c] * 100
            f.write(f"  {CLASS_NAMES[c]:12s}: {acc:6.2f}%\n")
    f.write(f"  {'OVERALL':12s}: {overall:6.2f}%\n\n")
    f.write("DENOISING QUALITY:\n")
    f.write(f"  SNR Improvement: {np.mean(snr_imps):.2f} dB\n")
    f.write(f"  NCC (Shape):     {np.mean(nccs):.4f}\n")
    f.write(f"  RMSE:            {np.mean(rmses):.6f}\n")

print(f"\nResults saved to: {SAVE_DIR / 'final_eval_results.txt'}")
