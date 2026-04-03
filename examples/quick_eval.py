#!/usr/bin/env python
"""Quick evaluation script that writes to file."""

import sys
sys.path.insert(0, 'D:/New folder (2)')
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc

DEVICE = 'cuda'
SAVE_DIR = Path('D:/New folder (2)/outputs/extended_training')

# Load model
config = get_config()
model = create_model(config.model).to(DEVICE)
ckpt = torch.load(SAVE_DIR / 'sota_best.pth', map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

val_dataset = PDSignalDataset(config=config, num_samples=300, mode='val', epoch=50, seed=888)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Per-class stats
class_correct = [0, 0, 0, 0]
class_total = [0, 0, 0, 0]
CLASS_NAMES = ['Background', 'Corona', 'Surface', 'Internal']

snr_imps = []
nccs = []
spike_samples = 0

with torch.no_grad():
    for batch in val_loader:
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
            
            # Denoising metrics
            noisy_np = noisy[i, 0].cpu().numpy()
            clean_np = clean[i, 0].cpu().numpy()
            denoised_np = denoised[i, 0].cpu().numpy()
            
            if np.abs(denoised_np).max() > 3:
                spike_samples += 1
            
            try:
                snr_imp = calculate_snr_improvement(noisy_np, denoised_np, clean_np)
                ncc = calculate_ncc(denoised_np, clean_np)
                snr_imps.append(snr_imp)
                nccs.append(ncc)
            except:
                pass

# Write results
with open(SAVE_DIR / 'quick_eval_results.txt', 'w') as f:
    f.write('='*65 + '\n')
    f.write('SEGMENTATION CLASSIFICATION ACCURACY\n')
    f.write('='*65 + '\n')
    for c in range(4):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c] * 100
            f.write(f'{CLASS_NAMES[c]:12s}: {acc:6.2f}% ({class_correct[c]:,} / {class_total[c]:,} pixels)\n')
        else:
            f.write(f'{CLASS_NAMES[c]:12s}: N/A (no samples)\n')
    
    overall_acc = sum(class_correct) / sum(class_total) * 100
    f.write('-'*65 + '\n')
    f.write(f'OVERALL:      {overall_acc:.2f}%\n')
    
    f.write('\n')
    f.write('='*65 + '\n')
    f.write('DENOISING QUALITY\n')
    f.write('='*65 + '\n')
    f.write(f'SNR Improvement: {np.mean(snr_imps):.2f} +/- {np.std(snr_imps):.2f} dB\n')
    f.write(f'NCC (Shape):     {np.mean(nccs):.4f} +/- {np.std(nccs):.4f}\n')
    f.write(f'Spike Samples:   {spike_samples} / 300 ({spike_samples/300*100:.1f}%)\n')
    f.write('='*65 + '\n')

print("Results saved to:", SAVE_DIR / 'quick_eval_results.txt')
