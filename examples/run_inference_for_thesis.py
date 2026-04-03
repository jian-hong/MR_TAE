#!/usr/bin/env python
"""
run_inference_for_thesis.py - Run MR-TAE-Fusion inference on thesis data.

Loads noisy signals and runs the trained model to generate denoised outputs.
Results are saved as .mat files for MATLAB comparison.

Usage:
    python run_inference_for_thesis.py --data_dir Thesis_Data --model_path checkpoints/best_model.pth
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from mr_tae_fusion.config import get_config
from mr_tae_fusion.models import create_model


def run_inference(
    data_dir: str,
    model_path: str,
    output_dir: str = None,
    snr_levels: list = [-5, -10, -15, -20],
    device: str = 'auto',
    batch_size: int = 16
):
    """
    Run model inference on thesis data.
    
    Args:
        data_dir: Directory with noisy data files
        model_path: Path to trained model weights
        output_dir: Output directory (default: same as data_dir)
        snr_levels: SNR levels to process
        device: Device for inference
        batch_size: Batch size for inference
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    config = get_config()
    model = create_model(config.model)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.count_parameters():,} parameters")
    
    # Process each SNR level
    for snr in snr_levels:
        print(f"\n{'='*50}")
        print(f"Processing SNR = {snr} dB")
        print('='*50)
        
        # Load noisy data
        filename = f"Noisy_SNR_{abs(snr)}"
        
        if HAS_SCIPY and (data_dir / f"{filename}.mat").exists():
            data = sio.loadmat(str(data_dir / f"{filename}.mat"))
            noisy_signals = data['noisy_signals']
        elif (data_dir / f"{filename}.npz").exists():
            data = np.load(str(data_dir / f"{filename}.npz"))
            noisy_signals = data['noisy_signals']
        else:
            print(f"  File not found: {filename}.mat or {filename}.npz")
            continue
        
        print(f"  Loaded {len(noisy_signals)} signals, shape: {noisy_signals.shape}")
        
        # Run inference
        denoised_batch = []
        segmentation_batch = []
        
        num_signals = len(noisy_signals)
        num_batches = (num_signals + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Inference"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_signals)
                
                # Prepare batch
                batch = noisy_signals[start_idx:end_idx]
                input_tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)
                
                # Forward pass
                denoised, seg_logits = model(input_tensor)
                
                # Convert to numpy
                denoised_np = denoised.squeeze(1).cpu().numpy()
                seg_np = seg_logits.argmax(dim=1).cpu().numpy()
                
                denoised_batch.append(denoised_np)
                segmentation_batch.append(seg_np)
        
        # Concatenate results
        denoised_batch = np.concatenate(denoised_batch, axis=0)
        segmentation_batch = np.concatenate(segmentation_batch, axis=0)
        
        print(f"  Denoised shape: {denoised_batch.shape}")
        print(f"  Segmentation shape: {segmentation_batch.shape}")
        
        # Save results
        output_filename = f"Denoised_DL_SNR_{abs(snr)}"
        
        if HAS_SCIPY:
            sio.savemat(
                str(output_dir / f"{output_filename}.mat"),
                {
                    "denoised_signals": denoised_batch,
                    "segmentation_masks": segmentation_batch,
                    "snr": snr
                }
            )
            print(f"  Saved: {output_filename}.mat")
        else:
            np.savez(
                str(output_dir / f"{output_filename}.npz"),
                denoised_signals=denoised_batch,
                segmentation_masks=segmentation_batch,
                snr=snr
            )
            print(f"  Saved: {output_filename}.npz")
    
    print("\n" + "="*50)
    print("Inference complete!")
    print(f"Results saved to: {output_dir}")
    print("="*50)


def calculate_metrics(
    data_dir: str,
    snr_levels: list = [-5, -10, -15, -20]
):
    """
    Calculate and print comparison metrics.
    
    Args:
        data_dir: Directory with clean, noisy, and denoised data
        snr_levels: SNR levels to evaluate
    """
    from mr_tae_fusion.evaluation import calculate_snr_improvement, calculate_ncc, calculate_rmse
    
    data_dir = Path(data_dir)
    
    # Load clean signals
    if HAS_SCIPY and (data_dir / "Clean_Signals.mat").exists():
        data = sio.loadmat(str(data_dir / "Clean_Signals.mat"))
        clean_signals = data['clean_signals']
    else:
        data = np.load(str(data_dir / "Clean_Signals.npz"))
        clean_signals = data['clean_signals']
    
    print("\n" + "="*70)
    print("Performance Comparison: MR-TAE-Fusion")
    print("="*70)
    print(f"{'SNR Input':<12} | {'SNR Output':<12} | {'SNR Imp':<10} | {'NCC':<8} | {'RMSE':<8}")
    print("-"*70)
    
    for snr in snr_levels:
        # Load denoised
        filename = f"Denoised_DL_SNR_{abs(snr)}"
        
        if HAS_SCIPY and (data_dir / f"{filename}.mat").exists():
            data = sio.loadmat(str(data_dir / f"{filename}.mat"))
            denoised = data['denoised_signals']
        elif (data_dir / f"{filename}.npz").exists():
            data = np.load(str(data_dir / f"{filename}.npz"))
            denoised = data['denoised_signals']
        else:
            print(f"{snr:<12} | Not found")
            continue
        
        # Load noisy for comparison
        noisy_file = f"Noisy_SNR_{abs(snr)}"
        if HAS_SCIPY and (data_dir / f"{noisy_file}.mat").exists():
            noisy_data = sio.loadmat(str(data_dir / f"{noisy_file}.mat"))
            noisy = noisy_data['noisy_signals']
        else:
            noisy_data = np.load(str(data_dir / f"{noisy_file}.npz"))
            noisy = noisy_data['noisy_signals']
        
        # Calculate metrics
        snr_improvements = []
        nccs = []
        rmses = []
        
        for i in range(len(clean_signals)):
            try:
                snr_imp = calculate_snr_improvement(noisy[i], denoised[i], clean_signals[i])
                ncc = calculate_ncc(denoised[i], clean_signals[i])
                rmse = calculate_rmse(denoised[i], clean_signals[i])
                
                snr_improvements.append(snr_imp)
                nccs.append(ncc)
                rmses.append(rmse)
            except:
                continue
        
        avg_snr_imp = np.mean(snr_improvements)
        avg_snr_out = snr + avg_snr_imp
        avg_ncc = np.mean(nccs)
        avg_rmse = np.mean(rmses)
        
        print(f"{snr:<12} | {avg_snr_out:<12.2f} | {avg_snr_imp:<10.2f} | {avg_ncc:<8.4f} | {avg_rmse:<8.4f}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Run MR-TAE-Fusion inference")
    parser.add_argument('--data_dir', type=str, default='Thesis_Data',
                        help='Directory with thesis data')
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as data_dir)')
    parser.add_argument('--snr_levels', type=int, nargs='+',
                        default=[-5, -10, -15, -20],
                        help='SNR levels to process')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--calc_metrics', action='store_true',
                        help='Calculate and print metrics after inference')
    
    args = parser.parse_args()
    
    run_inference(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        snr_levels=args.snr_levels,
        device=args.device,
        batch_size=args.batch_size
    )
    
    if args.calc_metrics:
        output_dir = args.output_dir if args.output_dir else args.data_dir
        calculate_metrics(output_dir, args.snr_levels)


if __name__ == '__main__':
    main()
