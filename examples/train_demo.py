#!/usr/bin/env python
"""
MR-TAE-Fusion Training Demo

Demonstrates training the MR-TAE-Fusion model with curriculum learning.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

from mr_tae_fusion.config import Config, get_config
from mr_tae_fusion.models import MRTAEFusion, create_model
from mr_tae_fusion.data import PDSignalDataset
from mr_tae_fusion.training import Trainer, MultiTaskLoss
from mr_tae_fusion.evaluation import (
    evaluate_denoising,
    evaluate_segmentation,
    PDMetrics,
    plot_reconstruction,
    plot_segmentation_overlay,
    plot_training_history
)


def main():
    parser = argparse.ArgumentParser(description='MR-TAE-Fusion Training Demo')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--train-samples', type=int, default=500,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, or auto)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = get_config()
    config.training.total_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.device = device
    config.training.seed = args.seed
    
    print("\n" + "="*60)
    print("MR-TAE-Fusion Training Demo")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training samples: {args.train_samples}")
    print(f"Validation samples: {args.val_samples}")
    
    # Create model
    print("\n[1/4] Creating model...")
    model = create_model(config.model)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    # Print model summary
    summary = model.get_model_summary()
    for component, params in summary.items():
        print(f"  {component}: {params:,}")
    
    # Create datasets
    print("\n[2/4] Creating datasets...")
    train_dataset = PDSignalDataset(
        config=config,
        num_samples=args.train_samples,
        mode='train',
        seed=args.seed
    )
    
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=args.val_samples,
        mode='val',
        seed=args.seed + 1
    )
    
    # Test forward pass
    print("\n[3/4] Testing forward pass...")
    sample = train_dataset[0]
    noisy = sample['noisy'].unsqueeze(0).to(device)
    model = model.to(device)
    
    with torch.no_grad():
        denoised, segmentation = model(noisy)
    
    print(f"Input shape:  {noisy.shape}")
    print(f"Output denoised: {denoised.shape}")
    print(f"Output segmentation: {segmentation.shape}")
    
    # Training
    print("\n[4/4] Starting training...")
    trainer = Trainer(
        model=model,
        config=config,
        save_dir=str(output_dir / 'checkpoints')
    )
    
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs
    )
    
    # Plot training history
    print("\n[5/5] Generating visualizations...")
    fig = plot_training_history(history, save_path=str(output_dir / 'training_history.png'))
    plt.close(fig)
    
    # Evaluate on sample
    print("\nEvaluating on sample...")
    model.eval()
    
    test_sample = val_dataset[0]
    noisy = test_sample['noisy'].unsqueeze(0).to(device)
    clean = test_sample['clean'].unsqueeze(0).to(device)
    mask = test_sample['mask'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        denoised, seg_logits = model(noisy)
    
    # Calculate metrics
    denoising_metrics = evaluate_denoising(
        noisy.cpu().numpy(),
        denoised.cpu().numpy(),
        clean.cpu().numpy()
    )
    
    seg_metrics = evaluate_segmentation(
        seg_logits.cpu(),
        mask.cpu(),
        num_classes=config.signal.num_classes
    )
    
    # Create metrics object
    metrics = PDMetrics(
        snr_input=denoising_metrics['snr_input'],
        snr_output=denoising_metrics['snr_output'],
        snr_improvement=denoising_metrics['snr_improvement'],
        ncc=denoising_metrics['ncc'],
        rmse=denoising_metrics['rmse'],
        mean_iou=seg_metrics['mean_iou'],
        mean_dice=seg_metrics['mean_dice'],
        pulse_detection_rate=seg_metrics['pulse_detection_rate']
    )
    
    print("\n" + str(metrics))
    
    # Plot reconstruction
    fig = plot_reconstruction(
        noisy.squeeze().cpu().numpy(),
        denoised.squeeze().cpu().numpy(),
        clean.squeeze().cpu().numpy(),
        title=f"Sample Reconstruction (SNR imp: {metrics.snr_improvement:.1f} dB)",
        save_path=str(output_dir / 'reconstruction.png')
    )
    plt.close(fig)
    
    # Plot segmentation
    pred_mask = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
    fig = plot_segmentation_overlay(
        clean.squeeze().cpu().numpy(),
        pred_mask,
        mask.squeeze().cpu().numpy(),
        title="Segmentation Results",
        save_path=str(output_dir / 'segmentation.png')
    )
    plt.close(fig)
    
    print(f"\nResults saved to: {output_dir}")
    print("\nTraining demo complete!")


if __name__ == '__main__':
    main()
