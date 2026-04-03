#!/usr/bin/env python
"""
train_tgan.py - Train TGAN on real noise data from Q.Lin dataset.

This script extracts noise segments from the Q.Lin dataset and trains
a WGAN-GP to generate realistic noise patterns for data augmentation.

Usage:
    python train_tgan.py --data_dir "Data/OneDrive_2025-12-13/Datset from Q.lin"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from mr_tae_fusion.models.tgan import TGAN, NoiseGenerator, NoiseDiscriminator
from mr_tae_fusion.data.mat_loader import prepare_tgan_training_data
from mr_tae_fusion.data.dataset import RealNoiseDataset


def train_tgan(
    data_dir: str,
    output_dir: str = "checkpoints/tgan",
    epochs: int = 100,
    batch_size: int = 32,
    noise_dim: int = 100,
    seq_len: int = 2001,
    n_critic: int = 5,  # Train D n times per G train
    device: str = 'auto',
    save_every: int = 10
):
    """
    Train TGAN on real noise data.
    
    Args:
        data_dir: Path to Q.Lin dataset directory
        output_dir: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        noise_dim: Generator noise dimension
        seq_len: Signal sequence length
        n_critic: Critic training steps per generator step
        device: Device to train on
        save_every: Save checkpoint every N epochs
    """
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load real noise data
    print(f"\nLoading noise data from: {data_dir}")
    try:
        noise_segments = prepare_tgan_training_data(
            data_dir,
            segment_length=seq_len,
            max_segments_per_file=1000
        )
        print(f"Loaded {len(noise_segments)} noise segments")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic noise for demonstration...")
        # Fallback: generate synthetic noise for demo
        noise_segments = np.random.randn(1000, seq_len).astype(np.float32) * 0.1
    
    # Normalize noise to [-1, 1] for tanh output
    noise_max = np.abs(noise_segments).max()
    noise_segments = noise_segments / (noise_max + 1e-8)
    
    # Create dataset and loader
    dataset = RealNoiseDataset(noise_segments)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Create TGAN
    tgan = TGAN(
        noise_dim=noise_dim,
        seq_len=seq_len,
        device=device
    )
    
    print(f"\nGenerator params: {sum(p.numel() for p in tgan.generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in tgan.discriminator.parameters()):,}")
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = {'loss_D': [], 'loss_G': [], 'D_real': [], 'D_fake': [], 'gp': []}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, real_batch in enumerate(pbar):
            # Train Discriminator (n_critic times)
            for _ in range(n_critic):
                d_loss = tgan.train_step_discriminator(real_batch)
                epoch_losses['loss_D'].append(d_loss['loss_D'])
                epoch_losses['D_real'].append(d_loss['D_real'])
                epoch_losses['D_fake'].append(d_loss['D_fake'])
                epoch_losses['gp'].append(d_loss['gp'])
            
            # Train Generator
            g_loss = tgan.train_step_generator(batch_size)
            epoch_losses['loss_G'].append(g_loss['loss_G'])
            
            # Update progress bar
            pbar.set_postfix({
                'D': f"{np.mean(epoch_losses['loss_D'][-10:]):.4f}",
                'G': f"{np.mean(epoch_losses['loss_G'][-10:]):.4f}"
            })
        
        # Log epoch stats
        print(f"\nEpoch {epoch + 1} - "
              f"D: {np.mean(epoch_losses['loss_D']):.4f} | "
              f"G: {np.mean(epoch_losses['loss_G']):.4f} | "
              f"D(real): {np.mean(epoch_losses['D_real']):.4f} | "
              f"D(fake): {np.mean(epoch_losses['D_fake']):.4f}")
        
        # Save checkpoints
        if (epoch + 1) % save_every == 0:
            tgan.save(str(output_dir / f"tgan_epoch_{epoch + 1}.pt"))
            print(f"Saved checkpoint at epoch {epoch + 1}")
    
    # Save final model
    tgan.save(str(output_dir / "tgan_final.pt"))
    tgan.save_generator(str(output_dir / "tgan_generator.pth"))
    
    print(f"\nTraining complete!")
    print(f"Generator saved to: {output_dir / 'tgan_generator.pth'}")
    
    # Generate sample for verification
    print("\nGenerating sample noise...")
    with torch.no_grad():
        z = torch.randn(1, noise_dim, device=device)
        sample = tgan.generator(z).cpu().numpy().squeeze()
    
    # Save sample
    np.save(str(output_dir / "sample_generated_noise.npy"), sample)
    print(f"Sample saved to: {output_dir / 'sample_generated_noise.npy'}")
    
    return tgan


def main():
    parser = argparse.ArgumentParser(description="Train TGAN on real noise data")
    parser.add_argument('--data_dir', type=str, 
                        default="Data/OneDrive_2025-12-13/Datset from Q.lin",
                        help='Path to Q.Lin dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints/tgan',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--noise_dim', type=int, default=100,
                        help='Generator noise dimension')
    parser.add_argument('--seq_len', type=int, default=2001,
                        help='Sequence length')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Critic steps per generator step')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train_tgan(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        seq_len=args.seq_len,
        n_critic=args.n_critic,
        device=args.device,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
