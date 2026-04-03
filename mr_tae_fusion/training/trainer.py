"""
Training loop and utilities for MR-TAE-Fusion.

Implements curriculum learning, logging, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from tqdm import tqdm
import json

from .losses import MultiTaskLoss
from ..config import Config, TrainingConfig
from ..data.dataset import PDSignalDataset


@dataclass
class TrainingState:
    """Holds training state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    curriculum_phase: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'curriculum_phase': self.curriculum_phase,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingState':
        return cls(**d)


class Trainer:
    """
    Trainer for MR-TAE-Fusion with curriculum learning.
    
    Handles:
    - Multi-task loss optimization
    - Curriculum learning phases (easy → hard noise)
    - Validation and checkpointing
    - Learning rate scheduling
    
    Args:
        model: MR-TAE-Fusion model
        config: Training configuration
        save_dir: Directory for checkpoints
        device: Training device
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Config] = None,
        save_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        if config is None:
            config = Config()
        
        self.config = config
        self.train_config = config.training
        self.device = device or self.train_config.device
        
        # Model
        self.model = model.to(self.device)
        
        # Loss function with uncertainty weighting
        self.criterion = MultiTaskLoss(
            initial_sigma_recon=self.train_config.initial_sigma_recon,
            initial_sigma_seg=self.train_config.initial_sigma_seg,
            dice_weight=self.train_config.dice_weight
        ).to(self.device)
        
        # Optimizer (include loss parameters for uncertainty learning)
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=self.train_config.lr_min
        )
        
        # Save directory
        self.save_dir = Path(save_dir) if save_dir else Path('checkpoints')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.state = TrainingState()
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if 'cuda' in self.device else None
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'recon': 0.0, 
            'seg': 0.0,
            'dice': 0.0,
        }
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_signal, pred_seg = self.model(noisy)
                    loss, loss_dict = self.criterion(
                        pred_signal, clean, pred_seg, mask
                    )
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.gradient_clip_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_signal, pred_seg = self.model(noisy)
                loss, loss_dict = self.criterion(
                    pred_signal, clean, pred_seg, mask
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.gradient_clip_norm
                )
                
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            loss_components['recon'] += loss_dict['recon'].item()
            loss_components['seg'] += loss_dict['seg'].item()
            loss_components['dice'] += loss_dict['dice'].item()
            num_batches += 1
            
            self.state.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['recon'].item():.4f}",
                'seg': f"{loss_dict['seg'].item():.4f}",
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # Get task weights
        weight_recon, weight_seg = self.criterion.get_task_weights()
        
        return {
            'loss': avg_loss,
            **loss_components,
            'weight_recon': weight_recon,
            'weight_seg': weight_seg,
        }
    
    @torch.no_grad()
    def validate(
        self, 
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {'recon': 0.0, 'seg': 0.0, 'dice': 0.0}
        num_batches = 0
        
        for batch in val_loader:
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            pred_signal, pred_seg = self.model(noisy)
            loss, loss_dict = self.criterion(
                pred_signal, clean, pred_seg, mask
            )
            
            total_loss += loss.item()
            loss_components['recon'] += loss_dict['recon'].item()
            loss_components['seg'] += loss_dict['seg'].item()
            loss_components['dice'] += loss_dict['dice'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'loss': avg_loss, **loss_components}
    
    def train(
        self,
        train_dataset: PDSignalDataset,
        val_dataset: PDSignalDataset,
        num_epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict:
        """
        Full training loop with curriculum learning.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs (overrides config)
            callbacks: Optional callbacks called after each epoch
        
        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.train_config.total_epochs
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'train_seg': [],
            'val_recon': [],
            'val_seg': [],
            'task_weights': [],
            'curriculum_phase': [],
        }
        
        patience_counter = 0
        
        for epoch in range(self.state.epoch, num_epochs):
            self.state.epoch = epoch
            
            # Update curriculum phase in datasets
            train_dataset.update_epoch(epoch)
            val_dataset.update_epoch(epoch)
            
            curriculum_info = train_dataset.get_curriculum_info()
            self.state.curriculum_phase = curriculum_info['phase']
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs} | Phase {curriculum_info['phase']} | "
                  f"SNR: {curriculum_info['snr_range']} | Noise: {curriculum_info['noise_type']}")
            print(f"{'='*60}")
            
            # Create loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Recon: {train_metrics['recon']:.4f}/{val_metrics['recon']:.4f} | "
                  f"Seg: {train_metrics['seg']:.4f}/{val_metrics['seg']:.4f}")
            print(f"Task Weights - Recon: {train_metrics['weight_recon']:.3f} | "
                  f"Seg: {train_metrics['weight_seg']:.3f}")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_recon'].append(train_metrics['recon'])
            history['train_seg'].append(train_metrics['seg'])
            history['val_recon'].append(val_metrics['recon'])
            history['val_seg'].append(val_metrics['seg'])
            history['task_weights'].append({
                'recon': train_metrics['weight_recon'],
                'seg': train_metrics['weight_seg']
            })
            history['curriculum_phase'].append(curriculum_info['phase'])
            
            # Check for improvement
            if val_metrics['loss'] < self.state.best_val_loss:
                self.state.best_val_loss = val_metrics['loss']
                self.state.best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt')
                print(f"✓ New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.train_config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics, self.model)
        
        print(f"\nTraining complete! Best val loss: {self.state.best_val_loss:.4f} "
              f"at epoch {self.state.best_epoch + 1}")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'training_state': self.state.to_dict(),
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
            }
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.state = TrainingState.from_dict(checkpoint['training_state'])
        
        print(f"Loaded checkpoint from epoch {self.state.epoch}")


def train_model(
    model: nn.Module,
    config: Config,
    train_samples: int = 3000,
    val_samples: int = 500,
    save_dir: str = 'checkpoints'
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train MR-TAE-Fusion model.
    
    Args:
        model: Initialized model
        config: Configuration
        train_samples: Number of training samples
        val_samples: Number of validation samples
        save_dir: Directory for checkpoints
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create datasets
    train_dataset = PDSignalDataset(
        config=config,
        num_samples=train_samples,
        mode='train',
        seed=config.training.seed
    )
    
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=val_samples,
        mode='val',
        seed=config.training.seed + 1
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        save_dir=save_dir
    )
    
    # Train
    history = trainer.train(train_dataset, val_dataset)
    
    return model, history
