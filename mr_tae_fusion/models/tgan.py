"""
TGAN: Time-series Generative Adversarial Network for realistic noise generation.

Implements WGAN-GP (Wasserstein GAN with Gradient Penalty) for 1D signals.
Used to generate realistic noise learned from Q.Lin dataset for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Tuple, Optional


class NoiseGenerator(nn.Module):
    """
    TGAN Generator: Generates realistic 1D noise signals.
    
    Takes a random noise vector and upsamples it to a full-length signal
    using transposed convolutions.
    
    Architecture: Linear -> Reshape -> ConvTranspose1D blocks -> Final Conv
    
    Args:
        noise_dim: Dimension of input noise vector (z)
        seq_len: Target output sequence length
        base_channels: Base number of channels (doubles at each layer)
    """
    
    def __init__(
        self, 
        noise_dim: int = 100, 
        seq_len: int = 2001,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.dim = base_channels
        
        # Calculate initial size for upsampling
        # We'll upsample through multiple stages
        self.init_size = 25  # Initial spatial size
        self.final_upsample = seq_len
        
        # Project noise to initial feature map
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, self.dim * 8 * self.init_size),
            nn.BatchNorm1d(self.dim * 8 * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Upsample blocks: 25 -> 100 -> 500 -> 2000+
        self.upsample = nn.Sequential(
            # Block 1: 25 -> 100
            nn.ConvTranspose1d(self.dim * 8, self.dim * 4, 
                              kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: 100 -> 500
            nn.ConvTranspose1d(self.dim * 4, self.dim * 2,
                              kernel_size=5, stride=5, padding=0),
            nn.BatchNorm1d(self.dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: 500 -> 2000
            nn.ConvTranspose1d(self.dim * 2, self.dim,
                              kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final refinement + output
            nn.Conv1d(self.dim, self.dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(self.dim // 2, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate noise signal from random vector.
        
        Args:
            z: Random noise vector (B, noise_dim) or (B, noise_dim, 1)
        
        Returns:
            Generated noise signal (B, 1, seq_len)
        """
        # Handle different input shapes
        if z.dim() == 3:
            z = z.squeeze(-1)
        
        # Project to initial feature map
        x = self.fc(z)
        x = x.view(-1, self.dim * 8, self.init_size)
        
        # Upsample
        x = self.upsample(x)
        
        # Interpolate to exact target length if needed
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        
        return x


class NoiseDiscriminator(nn.Module):
    """
    TGAN Discriminator: Classifies real vs fake noise signals.
    
    Uses strided convolutions to downsample, outputs a scalar score.
    For WGAN-GP, this is a critic (no sigmoid at output).
    
    Args:
        seq_len: Input sequence length
        base_channels: Base number of channels
    """
    
    def __init__(
        self, 
        seq_len: int = 2001,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.dim = base_channels
        
        self.features = nn.Sequential(
            # Block 1: 2001 -> ~500
            nn.Conv1d(1, self.dim, kernel_size=4, stride=4, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: ~500 -> ~125
            nn.Conv1d(self.dim, self.dim * 2, kernel_size=4, stride=4, padding=1),
            nn.InstanceNorm1d(self.dim * 2),  # InstanceNorm for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: ~125 -> ~25
            nn.Conv1d(self.dim * 2, self.dim * 4, kernel_size=5, stride=5, padding=0),
            nn.InstanceNorm1d(self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4: ~25 -> ~5
            nn.Conv1d(self.dim * 4, self.dim * 8, kernel_size=5, stride=5, padding=0),
            nn.InstanceNorm1d(self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Adaptive pooling for any input size, then FC
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.dim * 8, 1)
            # No sigmoid for WGAN
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute critic score for input signal.
        
        Args:
            x: Input signal (B, 1, L) or (B, L)
        
        Returns:
            Critic score (B, 1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = self.features(x)
        score = self.classifier(features)
        return score


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Enforces Lipschitz constraint by penalizing gradients of the critic.
    
    Args:
        discriminator: Discriminator/Critic network
        real_samples: Real noise samples (B, 1, L)
        fake_samples: Generated fake samples (B, 1, L)
        device: Torch device
    
    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.size(0)
    
    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Interpolated samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)
    
    # Critic output for interpolates
    d_interpolates = discriminator(interpolates)
    
    # Compute gradients
    fake = torch.ones(batch_size, 1, device=device, requires_grad=False)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Gradient penalty: (||grad|| - 1)^2
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


class TGANNoiseLoader:
    """
    Utility class to load pre-trained TGAN and generate noise on-the-fly.
    
    Use this in your data pipeline after training the TGAN on real noise data.
    
    Args:
        weights_path: Path to trained generator weights
        noise_dim: Noise vector dimension (must match training)
        seq_len: Output sequence length
        device: Device to run on
    """
    
    def __init__(
        self,
        weights_path: str,
        noise_dim: int = 100,
        seq_len: int = 2001,
        device: str = 'cuda'
    ):
        self.device = device
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        
        # Load pre-trained generator
        self.generator = NoiseGenerator(
            noise_dim=noise_dim,
            seq_len=seq_len
        ).to(device)
        
        self.generator.load_state_dict(torch.load(weights_path, map_location=device))
        self.generator.eval()
    
    @torch.no_grad()
    def get_batch_noise(self, batch_size: int) -> torch.Tensor:
        """
        Generate a batch of realistic noise signals.
        
        Args:
            batch_size: Number of noise samples to generate
        
        Returns:
            Generated noise (batch_size, seq_len) as numpy array
        """
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        generated = self.generator(z)
        return generated.squeeze(1).cpu().numpy()
    
    @torch.no_grad()
    def get_single_noise(self) -> torch.Tensor:
        """Generate a single noise sample."""
        return self.get_batch_noise(1)[0]


class TGAN:
    """
    Complete TGAN system for training and generation.
    
    Wraps generator, discriminator, and training logic.
    
    Args:
        noise_dim: Dimension of noise vector
        seq_len: Sequence length of signals
        base_channels: Base channel count
        device: Device to train on
        lambda_gp: Gradient penalty weight
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        seq_len: int = 2001,
        base_channels: int = 64,
        device: str = 'cuda',
        lambda_gp: float = 10.0
    ):
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.lambda_gp = lambda_gp
        self.device = torch.device(device)
        
        # Networks
        self.generator = NoiseGenerator(noise_dim, seq_len, base_channels).to(self.device)
        self.discriminator = NoiseDiscriminator(seq_len, base_channels).to(self.device)
        
        # Optimizers (WGAN-GP uses lower lr for stability)
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), 
            lr=1e-4, betas=(0.0, 0.9)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=1e-4, betas=(0.0, 0.9)
        )
    
    def train_step_discriminator(self, real_samples: torch.Tensor) -> dict:
        """Train discriminator for one step."""
        batch_size = real_samples.size(0)
        real_samples = real_samples.to(self.device)
        
        if real_samples.dim() == 2:
            real_samples = real_samples.unsqueeze(1)
        
        self.optimizer_D.zero_grad()
        
        # Generate fake samples
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_samples = self.generator(z).detach()
        
        # Critic scores
        D_real = self.discriminator(real_samples)
        D_fake = self.discriminator(fake_samples)
        
        # Gradient penalty
        gp = compute_gradient_penalty(
            self.discriminator, real_samples, fake_samples, self.device
        )
        
        # WGAN-GP loss: maximize D(real) - D(fake) with GP
        loss_D = D_fake.mean() - D_real.mean() + self.lambda_gp * gp
        
        loss_D.backward()
        self.optimizer_D.step()
        
        return {
            'loss_D': loss_D.item(),
            'D_real': D_real.mean().item(),
            'D_fake': D_fake.mean().item(),
            'gp': gp.item()
        }
    
    def train_step_generator(self, batch_size: int) -> dict:
        """Train generator for one step."""
        self.optimizer_G.zero_grad()
        
        # Generate fake samples
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_samples = self.generator(z)
        
        # Critic score for fake
        D_fake = self.discriminator(fake_samples)
        
        # Generator loss: maximize D(fake) = minimize -D(fake)
        loss_G = -D_fake.mean()
        
        loss_G.backward()
        self.optimizer_G.step()
        
        return {'loss_G': loss_G.item()}
    
    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    
    def save_generator(self, path: str):
        """Save only generator weights (for inference)."""
        torch.save(self.generator.state_dict(), path)
