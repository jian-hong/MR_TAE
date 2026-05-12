"""Conditional pulse-shape WGAN module with anti-overfitting controls."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PulseWGANConfig:
    latent_dim: int = 64
    n_classes: int = 3
    signal_len: int = 2048
    gp_lambda: float = 10.0
    feature_match_weight: float = 0.1
    mmd_patience: int = 10
    real_train_fraction: float = 0.10


class ConditionalPulseGenerator(nn.Module):
    def __init__(self, z_dim: int = 64, n_classes: int = 3, out_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(n_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 16, 512),
            nn.GELU(),
            nn.Linear(512, out_len),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.embed(labels)
        x = torch.cat([z, cond], dim=1)
        return self.net(x).unsqueeze(1)


class ConditionalPulseDiscriminator(nn.Module):
    def __init__(self, n_classes: int = 3, signal_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(n_classes, signal_len)
        self.conv1 = nn.utils.spectral_norm(nn.Conv1d(2, 32, kernel_size=7, padding=3))
        self.conv2 = nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=5, padding=2))
        self.conv3 = nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=3, padding=1))
        self.head = nn.utils.spectral_norm(nn.Linear(128 * signal_len, 1))

    def features(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.embed(labels).unsqueeze(1)
        x = torch.cat([x, cond], dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = self.features(x, labels)
        return self.head(feat.flatten(1))
