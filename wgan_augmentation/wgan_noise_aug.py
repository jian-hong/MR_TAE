"""WGAN-GP noise augmentation module with anti-overfitting controls."""

# NOTE: This module is constrained to fit WGAN only on the 10% real-data training split.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NoiseWGANConfig:
    latent_dim: int = 64
    signal_len: int = 2048
    gp_lambda: float = 10.0
    feature_match_weight: float = 0.1
    mmd_patience: int = 10
    real_train_fraction: float = 0.10


class NoiseGenerator(nn.Module):
    def __init__(self, z_dim: int = 64, out_len: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_len),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).unsqueeze(1)


class NoiseDiscriminator(nn.Module):
    def __init__(self, signal_len: int = 2048):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv1d(1, 32, kernel_size=7, padding=3))
        self.conv2 = nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=5, padding=2))
        self.conv3 = nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=3, padding=1))
        self.head = nn.utils.spectral_norm(nn.Linear(128 * signal_len, 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.head(feat.flatten(1))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    kxx = torch.exp(-xx / (2 * sigma * sigma)).mean()
    kyy = torch.exp(-yy / (2 * sigma * sigma)).mean()
    kxy = torch.exp(-xy / (2 * sigma * sigma)).mean()
    return kxx + kyy - 2 * kxy
