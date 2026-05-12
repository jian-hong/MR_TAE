"""Test new SOTA features."""
import torch
print("Testing new imports...")

# Test losses
from mr_tae_fusion.training import CharbonnierLoss, JointLoss, JointLossWithUncertainty
print("Losses: OK")

# Test TGAN
from mr_tae_fusion.models import NoiseGenerator, NoiseDiscriminator, TGAN
print("TGAN: OK")

# Test complete integration
from mr_tae_fusion.models import create_model
from mr_tae_fusion.config import get_config

config = get_config()
model = create_model(config.model)

# Test with JointLoss
joint_loss = JointLoss()

x = torch.randn(2, 1, 2001)
with torch.no_grad():
    denoised, seg = model(x)

# Dummy losses
target = torch.randn_like(denoised)
mask = torch.randint(0, 4, (2, 2001))

loss, loss_dict = joint_loss(denoised, target, seg, mask)
print(f"JointLoss: {loss.item():.4f}")
print(f"  - Denoise: {loss_dict['denoise']:.4f}")
print(f"  - Seg: {loss_dict['seg']:.4f}")

# Test TGAN forward pass
print("\nTesting TGAN...")
gen = NoiseGenerator(noise_dim=100, seq_len=2001)
disc = NoiseDiscriminator(seq_len=2001)

z = torch.randn(2, 100)
fake_noise = gen(z)
score = disc(fake_noise)

print(f"Generator output: {fake_noise.shape}")
print(f"Discriminator score: {score.shape}")

print("\n=== ALL TESTS PASSED! ===")
