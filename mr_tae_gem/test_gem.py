"""Quick test to verify MR-TAE-GEM imports and curriculum."""
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

print("Testing MR-TAE-GEM imports...")

# Test 1: Colored noise
from colored_noise import FlickerNoise, SubstationNoise, ColoredNoiseGenerator, add_colored_noise_at_snr
import numpy as np

f = FlickerNoise()
noise = f.generate(2001)
print(f"  ✓ FlickerNoise: len={len(noise)}, std={noise.std():.4f}")

s = SubstationNoise()
noise = s.generate(2001)
print(f"  ✓ SubstationNoise: len={len(noise)}, std={noise.std():.4f}")

c = ColoredNoiseGenerator(noise_type='composite')
noise = c.generate(2001)
print(f"  ✓ ColoredNoiseGenerator: len={len(noise)}, std={noise.std():.4f}")

# Test 2: Curriculum phases
from gem_dataset import get_overlapping_curriculum_phase

print("\nTesting overlapping curriculum phases:")
print("  (No 'pure noise' phase - always includes PD)")
for ep in [0, 50, 150, 300, 450, 600, 680]:
    p = get_overlapping_curriculum_phase(ep, 700)
    print(f"  Epoch {ep:3d}: Phase {p['phase']} - {p['name']:15s} SNR:{p['snr_range']} pd_inc:{p['pd_inclusion']}")

# Test 3: Dataset creation
print("\nTesting GEMDataset creation...")
from mr_tae_fusion.config import get_config
from gem_dataset import GEMDataset

config = get_config()
config.signal.num_classes = 5

dataset = GEMDataset(
    config=config,
    num_samples=100,
    mode='train',
    epoch=0,
    total_epochs=700,
    qlin_loader=None,
    use_replay=True
)
print(f"  ✓ GEMDataset created: {len(dataset)} samples")

# Test epoch update (the Validation Mirage fix)
print("\nTesting epoch synchronization (Validation Mirage fix):")
for test_epoch in [0, 100, 300, 500, 650]:
    dataset.update_epoch(test_epoch)
    phase = dataset.get_phase_info()
    print(f"  Epoch {test_epoch:3d}: Phase {phase['phase']} - {phase['name']}")

# Test getting a sample
print("\nTesting sample generation...")
sample = dataset[0]
print(f"  ✓ Sample generated:")
print(f"    noisy shape: {sample['noisy'].shape}")
print(f"    clean shape: {sample['clean'].shape}")
print(f"    mask shape: {sample['mask'].shape}")
print(f"    snr: {sample['snr'].item():.2f} dB")

print("\n✓ All MR-TAE-GEM tests passed!")
