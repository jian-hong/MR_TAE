"""Integration test for MR-TAE-Fusion."""
import torch
from mr_tae_fusion.config import Config, get_config
from mr_tae_fusion.models import create_model
from mr_tae_fusion.data import PDSignalDataset

try:
    # Test with default config
    print("Testing with full config...")
    config = get_config()
    model = create_model(config.model)
    print(f"Full model parameters: {model.count_parameters():,}")

    # Test data generation
    print("\nTesting data generation...")
    dataset = PDSignalDataset(config, num_samples=10, seed=42)
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Noisy shape: {sample['noisy'].shape}")
    print(f"Clean shape: {sample['clean'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")

    # Forward pass with data
    print("\nTesting forward pass with dataset sample...")
    model.eval()
    with torch.no_grad():
        noisy = sample['noisy'].unsqueeze(0)
        denoised, seg = model(noisy)
        print(f"Denoised: {denoised.shape}")
        print(f"Segmentation: {seg.shape}")

    print("\n=== ALL TESTS PASSED! ===")
except Exception as e:
    import traceback
    traceback.print_exc()
