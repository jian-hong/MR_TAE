"""Quick test for model forward pass."""
import torch
import sys
from mr_tae_fusion.config import ModelConfig
from mr_tae_fusion.models.mr_tae_fusion import MRTAEFusion

try:
    config = ModelConfig()
    config.signal_length = 512
    config.encoder_channels = [16, 32, 64]
    config.decoder_channels = [64, 32, 16]
    config.bottleneck_channels = 128
    config.gru_hidden_size = 32  
    config.gru_num_layers = 1
    config.swin_embed_dim = 128
    config.swin_depth = 1
    config.swin_num_heads = 4
    config.swin_window_size = 16

    print("Creating model...")
    model = MRTAEFusion(config)
    print(f"Model params: {model.count_parameters():,}")
    
    print("\nTesting forward pass...")
    x = torch.randn(2, 1, 512)
    print(f"Input shape: {x.shape}")
    
    denoised, seg = model(x)
    print(f"Denoised shape: {denoised.shape}")
    print(f"Segmentation shape: {seg.shape}")
    print("\nSUCCESS!")
except Exception as e:
    import traceback
    with open('error.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("ERROR - see error.txt")
    sys.exit(1)
