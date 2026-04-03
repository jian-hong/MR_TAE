"""
Unit tests for MR-TAE-Fusion model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from mr_tae_fusion.config import Config, ModelConfig
from mr_tae_fusion.models import MRTAEFusion, create_model
from mr_tae_fusion.models.mwcnn import MWCNNEncoder, MWCNNDecoder
from mr_tae_fusion.models.attention import AttentionGate
from mr_tae_fusion.models.swin_transformer import SwinTransformer1D


class TestMWCNNEncoder:
    """Test MWCNN Encoder."""
    
    def test_forward_shape(self):
        """Test encoder output shape."""
        encoder = MWCNNEncoder(
            in_channels=32,
            channel_list=[32, 64, 128],
            wavelet='db4'
        )
        
        x = torch.randn(2, 32, 2001)
        encoded, skips, lengths = encoder(x)
        
        # Check encoded shape
        assert encoded.dim() == 3
        assert encoded.shape[0] == 2  # Batch size preserved
        
        # Check skip connections
        assert len(skips) == 3
        assert len(lengths) == 3
    
    def test_channel_progression(self):
        """Test channel sizes through encoder."""
        encoder = MWCNNEncoder(
            in_channels=1,
            channel_list=[32, 64, 128],
            wavelet='db4'
        )
        
        x = torch.randn(1, 1, 1024)
        encoded, skips, lengths = encoder(x)
        
        # Skip channels should match encoder channel list
        assert skips[0].shape[1] == 32
        assert skips[1].shape[1] == 64
        assert skips[2].shape[1] == 128


class TestAttentionGate:
    """Test Attention Gate module."""
    
    def test_forward_shape(self):
        """Test attention gate preserves shape."""
        ag = AttentionGate(
            gate_channels=128,
            skip_channels=64,
            intermediate_channels=32
        )
        
        skip = torch.randn(2, 64, 500)
        gate = torch.randn(2, 128, 250)
        
        output = ag(skip, gate)
        
        assert output.shape == skip.shape
    
    def test_attention_range(self):
        """Test attention values are in [0, 1]."""
        ag = AttentionGate(
            gate_channels=64,
            skip_channels=64
        )
        
        skip = torch.randn(2, 64, 100)
        gate = torch.randn(2, 64, 50)
        
        # Hook to capture attention weights
        attention_values = []
        def hook(module, input, output):
            attention_values.append(output)
        
        ag.sigmoid.register_forward_hook(hook)
        ag(skip, gate)
        
        if attention_values:
            attn = attention_values[0]
            assert attn.min() >= 0.0
            assert attn.max() <= 1.0


class TestSwinTransformer1D:
    """Test 1D Swin Transformer."""
    
    def test_forward_shape(self):
        """Test Swin output shape."""
        swin = SwinTransformer1D(
            dim=128,
            depth=2,
            num_heads=4,
            window_size=16
        )
        
        x = torch.randn(2, 128, 256)  # (B, C, L)
        y = swin(x)
        
        assert y.shape == x.shape
    
    def test_different_lengths(self):
        """Test with various sequence lengths."""
        swin = SwinTransformer1D(
            dim=64,
            depth=2,
            num_heads=4,
            window_size=32
        )
        
        for length in [64, 128, 256, 512]:
            x = torch.randn(1, 64, length)
            y = swin(x)
            assert y.shape == x.shape


class TestMRTAEFusion:
    """Test complete MR-TAE-Fusion model."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = ModelConfig()
        config.signal_length = 512  # Smaller for testing
        config.encoder_channels = [16, 32, 64]
        config.decoder_channels = [64, 32, 16]
        config.bottleneck_channels = 128
        config.gru_hidden_size = 32
        config.gru_num_layers = 1
        config.swin_embed_dim = 128
        config.swin_depth = 1
        config.swin_num_heads = 4
        config.swin_window_size = 16
        return config
    
    def test_forward_shapes(self, config):
        """Test model output shapes."""
        model = MRTAEFusion(config)
        
        x = torch.randn(2, 1, config.signal_length)
        denoised, segmentation = model(x)
        
        # Denoised should match input shape
        assert denoised.shape == x.shape
        
        # Segmentation should be (B, num_classes, L)
        assert segmentation.shape == (2, config.num_classes, config.signal_length)
    
    def test_2d_input(self, config):
        """Test model handles 2D input (B, L)."""
        model = MRTAEFusion(config)
        
        x = torch.randn(2, config.signal_length)
        denoised, segmentation = model(x)
        
        assert denoised.shape == (2, 1, config.signal_length)
    
    def test_differentiable(self, config):
        """Test gradients flow through entire model."""
        model = MRTAEFusion(config)
        
        x = torch.randn(2, 1, config.signal_length, requires_grad=True)
        denoised, segmentation = model(x)
        
        loss = denoised.sum() + segmentation.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_get_denoised(self, config):
        """Test denoised-only inference."""
        model = MRTAEFusion(config)
        
        x = torch.randn(1, 1, config.signal_length)
        denoised = model.get_denoised(x)
        
        assert denoised.shape == x.shape
    
    def test_get_segmentation(self, config):
        """Test segmentation-only inference."""
        model = MRTAEFusion(config)
        
        x = torch.randn(1, 1, config.signal_length)
        seg = model.get_segmentation(x)
        
        assert seg.shape == (1, config.num_classes, config.signal_length)
    
    def test_count_parameters(self, config):
        """Test parameter counting."""
        model = MRTAEFusion(config)
        
        num_params = model.count_parameters()
        assert num_params > 0
        
        summary = model.get_model_summary()
        assert 'total_params' in summary


class TestCreateModel:
    """Test model factory function."""
    
    def test_create_default(self):
        """Test creating model with default config."""
        model = create_model()
        
        assert isinstance(model, MRTAEFusion)
        assert model.count_parameters() > 0
    
    def test_create_custom(self):
        """Test creating model with custom config."""
        config = ModelConfig()
        config.encoder_channels = [8, 16, 32]
        
        model = create_model(config)
        
        assert isinstance(model, MRTAEFusion)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
