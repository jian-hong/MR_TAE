"""
Unit tests for wavelet transform layers.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from mr_tae_fusion.models.wavelet_layers import (
    DWT1d, IDWT1d, get_wavelet_filters,
    MultiLevelDWT1d, MultiLevelIDWT1d
)


class TestWaveletFilters:
    """Test wavelet filter retrieval."""
    
    def test_get_haar_filters(self):
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters('haar')
        assert len(dec_lo) == 2
        assert len(dec_hi) == 2
        
    def test_get_db4_filters(self):
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters('db4')
        assert len(dec_lo) == 8
        assert len(dec_hi) == 8
        
    def test_invalid_wavelet(self):
        with pytest.raises(ValueError):
            get_wavelet_filters('invalid_wavelet')


class TestDWT1d:
    """Test 1D Discrete Wavelet Transform."""
    
    def test_forward_shape(self):
        """Test output shape is correct."""
        dwt = DWT1d(wavelet='db4', in_channels=1)
        x = torch.randn(2, 1, 1024)
        y = dwt(x)
        
        # Output should be (B, 2*C, L//2)
        assert y.shape == (2, 2, 512)
    
    def test_multi_channel(self):
        """Test with multiple channels."""
        dwt = DWT1d(wavelet='db4', in_channels=4)
        x = torch.randn(2, 4, 1024)
        y = dwt(x)
        
        assert y.shape == (2, 8, 512)
    
    def test_differentiable(self):
        """Test that gradients flow through."""
        dwt = DWT1d(wavelet='db4', in_channels=1)
        x = torch.randn(2, 1, 1024, requires_grad=True)
        y = dwt(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestIDWT1d:
    """Test 1D Inverse Discrete Wavelet Transform."""
    
    def test_forward_shape(self):
        """Test output shape is correct."""
        idwt = IDWT1d(wavelet='db4', out_channels=1)
        x = torch.randn(2, 2, 512)  # (B, 2*C, L)
        y = idwt(x)
        
        # Output should be (B, C, 2*L)
        assert y.shape == (2, 1, 1024)
    
    def test_multi_channel(self):
        """Test with multiple channels."""
        idwt = IDWT1d(wavelet='db4', out_channels=4)
        x = torch.randn(2, 8, 512)
        y = idwt(x)
        
        assert y.shape == (2, 4, 1024)


class TestDWTIDWTConsistency:
    """Test DWT/IDWT reconstruction."""
    
    def test_haar_reconstruction(self):
        """Test that IDWT(DWT(x)) ≈ x for Haar."""
        dwt = DWT1d(wavelet='haar', in_channels=1)
        idwt = IDWT1d(wavelet='haar', out_channels=1)
        
        x = torch.randn(2, 1, 256)
        y = dwt(x)
        x_rec = idwt(y, output_length=256)
        
        # Should reconstruct with small error
        error = torch.mean((x - x_rec) ** 2)
        assert error < 0.1, f"Reconstruction error too high: {error}"
    
    def test_db4_reconstruction(self):
        """Test that IDWT(DWT(x)) ≈ x for db4."""
        dwt = DWT1d(wavelet='db4', in_channels=1)
        idwt = IDWT1d(wavelet='db4', out_channels=1)
        
        x = torch.randn(2, 1, 256)
        y = dwt(x)
        x_rec = idwt(y, output_length=256)
        
        error = torch.mean((x - x_rec) ** 2)
        assert error < 0.1, f"Reconstruction error too high: {error}"


class TestMultiLevelDWT:
    """Test multi-level wavelet decomposition."""
    
    def test_multilevel_shapes(self):
        """Test output shapes at each level."""
        dwt = MultiLevelDWT1d(wavelet='db4', in_channels=1, levels=3)
        x = torch.randn(2, 1, 1024)
        
        coeffs = dwt(x)
        
        # Should return (approx, detail_3, detail_2, detail_1)
        assert len(coeffs) == 4
        
        # Check shapes
        # After 3 levels: 1024 -> 512 -> 256 -> 128
        assert coeffs[0].shape[-1] == 128  # Approximation
        assert coeffs[1].shape[-1] == 128  # Detail level 3
        assert coeffs[2].shape[-1] == 256  # Detail level 2
        assert coeffs[3].shape[-1] == 512  # Detail level 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
