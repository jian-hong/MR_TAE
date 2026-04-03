"""
Differentiable Discrete Wavelet Transform layers for PyTorch.

Implements DWT and IDWT using 1D convolutions with frozen wavelet filter weights.
Uses Daubechies-4 (db4) wavelets by default for optimal PD pulse correlation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# Daubechies wavelet filter coefficients
WAVELET_FILTERS = {
    'haar': {
        'dec_lo': [0.7071067811865476, 0.7071067811865476],
        'dec_hi': [-0.7071067811865476, 0.7071067811865476],
        'rec_lo': [0.7071067811865476, 0.7071067811865476],
        'rec_hi': [0.7071067811865476, -0.7071067811865476],
    },
    'db2': {
        'dec_lo': [-0.12940952255092145, 0.22414386804185735, 
                   0.836516303737469, 0.48296291314469025],
        'dec_hi': [-0.48296291314469025, 0.836516303737469, 
                   -0.22414386804185735, -0.12940952255092145],
        'rec_lo': [0.48296291314469025, 0.836516303737469, 
                   0.22414386804185735, -0.12940952255092145],
        'rec_hi': [-0.12940952255092145, -0.22414386804185735, 
                   0.836516303737469, -0.48296291314469025],
    },
    'db4': {
        'dec_lo': [-0.010597401784997278, 0.032883011666982945, 
                   0.030841381835986965, -0.18703481171888114,
                   -0.02798376941698385, 0.6308807679295904, 
                   0.7148465705525415, 0.23037781330885523],
        'dec_hi': [-0.23037781330885523, 0.7148465705525415, 
                   -0.6308807679295904, -0.02798376941698385,
                   0.18703481171888114, 0.030841381835986965, 
                   -0.032883011666982945, -0.010597401784997278],
        'rec_lo': [0.23037781330885523, 0.7148465705525415, 
                   0.6308807679295904, -0.02798376941698385,
                   -0.18703481171888114, 0.030841381835986965, 
                   0.032883011666982945, -0.010597401784997278],
        'rec_hi': [-0.010597401784997278, -0.032883011666982945, 
                   0.030841381835986965, 0.18703481171888114,
                   -0.02798376941698385, -0.6308807679295904, 
                   0.7148465705525415, -0.23037781330885523],
    },
    'db6': {
        'dec_lo': [
            -0.00107730108499558, 0.004777257511010651,
            0.0005538422009938016, -0.031582039318031156,
            0.02752286553001629, 0.09750160558707936,
            -0.12976686756709563, -0.22626469396516913,
            0.3152503517092432, 0.7511339080215775,
            0.4946238903983854, 0.11154074335008017
        ],
        'dec_hi': [
            -0.11154074335008017, 0.4946238903983854,
            -0.7511339080215775, 0.3152503517092432,
            0.22626469396516913, -0.12976686756709563,
            -0.09750160558707936, 0.02752286553001629,
            0.031582039318031156, 0.0005538422009938016,
            -0.004777257511010651, -0.00107730108499558
        ],
        'rec_lo': [
            0.11154074335008017, 0.4946238903983854,
            0.7511339080215775, 0.3152503517092432,
            -0.22626469396516913, -0.12976686756709563,
            0.09750160558707936, 0.02752286553001629,
            -0.031582039318031156, 0.0005538422009938016,
            0.004777257511010651, -0.00107730108499558
        ],
        'rec_hi': [
            -0.00107730108499558, -0.004777257511010651,
            0.0005538422009938016, 0.031582039318031156,
            0.02752286553001629, -0.09750160558707936,
            -0.12976686756709563, 0.22626469396516913,
            0.3152503517092432, -0.7511339080215775,
            0.4946238903983854, -0.11154074335008017
        ],
    }
}


def get_wavelet_filters(wavelet: str = 'db4') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get wavelet filter coefficients.
    
    Args:
        wavelet: Wavelet name ('haar', 'db2', 'db4', 'db6')
    
    Returns:
        Tuple of (dec_lo, dec_hi, rec_lo, rec_hi) filter arrays
    """
    if wavelet not in WAVELET_FILTERS:
        raise ValueError(f"Unknown wavelet: {wavelet}. Available: {list(WAVELET_FILTERS.keys())}")
    
    filters = WAVELET_FILTERS[wavelet]
    return (
        np.array(filters['dec_lo']),
        np.array(filters['dec_hi']),
        np.array(filters['rec_lo']),
        np.array(filters['rec_hi'])
    )


class DWT1d(nn.Module):
    """
    1D Discrete Wavelet Transform layer.
    
    Performs single-level DWT decomposition using 1D convolutions.
    Input: (B, C, L) -> Output: (B, 2*C, L//2)
    
    The high-frequency (detail) and low-frequency (approximation) coefficients
    are stacked along the channel dimension.
    
    Args:
        wavelet: Wavelet type ('haar', 'db2', 'db4', 'db6')
        in_channels: Number of input channels (applied per-channel)
    """
    
    def __init__(self, wavelet: str = 'db4', in_channels: int = 1):
        super().__init__()
        
        self.wavelet = wavelet
        self.in_channels = in_channels
        
        # Get filter coefficients
        dec_lo, dec_hi, _, _ = get_wavelet_filters(wavelet)
        filter_len = len(dec_lo)
        
        # Create convolution kernels
        # Shape: (out_channels, in_channels/groups, kernel_size)
        # Using groups=in_channels for channel-wise convolution
        lo_filter = torch.tensor(dec_lo, dtype=torch.float32).flip(0)
        hi_filter = torch.tensor(dec_hi, dtype=torch.float32).flip(0)
        
        # Expand for all input channels
        lo_filter = lo_filter.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1)
        hi_filter = hi_filter.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1)
        
        # Register as buffers (not trainable)
        self.register_buffer('lo_filter', lo_filter)
        self.register_buffer('hi_filter', hi_filter)
        
        self.filter_len = filter_len
        self.pad = filter_len // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward DWT.
        
        Args:
            x: Input tensor (B, C, L)
        
        Returns:
            Output tensor (B, 2*C, L//2) with [low, high] stacked
        """
        B, C, L = x.shape
        
        # Pad input for valid convolution
        # Use reflection padding to handle boundaries
        x_padded = F.pad(x, (self.pad, self.pad), mode='reflect')
        
        # Apply low-pass and high-pass filters (grouped convolution)
        lo = F.conv1d(x_padded, self.lo_filter, groups=C)
        hi = F.conv1d(x_padded, self.hi_filter, groups=C)
        
        # Downsample by factor of 2
        lo = lo[:, :, ::2]
        hi = hi[:, :, ::2]
        
        # Stack along channel dimension: [low_freq, high_freq]
        out = torch.cat([lo, hi], dim=1)
        
        return out


class IDWT1d(nn.Module):
    """
    1D Inverse Discrete Wavelet Transform layer.
    
    Performs single-level IDWT reconstruction using 1D transposed convolutions.
    Input: (B, 2*C, L) -> Output: (B, C, 2*L)
    
    Expects input with [low, high] coefficients stacked along channel dimension.
    
    Args:
        wavelet: Wavelet type ('haar', 'db2', 'db4', 'db6')
        out_channels: Number of output channels
    """
    
    def __init__(self, wavelet: str = 'db4', out_channels: int = 1):
        super().__init__()
        
        self.wavelet = wavelet
        self.out_channels = out_channels
        
        # Get reconstruction filter coefficients
        _, _, rec_lo, rec_hi = get_wavelet_filters(wavelet)
        filter_len = len(rec_lo)
        
        # Create reconstruction kernels
        lo_filter = torch.tensor(rec_lo, dtype=torch.float32)
        hi_filter = torch.tensor(rec_hi, dtype=torch.float32)
        
        # Expand for all output channels
        lo_filter = lo_filter.unsqueeze(0).unsqueeze(0).repeat(out_channels, 1, 1)
        hi_filter = hi_filter.unsqueeze(0).unsqueeze(0).repeat(out_channels, 1, 1)
        
        self.register_buffer('lo_filter', lo_filter)
        self.register_buffer('hi_filter', hi_filter)
        
        self.filter_len = filter_len
    
    def forward(self, x: torch.Tensor, output_length: Optional[int] = None) -> torch.Tensor:
        """
        Inverse DWT.
        
        Args:
            x: Input tensor (B, 2*C, L) with [low, high] stacked
            output_length: Desired output length (for handling odd-length signals)
        
        Returns:
            Output tensor (B, C, 2*L or output_length)
        """
        B, C2, L = x.shape
        C = C2 // 2
        
        # Split low and high frequency components
        lo = x[:, :C, :]
        hi = x[:, C:, :]
        
        # Upsample by inserting zeros
        lo_up = torch.zeros(B, C, L * 2, device=x.device, dtype=x.dtype)
        hi_up = torch.zeros(B, C, L * 2, device=x.device, dtype=x.dtype)
        lo_up[:, :, ::2] = lo
        hi_up[:, :, ::2] = hi
        
        # Pad for convolution
        pad = self.filter_len - 1
        lo_up = F.pad(lo_up, (pad, pad), mode='reflect')
        hi_up = F.pad(hi_up, (pad, pad), mode='reflect')
        
        # Apply reconstruction filters
        rec_lo = F.conv1d(lo_up, self.lo_filter, groups=C)
        rec_hi = F.conv1d(hi_up, self.hi_filter, groups=C)
        
        # Sum contributions
        out = rec_lo + rec_hi
        
        # Trim to correct length
        if output_length is not None:
            out = out[:, :, :output_length]
        else:
            out = out[:, :, :L * 2]
        
        return out


class MultiLevelDWT1d(nn.Module):
    """
    Multi-level 1D DWT decomposition.
    
    Applies DWT recursively to the low-frequency component.
    
    Args:
        wavelet: Wavelet type
        in_channels: Number of input channels
        levels: Number of decomposition levels
    """
    
    def __init__(self, wavelet: str = 'db4', in_channels: int = 1, levels: int = 3):
        super().__init__()
        
        self.wavelet = wavelet
        self.levels = levels
        
        # Create DWT layers for each level
        self.dwt_layers = nn.ModuleList()
        current_channels = in_channels
        
        for _ in range(levels):
            self.dwt_layers.append(DWT1d(wavelet, current_channels))
            current_channels *= 2  # Channels double at each level
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Multi-level DWT decomposition.
        
        Args:
            x: Input (B, C, L)
        
        Returns:
            Tuple of (final_approx, detail_level_n, ..., detail_level_1)
        """
        details = []
        current = x
        
        for i, dwt in enumerate(self.dwt_layers):
            # Apply DWT
            coeffs = dwt(current)
            
            # Split into approximation and detail
            C = coeffs.shape[1] // 2
            approx = coeffs[:, :C, :]
            detail = coeffs[:, C:, :]
            
            details.append(detail)
            current = approx
        
        # Return (approx, detail_n, detail_n-1, ..., detail_1)
        return (current,) + tuple(reversed(details))


class MultiLevelIDWT1d(nn.Module):
    """
    Multi-level 1D IDWT reconstruction.
    
    Reconstructs signal from multi-level DWT coefficients.
    
    Args:
        wavelet: Wavelet type
        out_channels: Number of output channels  
        levels: Number of reconstruction levels
    """
    
    def __init__(self, wavelet: str = 'db4', out_channels: int = 1, levels: int = 3):
        super().__init__()
        
        self.wavelet = wavelet
        self.levels = levels
        
        # Create IDWT layers for each level (reverse order)
        self.idwt_layers = nn.ModuleList()
        
        # Channel sequence: out_channels * 2^levels down to out_channels
        for i in range(levels):
            level_channels = out_channels * (2 ** (levels - i - 1))
            self.idwt_layers.append(IDWT1d(wavelet, level_channels))
    
    def forward(
        self, 
        coeffs: Tuple[torch.Tensor, ...],
        output_lengths: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Multi-level IDWT reconstruction.
        
        Args:
            coeffs: Tuple of (approx, detail_n, ..., detail_1)
            output_lengths: Tuple of expected output lengths at each level
        
        Returns:
            Reconstructed signal (B, C, L)
        """
        approx = coeffs[0]
        details = coeffs[1:]  # Already in correct order (n to 1)
        
        current = approx
        
        for i, (idwt, detail) in enumerate(zip(self.idwt_layers, details)):
            # Combine approximation and detail
            combined = torch.cat([current, detail], dim=1)
            
            # Get output length if provided
            out_len = output_lengths[i] if output_lengths else None
            
            # Apply IDWT
            current = idwt(combined, out_len)
        
        return current
