"""
Multilevel Wavelet CNN (MWCNN) encoder and decoder.

Replaces pooling with DWT for lossless downsampling, preserving high-frequency
transient edges critical for PD pulse detection at extreme noise levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .wavelet_layers import DWT1d, IDWT1d


class ConvBlock(nn.Module):
    """
    Basic convolutional block with BatchNorm and LeakyReLU.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        padding: Padding mode
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: str = 'same',
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MWCNNBlock(nn.Module):
    """
    MWCNN encoder block: Conv -> Conv -> DWT downsampling.
    
    Performs 2x convolutions followed by DWT to halve spatial resolution
    while preserving frequency information in expanded channels.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels after convolutions (before DWT)
        wavelet: Wavelet type for DWT
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet: str = 'db4',
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, dropout=dropout)
        self.dwt = DWT1d(wavelet=wavelet, in_channels=out_channels)
        
        # Output channels after DWT = 2 * out_channels
        self.out_channels = out_channels * 2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input (B, C_in, L)
        
        Returns:
            Tuple of (downsampled_output, skip_features)
            - downsampled: (B, 2*C_out, L//2)
            - skip: (B, C_out, L) - features before DWT for skip connection
        """
        features = self.conv1(x)
        features = self.conv2(features)
        
        # Keep pre-DWT features for skip connection
        skip = features
        
        # DWT downsampling
        down = self.dwt(features)
        
        return down, skip


class MWCNNEncoder(nn.Module):
    """
    MWCNN Encoder with hierarchical DWT downsampling.
    
    Progressively extracts multi-scale features while preserving
    high-frequency information through wavelet decomposition.
    
    Args:
        in_channels: Number of input channels
        channel_list: List of channels at each level [32, 64, 128]
        wavelet: Wavelet type for DWT
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        channel_list: List[int] = [32, 64, 128],
        wavelet: str = 'db4',
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.levels = len(channel_list)
        self.blocks = nn.ModuleList()
        
        current_channels = in_channels
        
        for i, out_channels in enumerate(channel_list):
            block = MWCNNBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                wavelet=wavelet,
                kernel_size=kernel_size,
                dropout=dropout
            )
            self.blocks.append(block)
            # After DWT, channels double
            current_channels = out_channels * 2
        
        self.final_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
        """
        Encode input signal.
        
        Args:
            x: Input (B, C_in, L)
        
        Returns:
            Tuple of:
            - encoded: Final encoded features (B, C_final, L_final)
            - skip_features: List of skip connection features
            - skip_lengths: List of skip feature lengths for reconstruction
        """
        skip_features = []
        skip_lengths = []
        
        current = x
        
        for block in self.blocks:
            current, skip = block(current)
            skip_features.append(skip)
            skip_lengths.append(skip.shape[-1])
        
        return current, skip_features, skip_lengths


class MWCNNDecoderBlock(nn.Module):
    """
    MWCNN decoder block: Upsample -> Conv -> Conv.
    
    Performs upsampling (IDWT or interpolation) then refines with convolutions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels after convolutions
        skip_channels: Skip connection channels
        wavelet: Wavelet type for IDWT
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        use_idwt: Whether to use IDWT (True) or interpolation (False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        wavelet: str = 'db4',
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_idwt: bool = False
    ):
        super().__init__()
        
        self.use_idwt = use_idwt and (in_channels % 2 == 0)
        
        if self.use_idwt:
            # IDWT expects 2*channels input and produces channels output
            self.idwt = IDWT1d(wavelet=wavelet, out_channels=in_channels // 2)
            upsample_out = in_channels // 2
        else:
            # Use transposed convolution for upsampling
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(in_channels // 2),
                nn.LeakyReLU(0.1, inplace=True)
            )
            upsample_out = in_channels // 2
        
        # After upsample + skip concatenation
        conv_in_channels = upsample_out + skip_channels
        
        self.conv1 = ConvBlock(conv_in_channels, out_channels, kernel_size, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, dropout=dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor,
        output_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (B, C_in, L)
            skip: Skip connection features (B, C_skip, L_skip)
            output_length: Target output length
        
        Returns:
            Upsampled and refined features (B, C_out, L_out)
        """
        # Upsampling
        if self.use_idwt:
            up = self.idwt(x, output_length)
        else:
            up = self.upsample(x)
            if output_length is not None and up.shape[-1] != output_length:
                up = F.interpolate(up, size=output_length, mode='linear', align_corners=False)
        
        # Ensure skip connection alignment
        if up.shape[-1] != skip.shape[-1]:
            # Adjust length if needed
            min_len = min(up.shape[-1], skip.shape[-1])
            up = up[:, :, :min_len]
            skip = skip[:, :, :min_len]
        
        # Concatenate with skip connection
        combined = torch.cat([up, skip], dim=1)
        
        # Convolutions
        out = self.conv1(combined)
        out = self.conv2(out)
        
        return out


class MWCNNDecoder(nn.Module):
    """
    MWCNN Decoder with hierarchical IDWT upsampling.
    
    Progressively reconstructs signal using IDWT and skip connections.
    
    Args:
        bottleneck_channels: Number of channels from bottleneck
        channel_list: List of output channels at each level [128, 64, 32]
        skip_channels: List of skip connection channels
        wavelet: Wavelet type for IDWT
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        bottleneck_channels: int = 256,
        channel_list: List[int] = [128, 64, 32],
        skip_channels: List[int] = [128, 64, 32],
        wavelet: str = 'db4',
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.levels = len(channel_list)
        self.blocks = nn.ModuleList()
        
        current_channels = bottleneck_channels
        
        for i, (out_ch, skip_ch) in enumerate(zip(channel_list, skip_channels)):
            block = MWCNNDecoderBlock(
                in_channels=current_channels,
                out_channels=out_ch,
                skip_channels=skip_ch,
                wavelet=wavelet,
                kernel_size=kernel_size,
                dropout=dropout
            )
            self.blocks.append(block)
            # Next block receives output from this block
            current_channels = out_ch
        
        self.final_channels = channel_list[-1]
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: List[torch.Tensor],
        skip_lengths: List[int]
    ) -> torch.Tensor:
        """
        Decode bottleneck features.
        
        Args:
            x: Bottleneck features (B, C_bottleneck, L_bottleneck)
            skip_features: List of encoder skip features (reversed order)
            skip_lengths: List of target lengths for each level
        
        Returns:
            Reconstructed features (B, C_final, L_original)
        """
        current = x
        
        # Process in reverse order (deepest to shallowest)
        for i, block in enumerate(self.blocks):
            skip = skip_features[-(i + 1)]  # Reverse order
            target_length = skip_lengths[-(i + 1)]
            current = block(current, skip, target_length)
        
        return current
