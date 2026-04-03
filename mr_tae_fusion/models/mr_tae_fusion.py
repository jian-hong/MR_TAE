"""
MR-TAE-Fusion: Complete Multi-Task Recurrent-Transformer Autoencoder.

Integrates MWCNN backbone, BiGRU-Swin bottleneck, Attention Gates,
and Dual-Stream Fusion Head for joint denoising and segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict

from .mwcnn import MWCNNEncoder, MWCNNDecoder, ConvBlock
from .attention import AttentionGate, GatedSkipConnection
from .swin_transformer import SwinTransformer1D
from .fusion_head import DualStreamFusionHead, ReconstructionHead
from ..config import ModelConfig


class HybridBottleneck(nn.Module):
    """
    Hybrid bottleneck combining BiGRU and Swin Transformer.
    
    BiGRU captures sequential temporal dependencies while
    Swin Transformer captures global long-range context.
    
    Args:
        in_channels: Input feature channels
        gru_hidden: GRU hidden size
        gru_layers: Number of GRU layers
        gru_dropout: GRU dropout
        swin_dim: Swin Transformer dimension
        swin_heads: Number of attention heads
        swin_depth: Number of Swin blocks
        swin_window: Window size for Swin attention
        swin_mlp_ratio: MLP ratio for Swin
        swin_dropout: Swin dropout
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        swin_dim: int = 256,
        swin_heads: int = 8,
        swin_depth: int = 2,
        swin_window: int = 32,
        swin_mlp_ratio: float = 4.0,
        swin_dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.gru_hidden = gru_hidden
        self.swin_dim = swin_dim
        
        # Project to GRU dimension
        self.pre_gru = nn.Conv1d(in_channels, gru_hidden * 2, kernel_size=1)
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=gru_hidden * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0
        )
        
        # GRU output is 2*hidden (bidirectional), project to Swin dimension
        self.gru_to_swin = nn.Conv1d(gru_hidden * 2, swin_dim, kernel_size=1)
        
        # Swin Transformer for global context
        self.swin = SwinTransformer1D(
            dim=swin_dim,
            depth=swin_depth,
            num_heads=swin_heads,
            window_size=swin_window,
            mlp_ratio=swin_mlp_ratio,
            dropout=swin_dropout,
            attention_dropout=swin_dropout
        )
        
        # Final projection back to expected dimension
        self.output_proj = nn.Conv1d(swin_dim, in_channels, kernel_size=1)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        self.output_channels = in_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid bottleneck.
        
        Args:
            x: Input features (B, C, L)
        
        Returns:
            Bottleneck features (B, C, L)
        """
        residual = self.residual(x)
        
        # Prepare for GRU
        x_gru = self.pre_gru(x)
        
        # GRU expects (B, L, C)
        x_gru = x_gru.permute(0, 2, 1)
        
        # BiGRU
        gru_out, _ = self.gru(x_gru)  # (B, L, 2*hidden)
        
        # Back to (B, C, L)
        gru_out = gru_out.permute(0, 2, 1)
        
        # Project and apply Swin Transformer
        swin_in = self.gru_to_swin(gru_out)
        swin_out = self.swin(swin_in)
        
        # Project to output dimension
        out = self.output_proj(swin_out)
        
        # Add residual
        out = out + residual
        
        return out


class MRTAEFusion(nn.Module):
    """
    MR-TAE-Fusion: Multi-Task Recurrent-Transformer Autoencoder.
    
    Complete architecture for joint PD signal denoising and semantic segmentation.
    
    Architecture:
    - MWCNN Encoder with DWT downsampling
    - Hybrid BiGRU-Swin Transformer bottleneck
    - MWCNN Decoder with IDWT upsampling
    - Attention-gated skip connections
    - Dual output heads (reconstruction + segmentation)
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        
        # Input projection
        self.input_proj = nn.Conv1d(
            config.in_channels,
            config.encoder_channels[0],
            kernel_size=7,
            padding=3
        )
        
        # MWCNN Encoder
        self.encoder = MWCNNEncoder(
            in_channels=config.encoder_channels[0],
            channel_list=config.encoder_channels,
            wavelet='db4',
            kernel_size=config.encoder_kernel_size
        )
        
        # Pre-bottleneck projection
        encoder_out_channels = config.encoder_channels[-1] * 2  # After DWT
        self.pre_bottleneck = nn.Conv1d(
            encoder_out_channels,
            config.bottleneck_channels,
            kernel_size=1
        )
        
        # Hybrid Bottleneck
        self.bottleneck = HybridBottleneck(
            in_channels=config.bottleneck_channels,
            gru_hidden=config.gru_hidden_size,
            gru_layers=config.gru_num_layers,
            gru_dropout=config.gru_dropout,
            swin_dim=config.swin_embed_dim,
            swin_heads=config.swin_num_heads,
            swin_depth=config.swin_depth,
            swin_window=config.swin_window_size,
            swin_mlp_ratio=config.swin_mlp_ratio,
            swin_dropout=config.swin_dropout
        )
        
        # Attention Gates for skip connections
        # Each gate uses the decoder output from the previous level as the gating signal
        # First gate uses bottleneck, subsequent gates use previous decoder output
        self.attention_gates = nn.ModuleList()
        skip_channels = config.encoder_channels[::-1]  # Reverse order: [128, 64, 32]
        
        # Gate channels: bottleneck -> decoder[0] -> decoder[1] -> ...
        gate_channel_list = [config.bottleneck_channels] + list(config.decoder_channels[:-1])
        
        for skip_ch, gate_ch in zip(skip_channels, gate_channel_list):
            self.attention_gates.append(
                GatedSkipConnection(
                    gate_channels=gate_ch,
                    skip_channels=skip_ch,
                    use_residual=True,
                    intermediate_channels=config.attention_intermediate_channels
                )
            )
        
        # MWCNN Decoder
        self.decoder = MWCNNDecoder(
            bottleneck_channels=config.bottleneck_channels,
            channel_list=config.decoder_channels,
            skip_channels=skip_channels,
            wavelet='db4',
            kernel_size=config.encoder_kernel_size
        )
        
        # Reconstruction Head
        self.recon_head = ReconstructionHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.in_channels
        )
        
        # Segmentation Fusion Head
        self.seg_head = DualStreamFusionHead(
            global_channels=config.bottleneck_channels,
            local_channels=config.decoder_channels[-1],
            fusion_channels=config.fusion_channels,
            num_classes=config.num_classes
        )
        
        # Store input length for reference
        self.input_length = config.signal_length
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Noisy input signal (B, C, L) or (B, L)
        
        Returns:
            Tuple of:
            - denoised: Reconstructed clean signal (B, C, L)
            - segmentation: Per-timestamp class logits (B, num_classes, L)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        input_length = x.shape[-1]
        
        # Input projection
        x = self.input_proj(x)
        
        # Encode
        encoded, skip_features, skip_lengths = self.encoder(x)
        
        # Pre-bottleneck projection
        encoded = self.pre_bottleneck(encoded)
        
        # Bottleneck processing
        bottleneck_features = self.bottleneck(encoded)
        
        # Progressive decoding with attention-gated skip connections
        # Skip features are in order: [level1, level2, level3] (shallow to deep)
        # We process from deep to shallow, so reverse the skip features
        reversed_skips = list(reversed(skip_features))  # [level3, level2, level1]
        reversed_lengths = list(reversed(skip_lengths))
        
        # Apply attention gates progressively at each decoder level
        # Level 0: gate with bottleneck, get decoder output
        # Level 1: gate with decoder[0] output, get decoder output
        # ...
        current = bottleneck_features
        gated_skips = []
        
        for i, (ag, skip) in enumerate(zip(self.attention_gates, reversed_skips)):
            # Gate the skip connection using current decoder features
            gated = ag(skip, current)
            gated_skips.append(gated)
            
            # Decode this level (process through decoder block)
            target_length = reversed_lengths[i]
            current = self.decoder.blocks[i](current, gated, target_length)
        
        decoded = current
        
        # Reconstruction output
        denoised = self.recon_head(decoded, input_length)
        
        # Segmentation output
        segmentation = self.seg_head(bottleneck_features, decoded, input_length)
        
        return denoised, segmentation
    
    def get_denoised(self, x: torch.Tensor) -> torch.Tensor:
        """Get only denoised signal (for inference)."""
        denoised, _ = self.forward(x)
        return denoised
    
    def get_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Get only segmentation output (for inference)."""
        _, segmentation = self.forward(x)
        return segmentation
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        return {
            'total_params': self.count_parameters(),
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'bottleneck_params': sum(p.numel() for p in self.bottleneck.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
            'attention_params': sum(p.numel() for p in self.attention_gates.parameters()),
            'head_params': sum(p.numel() for p in self.recon_head.parameters()) + 
                          sum(p.numel() for p in self.seg_head.parameters())
        }


def create_model(config: Optional[ModelConfig] = None) -> MRTAEFusion:
    """
    Factory function to create MR-TAE-Fusion model.
    
    Args:
        config: Model configuration
    
    Returns:
        Initialized model
    """
    model = MRTAEFusion(config)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model
