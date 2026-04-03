"""
Dual-Stream Fusion Head for semantic segmentation.

Combines global context from bottleneck with local detail from decoder
for per-timestamp defect classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DualStreamFusionHead(nn.Module):
    """
    Dual-Stream Fusion Head for semantic segmentation.
    
    Fuses:
    - Stream 1 (Global Context): Upsampled bottleneck features with semantic info
    - Stream 2 (Local Detail): Final decoder features with precise localization
    
    Output: Per-timestamp class probabilities (B, num_classes, L)
    
    Args:
        global_channels: Channels from bottleneck (after upsampling)
        local_channels: Channels from final decoder stage
        fusion_channels: Intermediate fusion channels
        num_classes: Number of segmentation classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        global_channels: int,
        local_channels: int,
        fusion_channels: int = 64,
        num_classes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Global context stream processing
        self.global_conv = nn.Sequential(
            nn.Conv1d(global_channels, fusion_channels, kernel_size=1),
            nn.BatchNorm1d(fusion_channels),
            nn.ReLU(inplace=True)
        )
        
        # Local detail stream processing
        self.local_conv = nn.Sequential(
            nn.Conv1d(local_channels, fusion_channels, kernel_size=1),
            nn.BatchNorm1d(fusion_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_channels * 2, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final classification layer
        self.classifier = nn.Conv1d(fusion_channels // 2, num_classes, kernel_size=1)
    
    def forward(
        self, 
        global_features: torch.Tensor, 
        local_features: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            global_features: Bottleneck features (B, C_global, L_global)
            local_features: Decoder features (B, C_local, L_local)
            target_length: Target output length
        
        Returns:
            Segmentation logits (B, num_classes, L)
        """
        # Get target length from local features if not specified
        if target_length is None:
            target_length = local_features.shape[-1]
        
        # Upsample global features to match local
        global_up = F.interpolate(
            global_features,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        # Ensure local features match target length
        if local_features.shape[-1] != target_length:
            local_features = F.interpolate(
                local_features,
                size=target_length,
                mode='linear',
                align_corners=False
            )
        
        # Process streams
        global_processed = self.global_conv(global_up)
        local_processed = self.local_conv(local_features)
        
        # Concatenate and fuse
        fused = torch.cat([global_processed, local_processed], dim=1)
        fused = self.fusion(fused)
        
        # Final classification
        logits = self.classifier(fused)
        
        return logits


class ReconstructionHead(nn.Module):
    """
    Reconstruction head for denoised signal output.
    
    Takes decoder features and produces clean signal estimate.
    
    Args:
        in_channels: Input channels from decoder
        out_channels: Output channels (typically 1 for signal)
    """
    
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 2, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        Generate reconstructed signal.
        
        Args:
            x: Decoder features (B, C, L)
            target_length: Target output length
        
        Returns:
            Reconstructed signal (B, out_channels, L)
        """
        out = self.conv(x)
        
        if target_length is not None and out.shape[-1] != target_length:
            out = F.interpolate(out, size=target_length, mode='linear', align_corners=False)
        
        return out


class AttentionFusionHead(nn.Module):
    """
    Attention-based fusion head that learns to weight contributions
    from global and local streams dynamically.
    
    Args:
        global_channels: Channels from bottleneck
        local_channels: Channels from decoder
        fusion_channels: Intermediate fusion channels
        num_classes: Number of segmentation classes
    """
    
    def __init__(
        self,
        global_channels: int,
        local_channels: int,
        fusion_channels: int = 64,
        num_classes: int = 4
    ):
        super().__init__()
        
        # Project to common dimension
        self.global_proj = nn.Conv1d(global_channels, fusion_channels, kernel_size=1)
        self.local_proj = nn.Conv1d(local_channels, fusion_channels, kernel_size=1)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Conv1d(fusion_channels * 2, fusion_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fusion_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Output
        self.classifier = nn.Conv1d(fusion_channels, num_classes, kernel_size=1)
    
    def forward(
        self, 
        global_features: torch.Tensor, 
        local_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with learned attention fusion.
        
        Args:
            global_features: (B, C_global, L_global)
            local_features: (B, C_local, L_local)
        
        Returns:
            Segmentation logits (B, num_classes, L)
        """
        target_length = local_features.shape[-1]
        
        # Upsample and project global
        global_up = F.interpolate(
            global_features,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        global_proj = self.global_proj(global_up)
        
        # Project local
        local_proj = self.local_proj(local_features)
        
        # Compute attention weights
        concat = torch.cat([global_proj, local_proj], dim=1)
        weights = self.attention(concat)  # (B, 2, L)
        
        # Weighted sum
        fused = weights[:, 0:1, :] * global_proj + weights[:, 1:2, :] * local_proj
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
