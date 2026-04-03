"""
Attention Gate module for skip connections.

Implements learnable attention that suppresses irrelevant noise regions
while highlighting salient PD pulse features in skip connections.
"""

import torch
import torch.nn as nn
from typing import Optional


class AttentionGate(nn.Module):
    """
    Attention Gate for filtering skip connections.
    
    Computes attention coefficients α ∈ [0, 1] that weight skip connection
    features based on gating signal from deeper layers.
    
    Mathematical formulation:
    α = σ(ψ(ReLU(W_x·x + W_g·g + b_g) + b_ψ))
    output = x * α
    
    This allows the network to use global context (g) to filter local
    features (x), suppressing stochastic noise while preserving PD pulses.
    
    Args:
        gate_channels: Channels in gating signal (from decoder)
        skip_channels: Channels in skip connection (from encoder)
        intermediate_channels: Channels in intermediate representation
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        intermediate_channels: Optional[int] = None
    ):
        super().__init__()
        
        if intermediate_channels is None:
            intermediate_channels = skip_channels // 2
        
        # Linear projections (1x1 convolutions)
        self.W_gate = nn.Conv1d(
            gate_channels, 
            intermediate_channels, 
            kernel_size=1,
            bias=True
        )
        
        self.W_skip = nn.Conv1d(
            skip_channels, 
            intermediate_channels, 
            kernel_size=1,
            bias=False  # Bias in W_gate is sufficient
        )
        
        # Final projection to scalar attention map
        self.psi = nn.Conv1d(
            intermediate_channels,
            1,
            kernel_size=1,
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Optional batch normalization for stable training
        self.bn = nn.BatchNorm1d(intermediate_channels)
    
    def forward(
        self, 
        skip: torch.Tensor, 
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention gating to skip connection.
        
        Args:
            skip: Skip connection features from encoder (B, C_skip, L_skip)
            gate: Gating signal from decoder (B, C_gate, L_gate)
        
        Returns:
            Gated skip features (B, C_skip, L_skip)
        """
        # Ensure gate signal matches skip spatial dimensions
        if gate.shape[-1] != skip.shape[-1]:
            gate = nn.functional.interpolate(
                gate, 
                size=skip.shape[-1],
                mode='linear',
                align_corners=False
            )
        
        # Project to intermediate space
        g_proj = self.W_gate(gate)
        x_proj = self.W_skip(skip)
        
        # Additive attention
        attention = self.relu(g_proj + x_proj)
        attention = self.bn(attention)
        
        # Generate attention coefficients
        alpha = self.psi(attention)
        alpha = self.sigmoid(alpha)  # (B, 1, L)
        
        # Apply attention to skip connection
        gated = skip * alpha
        
        return gated


class MultiHeadAttentionGate(nn.Module):
    """
    Multi-head version of Attention Gate.
    
    Allows the model to attend to different aspects of the skip features
    simultaneously (e.g., pulse onset, pulse peak, pulse decay).
    
    Args:
        gate_channels: Channels in gating signal
        skip_channels: Channels in skip connection
        num_heads: Number of attention heads
        intermediate_channels: Channels per head
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        num_heads: int = 4,
        intermediate_channels: Optional[int] = None
    ):
        super().__init__()
        
        self.num_heads = num_heads
        
        if intermediate_channels is None:
            intermediate_channels = skip_channels // num_heads
        
        self.heads = nn.ModuleList([
            AttentionGate(gate_channels, skip_channels, intermediate_channels)
            for _ in range(num_heads)
        ])
        
        # Projection to combine head outputs
        self.combine = nn.Conv1d(
            skip_channels * num_heads,
            skip_channels,
            kernel_size=1
        )
    
    def forward(
        self, 
        skip: torch.Tensor, 
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply multi-head attention gating.
        
        Args:
            skip: Skip connection features (B, C_skip, L)
            gate: Gating signal (B, C_gate, L_gate)
        
        Returns:
            Gated features (B, C_skip, L)
        """
        # Apply each attention head
        head_outputs = [head(skip, gate) for head in self.heads]
        
        # Concatenate and project
        combined = torch.cat(head_outputs, dim=1)
        output = self.combine(combined)
        
        return output


class GatedSkipConnection(nn.Module):
    """
    Complete gated skip connection module.
    
    Combines Attention Gate with residual connection option.
    
    Args:
        gate_channels: Channels in gating signal
        skip_channels: Channels in skip connection
        use_residual: Whether to add residual connection
        intermediate_channels: Attention gate intermediate channels
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        use_residual: bool = True,
        intermediate_channels: Optional[int] = None
    ):
        super().__init__()
        
        self.attention_gate = AttentionGate(
            gate_channels,
            skip_channels,
            intermediate_channels
        )
        
        self.use_residual = use_residual
        
        # Learnable residual weight
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self, 
        skip: torch.Tensor, 
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gated skip connection with optional residual.
        
        Args:
            skip: Skip features (B, C_skip, L)
            gate: Gating signal (B, C_gate, L_gate)
        
        Returns:
            Gated features (B, C_skip, L)
        """
        gated = self.attention_gate(skip, gate)
        
        if self.use_residual:
            # Weighted residual connection
            output = gated + self.residual_weight * skip
        else:
            output = gated
        
        return output
