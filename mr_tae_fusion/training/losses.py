"""
Multi-task loss functions with uncertainty weighting.

Implements:
- Homoscedastic uncertainty weighting for task balancing
- Dice loss for class-imbalanced segmentation
- Smooth L1 loss for robust reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Handles class imbalance by measuring overlap between prediction and target.
    
    Dice = 2 * |P ∩ T| / (|P| + |T|)
    Loss = 1 - Dice
    
    Args:
        smooth: Smoothing factor to prevent division by zero
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted logits (B, C, L) where C is num_classes
            target: Target labels (B, L) with integer class labels
        
        Returns:
            Dice loss value
        """
        num_classes = pred.shape[1]
        
        # Softmax to get probabilities
        pred_soft = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_onehot = F.one_hot(target, num_classes=num_classes)  # (B, L, C)
        target_onehot = target_onehot.permute(0, 2, 1).float()  # (B, C, L)
        
        # Compute Dice per class
        dice_per_class = []
        
        for c in range(num_classes):
            p = pred_soft[:, c, :]  # (B, L)
            t = target_onehot[:, c, :]  # (B, L)
            
            intersection = (p * t).sum(dim=1)  # (B,)
            union = p.sum(dim=1) + t.sum(dim=1)  # (B,)
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)
        
        # Stack and average across classes
        dice_tensor = torch.stack(dice_per_class, dim=1)  # (B, C)
        dice_score = dice_tensor.mean(dim=1)  # (B,)
        
        # Dice loss
        loss = 1.0 - dice_score
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss with class weighting.
    
    Weights classes inversely proportional to their frequency,
    giving more importance to rare classes (like PD pulses).
    
    Args:
        smooth: Smoothing factor
        reduction: Reduction method
    """
    
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized Dice loss.
        
        Args:
            pred: Predicted logits (B, C, L)
            target: Target labels (B, L)
        
        Returns:
            GDL value
        """
        num_classes = pred.shape[1]
        
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 2, 1).float()
        
        # Compute class weights (inverse of class frequency)
        class_sums = target_onehot.sum(dim=(0, 2)) + self.smooth  # (C,)
        weights = 1.0 / (class_sums ** 2)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted intersection and union
        intersection = (pred_soft * target_onehot).sum(dim=2)  # (B, C)
        union = pred_soft.sum(dim=2) + target_onehot.sum(dim=2)  # (B, C)
        
        weighted_intersection = (weights.unsqueeze(0) * intersection).sum(dim=1)
        weighted_union = (weights.unsqueeze(0) * union).sum(dim=1)
        
        gdl = 1.0 - (2.0 * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        
        if self.reduction == 'mean':
            return gdl.mean()
        elif self.reduction == 'sum':
            return gdl.sum()
        else:
            return gdl


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with Homoscedastic Uncertainty Weighting.
    
    Automatically balances denoising (reconstruction) and segmentation losses
    by learning task-specific uncertainty parameters.
    
    L_total = (1/(2σ₁²)) * L_recon + (1/(2σ₂²)) * L_seg + log(σ₁) + log(σ₂)
    
    The log(σ) terms prevent σ from growing unboundedly.
    
    Args:
        initial_sigma_recon: Initial reconstruction uncertainty
        initial_sigma_seg: Initial segmentation uncertainty
        dice_weight: Weight for Dice loss in segmentation
        charbonnier_eps: Epsilon for Charbonnier reconstruction loss (HPO-tunable)
    """
    
    def __init__(
        self,
        initial_sigma_recon: float = 1.0,
        initial_sigma_seg: float = 1.0,
        dice_weight: float = 1.0,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        
        # Learnable log-variance parameters
        # We learn log(σ²) for numerical stability
        self.log_var_recon = nn.Parameter(
            torch.tensor(2.0 * torch.log(torch.tensor(initial_sigma_recon)))
        )
        self.log_var_seg = nn.Parameter(
            torch.tensor(2.0 * torch.log(torch.tensor(initial_sigma_seg)))
        )
        
        # Robust reconstruction (Charbonnier); epsilon is exposed for Optuna
        self.recon_loss = CharbonnierLoss(epsilon=charbonnier_eps)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = GeneralizedDiceLoss()
        
        self.dice_weight = dice_weight
    
    def forward(
        self,
        pred_signal: torch.Tensor,
        target_signal: torch.Tensor,
        pred_seg: torch.Tensor,
        target_seg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined multi-task loss.
        
        Args:
            pred_signal: Predicted denoised signal (B, 1, L)
            target_signal: Ground truth clean signal (B, 1, L) 
            pred_seg: Predicted segmentation logits (B, C, L)
            target_seg: Ground truth segmentation mask (B, L)
        
        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        # Reconstruction loss (Smooth L1 for robustness)
        loss_recon = self.recon_loss(pred_signal, target_signal)
        
        # Segmentation loss (CE + Dice)
        loss_ce = self.ce_loss(pred_seg, target_seg)
        loss_dice = self.dice_loss(pred_seg, target_seg)
        loss_seg = loss_ce + self.dice_weight * loss_dice
        
        # Extract variances from log-variances
        var_recon = torch.exp(self.log_var_recon)
        var_seg = torch.exp(self.log_var_seg)
        
        # Uncertainty-weighted combination
        # L = (1/(2σ²)) * L_task + log(σ) = (1/(2σ²)) * L_task + 0.5 * log(σ²)
        weighted_recon = loss_recon / (2.0 * var_recon) + 0.5 * self.log_var_recon
        weighted_seg = loss_seg / (2.0 * var_seg) + 0.5 * self.log_var_seg
        
        total_loss = weighted_recon + weighted_seg
        
        # Compute effective weights for logging
        weight_recon = 1.0 / (2.0 * var_recon)
        weight_seg = 1.0 / (2.0 * var_seg)
        
        loss_dict = {
            'total': total_loss,
            'recon': loss_recon,
            'ce': loss_ce,
            'dice': loss_dice,
            'seg': loss_seg,
            'weighted_recon': weighted_recon,
            'weighted_seg': weighted_seg,
            'sigma_recon': torch.sqrt(var_recon),
            'sigma_seg': torch.sqrt(var_seg),
            'weight_recon': weight_recon,
            'weight_seg': weight_seg,
        }
        
        return total_loss, loss_dict
    
    def get_task_weights(self) -> Tuple[float, float]:
        """Get current effective task weights."""
        var_recon = torch.exp(self.log_var_recon)
        var_seg = torch.exp(self.log_var_seg)
        
        weight_recon = 1.0 / (2.0 * var_recon.item())
        weight_seg = 1.0 / (2.0 * var_seg.item())
        
        return weight_recon, weight_seg


class FixedWeightMultiTaskLoss(nn.Module):
    """
    Multi-task loss with fixed (non-learnable) weights.
    
    Useful for comparison or when uncertainty learning is not desired.
    
    Args:
        recon_weight: Weight for reconstruction loss
        seg_weight: Weight for segmentation loss
        dice_weight: Weight for Dice loss component
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        seg_weight: float = 1.0,
        dice_weight: float = 1.0
    ):
        super().__init__()
        
        self.recon_weight = recon_weight
        self.seg_weight = seg_weight
        self.dice_weight = dice_weight
        
        self.recon_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = GeneralizedDiceLoss()
    
    def forward(
        self,
        pred_signal: torch.Tensor,
        target_signal: torch.Tensor,
        pred_seg: torch.Tensor,
        target_seg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss with fixed weights."""
        loss_recon = self.recon_loss(pred_signal, target_signal)
        loss_ce = self.ce_loss(pred_seg, target_seg)
        loss_dice = self.dice_loss(pred_seg, target_seg)
        loss_seg = loss_ce + self.dice_weight * loss_dice
        
        total = self.recon_weight * loss_recon + self.seg_weight * loss_seg
        
        return total, {
            'total': total,
            'recon': loss_recon,
            'ce': loss_ce,
            'dice': loss_dice,
            'seg': loss_seg,
        }


# =============================================================================
# SOTA Loss Functions for MR-TAE-Fusion
# =============================================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Smooth L1) for robust PD signal denoising.
    
    Unlike MSE which penalizes large errors quadratically, Charbonnier loss
    is more robust to high-amplitude impulse noise outliers common in PD signals.
    
    Formula: L = sqrt((pred - target)^2 + epsilon^2)
    
    This is a differentiable approximation of L1 that is smooth near zero,
    providing stable gradients while maintaining robustness to outliers.
    
    Args:
        epsilon: Small constant for numerical stability and smoothness
    """
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.eps_sq = epsilon ** 2
    
    def forward(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Charbonnier loss.
        
        Args:
            prediction: Predicted denoised signal
            target: Ground truth clean signal
        
        Returns:
            Scalar loss value
        """
        diff = prediction - target
        # Charbonnier: sqrt(diff^2 + eps^2)
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return torch.mean(loss)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in segmentation.
    
    Reduces the relative loss for well-classified examples,
    focusing training on hard negatives (e.g., pulse boundaries).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class balancing factor
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            inputs: Predicted logits (B, C, L)
            targets: Target labels (B, L)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities for correct class
        p_t = torch.exp(-ce_loss)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Weighted loss
        loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class JointLoss(nn.Module):
    """
    Joint Multi-Task Loss: Denoising (Charbonnier) + Segmentation (CCE + Dice).
    
    Combines robust Charbonnier loss for denoising with Categorical Cross-Entropy
    and Dice loss for segmentation. Uses fixed weights for simplicity.
    
    This is the SOTA loss recommended for PD signal processing, as:
    - Charbonnier handles impulsive noise outliers better than MSE
    - CCE provides pixel-wise classification
    - Dice handles class imbalance
    
    Args:
        alpha: Weight for denoising loss
        beta: Weight for segmentation loss  
        dice_weight: Weight for Dice component in segmentation
        use_focal: Whether to use Focal Loss instead of CCE
        focal_gamma: Gamma parameter for Focal Loss
        charbonnier_eps: Epsilon for Charbonnier loss
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        dice_weight: float = 1.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        charbonnier_eps: float = 1e-3
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.dice_weight = dice_weight
        
        # Denoising loss (Charbonnier for robustness)
        self.denoise_loss = CharbonnierLoss(epsilon=charbonnier_eps)
        
        # Segmentation losses
        if use_focal:
            self.ce_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.dice_loss = GeneralizedDiceLoss()
    
    def forward(
        self,
        denoise_pred: torch.Tensor,
        denoise_target: torch.Tensor,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined joint loss.
        
        Args:
            denoise_pred: Predicted denoised signal (B, 1, L) or (B, L)
            denoise_target: Ground truth clean signal (B, 1, L) or (B, L)
            seg_pred: Predicted segmentation logits (B, C, L)
            seg_target: Ground truth segmentation mask (B, L) as LongTensor
        
        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
        # 1. Denoising Loss (Charbonnier)
        l_denoise = self.denoise_loss(denoise_pred, denoise_target)
        
        # 2. Segmentation Loss (CCE/Focal + Dice)
        l_ce = self.ce_loss(seg_pred, seg_target)
        l_dice = self.dice_loss(seg_pred, seg_target)
        l_seg = l_ce + self.dice_weight * l_dice
        
        # 3. Total Loss
        total_loss = (self.alpha * l_denoise) + (self.beta * l_seg)
        
        # Return loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'denoise': l_denoise.item(),
            'ce': l_ce.item(),
            'dice': l_dice.item(),
            'seg': l_seg.item(),
        }
        
        return total_loss, loss_dict


class JointLossWithUncertainty(nn.Module):
    """
    Joint Multi-Task Loss with Learnable Uncertainty Weighting.
    
    Combines the robustness of Charbonnier loss with automatic task
    balancing via homoscedastic uncertainty.
    
    This is the most advanced loss option, combining:
    - Charbonnier for robust denoising
    - CCE + Dice for segmentation
    - Learnable uncertainty weights for automatic balancing
    
    Args:
        initial_sigma_denoise: Initial uncertainty for denoising
        initial_sigma_seg: Initial uncertainty for segmentation
        dice_weight: Weight for Dice loss component
        charbonnier_eps: Epsilon for Charbonnier loss
    """
    
    def __init__(
        self,
        initial_sigma_denoise: float = 1.0,
        initial_sigma_seg: float = 1.0,
        dice_weight: float = 1.0,
        charbonnier_eps: float = 1e-3
    ):
        super().__init__()
        
        # Learnable log-variance parameters
        self.log_var_denoise = nn.Parameter(
            torch.tensor(2.0 * torch.log(torch.tensor(initial_sigma_denoise)))
        )
        self.log_var_seg = nn.Parameter(
            torch.tensor(2.0 * torch.log(torch.tensor(initial_sigma_seg)))
        )
        
        # Loss functions
        self.denoise_loss = CharbonnierLoss(epsilon=charbonnier_eps)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = GeneralizedDiceLoss()
        
        self.dice_weight = dice_weight
    
    def forward(
        self,
        denoise_pred: torch.Tensor,
        denoise_target: torch.Tensor,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss with uncertainty weighting.
        
        Args:
            denoise_pred: Predicted denoised signal
            denoise_target: Ground truth clean signal
            seg_pred: Predicted segmentation logits
            seg_target: Ground truth segmentation mask
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Compute base losses
        l_denoise = self.denoise_loss(denoise_pred, denoise_target)
        l_ce = self.ce_loss(seg_pred, seg_target)
        l_dice = self.dice_loss(seg_pred, seg_target)
        l_seg = l_ce + self.dice_weight * l_dice
        
        # Get variances
        var_denoise = torch.exp(self.log_var_denoise)
        var_seg = torch.exp(self.log_var_seg)
        
        # Uncertainty-weighted loss
        weighted_denoise = l_denoise / (2.0 * var_denoise) + 0.5 * self.log_var_denoise
        weighted_seg = l_seg / (2.0 * var_seg) + 0.5 * self.log_var_seg
        
        total_loss = weighted_denoise + weighted_seg
        
        loss_dict = {
            'total': total_loss,
            'denoise': l_denoise,
            'ce': l_ce,
            'dice': l_dice,
            'seg': l_seg,
            'weighted_denoise': weighted_denoise,
            'weighted_seg': weighted_seg,
            'sigma_denoise': torch.sqrt(var_denoise),
            'sigma_seg': torch.sqrt(var_seg),
        }
        
        return total_loss, loss_dict
    
    def get_task_weights(self) -> Tuple[float, float]:
        """Get current effective task weights."""
        var_denoise = torch.exp(self.log_var_denoise)
        var_seg = torch.exp(self.log_var_seg)
        
        weight_denoise = 1.0 / (2.0 * var_denoise.item())
        weight_seg = 1.0 / (2.0 * var_seg.item())
        
        return weight_denoise, weight_seg
