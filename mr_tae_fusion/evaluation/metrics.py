"""
Evaluation metrics for PD signal denoising and segmentation.

Implements:
- SNR improvement
- Normalized Cross-Correlation (NCC) for shape fidelity
- RMSE for amplitude error
- IoU and Dice for segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass


def calculate_snr(
    signal: Union[np.ndarray, torch.Tensor],
    noise: Optional[Union[np.ndarray, torch.Tensor]] = None,
    clean: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.
    
    SNR = 10 * log10(signal_power / noise_power)
    
    Args:
        signal: Either clean signal (if noise provided) or noisy signal
        noise: Noise component (if None, uses signal-clean as noise)
        clean: Clean reference signal
    
    Returns:
        SNR in dB
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if noise is not None and isinstance(noise, torch.Tensor):
        noise = noise.detach().cpu().numpy()
    if clean is not None and isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    
    signal = signal.flatten()
    
    if noise is not None:
        noise = noise.flatten()
    elif clean is not None:
        clean = clean.flatten()
        noise = signal - clean
    else:
        raise ValueError("Must provide either noise or clean reference")
    
    signal_power = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    
    return 10 * np.log10(signal_power / noise_power)


def calculate_snr_improvement(
    noisy: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
    clean: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate SNR improvement from denoising.
    
    SNR_imp = SNR_out - SNR_in
    
    Args:
        noisy: Noisy input signal
        denoised: Denoised output signal
        clean: Clean ground truth signal
    
    Returns:
        SNR improvement in dB
    """
    snr_in = calculate_snr(clean, noise=noisy - clean)
    snr_out = calculate_snr(clean, noise=denoised - clean)
    
    return snr_out - snr_in


def calculate_ncc(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Normalized Cross-Correlation.
    
    Measures shape fidelity between predicted and target signals.
    NCC = Σ(pred * target) / √(Σ pred² * Σ target²)
    
    Args:
        pred: Predicted signal
        target: Target signal
    
    Returns:
        NCC value in [0, 1] (higher is better)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.flatten()
    target = target.flatten()
    
    # Zero-mean normalization
    pred = pred - np.mean(pred)
    target = target - np.mean(target)
    
    numerator = np.sum(pred * target)
    denominator = np.sqrt(np.sum(pred ** 2) * np.sum(target ** 2)) + 1e-10
    
    return abs(numerator / denominator)


def calculate_rmse(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        pred: Predicted signal
        target: Target signal
    
    Returns:
        RMSE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = pred.flatten()
    target = target.flatten()
    
    return np.sqrt(np.mean((pred - target) ** 2))


def calculate_iou(
    pred_mask: Union[np.ndarray, torch.Tensor],
    target_mask: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    ignore_background: bool = False
) -> Dict[str, float]:
    """
    Calculate Intersection over Union (IoU) for segmentation.
    
    Args:
        pred_mask: Predicted class labels (B, L) or (L,)
        target_mask: Target class labels (B, L) or (L,)
        num_classes: Number of classes
        ignore_background: Whether to exclude class 0
    
    Returns:
        Dictionary with per-class and mean IoU
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()
    
    pred_mask = pred_mask.flatten()
    target_mask = target_mask.flatten()
    
    results = {}
    ious = []
    
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        pred_c = pred_mask == c
        target_c = target_mask == c
        
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        
        if union > 0:
            iou = intersection / union
            results[f'iou_class_{c}'] = iou
            ious.append(iou)
    
    results['mean_iou'] = np.mean(ious) if ious else 0.0
    
    return results


def calculate_dice(
    pred_mask: Union[np.ndarray, torch.Tensor],
    target_mask: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    ignore_background: bool = False
) -> Dict[str, float]:
    """
    Calculate Dice Score for segmentation.
    
    Dice = 2 * |P ∩ T| / (|P| + |T|)
    
    Args:
        pred_mask: Predicted class labels
        target_mask: Target class labels
        num_classes: Number of classes
        ignore_background: Whether to exclude class 0
    
    Returns:
        Dictionary with per-class and mean Dice
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()
    
    pred_mask = pred_mask.flatten()
    target_mask = target_mask.flatten()
    
    results = {}
    dices = []
    
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        pred_c = pred_mask == c
        target_c = target_mask == c
        
        intersection = np.logical_and(pred_c, target_c).sum()
        total = pred_c.sum() + target_c.sum()
        
        if total > 0:
            dice = 2 * intersection / total
            results[f'dice_class_{c}'] = dice
            dices.append(dice)
    
    results['mean_dice'] = np.mean(dices) if dices else 0.0
    
    return results


def calculate_pulse_detection_rate(
    pred_mask: Union[np.ndarray, torch.Tensor],
    target_mask: Union[np.ndarray, torch.Tensor],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Pulse Detection Rate (PDR).
    
    A pulse is considered detected if its IoU with ground truth > threshold.
    
    Args:
        pred_mask: Predicted segmentation
        target_mask: Target segmentation
        iou_threshold: IoU threshold for detection
    
    Returns:
        Detection rate in [0, 1]
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()
    
    pred_mask = pred_mask.flatten()
    target_mask = target_mask.flatten()
    
    # Find pulse regions in target (non-zero)
    target_nonzero = target_mask > 0
    
    # Find connected components (simple approach: run-length)
    pulse_starts = np.where(np.diff(np.concatenate([[0], target_nonzero.astype(int)])) == 1)[0]
    pulse_ends = np.where(np.diff(np.concatenate([target_nonzero.astype(int), [0]])) == -1)[0]
    
    if len(pulse_starts) == 0:
        return 1.0  # No pulses to detect
    
    detected = 0
    total = len(pulse_starts)
    
    for start, end in zip(pulse_starts, pulse_ends):
        pred_region = pred_mask[start:end + 1] > 0
        target_region = target_mask[start:end + 1] > 0
        
        intersection = np.logical_and(pred_region, target_region).sum()
        union = np.logical_or(pred_region, target_region).sum()
        
        iou = intersection / (union + 1e-10)
        
        if iou >= iou_threshold:
            detected += 1
    
    return detected / total


def evaluate_denoising(
    noisy: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
    clean: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Comprehensive denoising evaluation.
    
    Args:
        noisy: Noisy input
        denoised: Model output
        clean: Ground truth
    
    Returns:
        Dictionary of metrics
    """
    return {
        'snr_input': calculate_snr(clean, noise=noisy - clean),
        'snr_output': calculate_snr(clean, noise=denoised - clean),
        'snr_improvement': calculate_snr_improvement(noisy, denoised, clean),
        'ncc': calculate_ncc(denoised, clean),
        'rmse': calculate_rmse(denoised, clean),
    }


def evaluate_segmentation(
    pred_logits: Union[np.ndarray, torch.Tensor],
    target_mask: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4
) -> Dict[str, float]:
    """
    Comprehensive segmentation evaluation.
    
    Args:
        pred_logits: Model logits (B, C, L) or probabilities
        target_mask: Ground truth labels (B, L)
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    if isinstance(pred_logits, torch.Tensor):
        if pred_logits.dim() == 3:
            pred_mask = pred_logits.argmax(dim=1)
        else:
            pred_mask = pred_logits
        pred_mask = pred_mask.detach().cpu().numpy()
    else:
        if pred_logits.ndim == 3:
            pred_mask = pred_logits.argmax(axis=1)
        else:
            pred_mask = pred_logits
    
    iou_results = calculate_iou(pred_mask, target_mask, num_classes, ignore_background=True)
    dice_results = calculate_dice(pred_mask, target_mask, num_classes, ignore_background=True)
    pdr = calculate_pulse_detection_rate(pred_mask, target_mask)
    
    return {
        **iou_results,
        **dice_results,
        'pulse_detection_rate': pdr,
    }


@dataclass
class PDMetrics:
    """Container for PD evaluation metrics."""
    
    # Denoising metrics
    snr_input: float = 0.0
    snr_output: float = 0.0
    snr_improvement: float = 0.0
    ncc: float = 0.0
    rmse: float = 0.0
    
    # Segmentation metrics
    mean_iou: float = 0.0
    mean_dice: float = 0.0
    pulse_detection_rate: float = 0.0
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PDMetrics':
        return cls(
            snr_input=d.get('snr_input', 0.0),
            snr_output=d.get('snr_output', 0.0),
            snr_improvement=d.get('snr_improvement', 0.0),
            ncc=d.get('ncc', 0.0),
            rmse=d.get('rmse', 0.0),
            mean_iou=d.get('mean_iou', 0.0),
            mean_dice=d.get('mean_dice', 0.0),
            pulse_detection_rate=d.get('pulse_detection_rate', 0.0),
        )
    
    def to_dict(self) -> Dict:
        return {
            'snr_input': self.snr_input,
            'snr_output': self.snr_output,
            'snr_improvement': self.snr_improvement,
            'ncc': self.ncc,
            'rmse': self.rmse,
            'mean_iou': self.mean_iou,
            'mean_dice': self.mean_dice,
            'pulse_detection_rate': self.pulse_detection_rate,
        }
    
    def __str__(self) -> str:
        return (
            f"=== Denoising Metrics ===\n"
            f"SNR Input:       {self.snr_input:.2f} dB\n"
            f"SNR Output:      {self.snr_output:.2f} dB\n"
            f"SNR Improvement: {self.snr_improvement:.2f} dB\n"
            f"NCC (Shape):     {self.ncc:.4f}\n"
            f"RMSE:            {self.rmse:.4f}\n"
            f"\n=== Segmentation Metrics ===\n"
            f"Mean IoU:        {self.mean_iou:.4f}\n"
            f"Mean Dice:       {self.mean_dice:.4f}\n"
            f"Pulse Detection: {self.pulse_detection_rate:.2%}\n"
        )
