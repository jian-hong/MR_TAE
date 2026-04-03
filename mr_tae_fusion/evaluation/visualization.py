"""
Visualization utilities for PD signal analysis.

Provides plotting functions for:
- Reconstruction comparison
- Segmentation overlay
- SNR performance curves
- Training history
- Spectral analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union, Tuple
import torch


def plot_reconstruction(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray,
    t: Optional[np.ndarray] = None,
    title: str = "Signal Reconstruction",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot signal reconstruction comparison.
    
    3-row plot showing noisy input, model output, and ground truth.
    
    Args:
        noisy: Noisy input signal
        denoised: Denoised output signal
        clean: Clean ground truth
        t: Time vector (optional)
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if isinstance(noisy, torch.Tensor):
        noisy = noisy.detach().cpu().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.detach().cpu().numpy()
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    
    noisy = noisy.flatten()
    denoised = denoised.flatten()
    clean = clean.flatten()
    
    if t is None:
        t = np.arange(len(noisy))
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Noisy input
    axes[0].plot(t, noisy, 'r', alpha=0.8, linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Noisy Input')
    axes[0].grid(True, alpha=0.3)
    
    # Denoised output
    axes[1].plot(t, denoised, 'b', linewidth=0.8)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Denoised Output (MR-TAE-Fusion)')
    axes[1].grid(True, alpha=0.3)
    
    # Clean ground truth
    axes[2].plot(t, clean, 'g', linewidth=0.8)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Clean Ground Truth')
    axes[2].set_xlabel('Time (samples)')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_segmentation_overlay(
    signal: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    t: Optional[np.ndarray] = None,
    title: str = "Segmentation Overlay",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot signal with segmentation mask overlay.
    
    Args:
        signal: Input signal
        pred_mask: Predicted segmentation mask
        target_mask: Optional ground truth mask
        class_names: Class label names
        t: Time vector
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if target_mask is not None and isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()
    
    signal = signal.flatten()
    pred_mask = pred_mask.flatten()
    
    if t is None:
        t = np.arange(len(signal))
    
    if class_names is None:
        class_names = ['Background', 'Corona', 'Surface', 'Internal']
    
    # Color map for classes
    colors = ['white', 'red', 'blue', 'green', 'orange', 'purple']
    
    num_plots = 2 if target_mask is not None else 1
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    
    if num_plots == 1:
        axes = [axes]
    
    # Predicted segmentation
    ax = axes[0]
    ax.plot(t, signal, 'k', linewidth=0.5, alpha=0.7)
    
    for c in range(1, len(class_names)):
        mask = pred_mask == c
        if mask.any():
            ax.fill_between(t, signal.min(), signal.max(),
                          where=mask, alpha=0.3, color=colors[c],
                          label=class_names[c])
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Predicted Segmentation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Ground truth (if provided)
    if target_mask is not None:
        target_mask = target_mask.flatten()
        ax = axes[1]
        ax.plot(t, signal, 'k', linewidth=0.5, alpha=0.7)
        
        for c in range(1, len(class_names)):
            mask = target_mask == c
            if mask.any():
                ax.fill_between(t, signal.min(), signal.max(),
                              where=mask, alpha=0.3, color=colors[c],
                              label=class_names[c])
        
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Time (samples)')
        ax.set_title('Ground Truth Segmentation')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_snr_curve(
    snr_values: np.ndarray,
    iou_values: np.ndarray,
    ncc_values: np.ndarray,
    title: str = "Performance vs SNR",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot model performance as function of input SNR.
    
    Characterizes the "breakdown point" of the model.
    
    Args:
        snr_values: Input SNR levels (dB)
        iou_values: IoU at each SNR level
        ncc_values: NCC at each SNR level
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Input SNR (dB)')
    ax1.set_ylabel('IoU', color=color1)
    ax1.plot(snr_values, iou_values, 'o-', color=color1, label='IoU')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('NCC', color=color2)
    ax2.plot(snr_values, ncc_values, 's-', color=color2, label='NCC')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add horizontal line at typical thresholds
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='IoU=0.5')
    ax2.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='NCC=0.9')
    
    plt.title(title)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with loss histories
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Total loss
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction loss
    ax = axes[0, 1]
    if 'train_recon' in history:
        ax.plot(epochs, history['train_recon'], 'b-', label='Train')
    if 'val_recon' in history:
        ax.plot(epochs, history['val_recon'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Segmentation loss
    ax = axes[1, 0]
    if 'train_seg' in history:
        ax.plot(epochs, history['train_seg'], 'b-', label='Train')
    if 'val_seg' in history:
        ax.plot(epochs, history['val_seg'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Segmentation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Task weights (if available)
    ax = axes[1, 1]
    if 'task_weights' in history and history['task_weights']:
        recon_weights = [w['recon'] for w in history['task_weights']]
        seg_weights = [w['seg'] for w in history['task_weights']]
        ax.plot(epochs, recon_weights, 'g-', label='Recon Weight')
        ax.plot(epochs, seg_weights, 'm-', label='Seg Weight')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('Learned Task Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_spectral_analysis(
    clean_signal: np.ndarray,
    impulsive_noise: np.ndarray,
    colored_noise: np.ndarray,
    fs: float = 1e9,
    title: str = "Spectral Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot FFT spectral analysis of signals and noise.
    
    Demonstrates spectral overlap that renders simple filtering ineffective.
    
    Args:
        clean_signal: Clean PD signal
        impulsive_noise: Impulsive noise component
        colored_noise: Colored/TGAN noise
        fs: Sampling frequency (Hz)
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if isinstance(clean_signal, torch.Tensor):
        clean_signal = clean_signal.detach().cpu().numpy()
    if isinstance(impulsive_noise, torch.Tensor):
        impulsive_noise = impulsive_noise.detach().cpu().numpy()
    if isinstance(colored_noise, torch.Tensor):
        colored_noise = colored_noise.detach().cpu().numpy()
    
    n = len(clean_signal)
    freq = np.fft.rfftfreq(n, 1/fs)
    
    # Compute FFTs
    fft_clean = np.abs(np.fft.rfft(clean_signal))
    fft_impulsive = np.abs(np.fft.rfft(impulsive_noise))
    fft_colored = np.abs(np.fft.rfft(colored_noise))
    
    # Convert to dB
    fft_clean_db = 20 * np.log10(fft_clean + 1e-10)
    fft_impulsive_db = 20 * np.log10(fft_impulsive + 1e-10)
    fft_colored_db = 20 * np.log10(fft_colored + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(freq / 1e6, fft_clean_db, 'g-', alpha=0.8, label='Clean PD Signal')
    ax.plot(freq / 1e6, fft_impulsive_db, 'r-', alpha=0.6, label='Impulsive Noise')
    ax.plot(freq / 1e6, fft_colored_db, 'b-', alpha=0.6, label='Colored/TGAN Noise')
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, fs / 2 / 1e6])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Pixel-level Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix for segmentation.
    
    Args:
        pred_mask: Predicted labels
        target_mask: Ground truth labels
        class_names: Class names
        title: Figure title
        save_path: Path to save
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()
    
    pred_mask = pred_mask.flatten()
    target_mask = target_mask.flatten()
    
    if class_names is None:
        class_names = ['Background', 'Corona', 'Surface', 'Internal']
    
    num_classes = len(class_names)
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target_mask, pred_mask):
        if t < num_classes and p < num_classes:
            cm[t, p] += 1
    
    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm_norm, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Labels
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = f'{cm_norm[i, j]:.2f}\n({cm[i, j]})'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if cm_norm[i, j] > 0.5 else 'black',
                   fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
