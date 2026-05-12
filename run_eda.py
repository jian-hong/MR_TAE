"""
EDA Script - Analyze Q.lin PD datasets
Run this to explore the data structure and characteristics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

# Data paths
DATA_DIR = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13')
QLIN_DIR = DATA_DIR / 'Datset from Q.lin'

print("="*70)
print("PD SIGNAL EDA - Q.lin Dataset Analysis")
print("="*70)

# Load mat file function
def load_mat_file(filepath):
    """Load .mat file, handling both v7 and v7.3 formats."""
    filepath = Path(filepath)
    
    try:
        import scipy.io as sio
        mat_data = sio.loadmat(str(filepath))
        data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        print(f"  Loaded with scipy.io")
        return data
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"  scipy.io failed: {e}")
    
    try:
        import mat73
        data = mat73.loadmat(str(filepath))
        print(f"  Loaded with mat73")
        return data
    except Exception as e:
        print(f"  mat73 failed: {e}")
    
    try:
        import h5py
        data = {}
        with h5py.File(str(filepath), 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        print(f"  Loaded with h5py")
        return data
    except Exception as e:
        print(f"  h5py failed: {e}")
    
    raise RuntimeError(f"Could not load {filepath}")

# Load Q.lin datasets
qlin_files = {
    '1.0mm': QLIN_DIR / 'data1.0mm.mat',
    '1.8mm': QLIN_DIR / 'data1.8mm.mat', 
    '2.0mm': QLIN_DIR / 'data2.0mm.mat',
    '2.5mm': QLIN_DIR / 'data2.5mm.mat',
}

datasets = {}
for label, filepath in qlin_files.items():
    if filepath.exists():
        print(f"\nLoading {label}...")
        try:
            data = load_mat_file(filepath)
            print(f"  Variables: {list(data.keys())}")
            
            main_var = None
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    if main_var is None or value.size > data.get(main_var, np.array([])).size:
                        main_var = key
            
            if main_var:
                datasets[label] = data[main_var]
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"\nFile not found: {filepath}")

print(f"\n" + "="*70)
print(f"LOADED {len(datasets)} DATASETS")
print("="*70)

# Dataset Summary
print(f"\n{'Dataset':<10} {'Shape':<25} {'Min':>12} {'Max':>12} {'Mean':>12}")
print("-"*70)
for label, data in datasets.items():
    if isinstance(data, np.ndarray):
        shape_str = str(data.shape)
        print(f"{label:<10} {shape_str:<25} {data.min():>12.4f} {data.max():>12.4f} {data.mean():>12.4f}")

# Check model compatibility
print(f"\n" + "="*70)
print("MODEL COMPATIBILITY CHECK")
print("="*70)
REQUIRED_LENGTH = 2001

for label, data in datasets.items():
    if data.ndim == 1:
        signal_len = len(data)
        n_signals = 1
    else:
        signal_len = data.shape[1] if data.shape[0] < data.shape[1] else data.shape[0]
        n_signals = data.shape[0] if data.shape[0] < data.shape[1] else data.shape[1]
    
    status = "✓ COMPATIBLE" if signal_len >= REQUIRED_LENGTH else "✗ TOO SHORT"
    print(f"{label}: {n_signals} signals × {signal_len} samples - {status}")

# Plot sample signals
print(f"\n" + "="*70)
print("GENERATING VISUALIZATION...")
print("="*70)

fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 3 * len(datasets)), sharex=True)
if len(datasets) == 1:
    axes = [axes]

colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

for idx, (label, data) in enumerate(datasets.items()):
    ax = axes[idx]
    
    if data.ndim == 1:
        sample = data
    else:
        sample = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
    
    plot_len = min(2001, len(sample))
    ax.plot(sample[:plot_len], color=colors[idx % len(colors)], linewidth=0.8, alpha=0.9)
    ax.set_ylabel(f'{label}', fontsize=12, fontweight='bold')
    ax.set_title(f'PD Signal - {label} Gap Distance (First {plot_len} samples)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    stats_text = f'Min: {sample.min():.2f}  Max: {sample.max():.2f}  Std: {sample.std():.4f}'
    ax.text(0.99, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[-1].set_xlabel('Sample Index', fontsize=12)
plt.tight_layout()
plt.suptitle('Q.lin Dataset - PD Signals by Gap Distance', fontsize=14, fontweight='bold', y=1.02)
plt.savefig(DATA_DIR / 'eda_signal_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {DATA_DIR / 'eda_signal_comparison.png'}")

# Try to load noisy data info (just check file size, don't load full data)
noisy_path = DATA_DIR / 'noisy_minus10dB_18mm.mat'
if noisy_path.exists():
    print(f"\nNoisy data file: {noisy_path.stat().st_size / 1e9:.2f} GB")
    print("  (Too large to load in full - use subsampling for analysis)")

print(f"\n" + "="*70)
print("EDA COMPLETE")
print("="*70)
