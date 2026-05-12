# Full Research Plan — AE-PD Denoising Pipeline
## MR-TAE v2: Corrected Signal Physics, Ablation Studies & Automated Training
**Hardware Target: RTX 4070 Laptop (12GB VRAM)**

---

## PART 0: CRITICAL CORRECTIONS FROM IMAGE ANALYSIS

### 0.1 — What Your Real Data Actually Shows

Studying the Q.Lin visualizations reveals several important facts your current code violates:

**Pulse Morphology (Images 1 & 4):**
- Onset is sharp: 1–3 samples rise time (~1–3 µs at 1 MHz sampling rate)
- The decay tail is LONG: ~500–1500 µs of decaying sinusoidal oscillation
- Carrier frequency from cycle counting: approximately **50–150 kHz** (not MHz!)
- The tail has ENVELOPE MODULATION — it is not a clean single-frequency sine. It beats slightly, suggesting multipath arrivals arriving ~50–200 µs after the initial wavefront
- Amplitude range: 0.3 V to >5 V peak (varies by defect size)

**Per-Class Structure (Images 2 & 3):**
- **1.0 mm bead**: near-continuous dense discharge with very small amplitude (~0.03–0.06 V). Looks like low-level surface streamer activity. Very different from other classes.
- **1.8 mm**: sparse, high-amplitude impulsive events with long oscillatory tails
- **2.0 mm**: similar to 1.8mm but slightly different inter-event timing and amplitude
- **2.5 mm**: similar sparse pattern, slightly higher peak amplitudes

**CRITICAL LABEL MISMATCH — Must Fix:**
Your synthetic generator produces TYPE labels (Corona / Internal / Surface / Treeing / etc.)
Your real data (Q.Lin) produces SEVERITY labels (1.0mm / 1.8mm / 2.0mm / 2.5mm)
These are DIFFERENT classification schemes. Your thesis labels them as severity-based surface discharge, not type-based discharge. You need to make a design decision:
- **Option A**: Relabel real data classes as "Severity 1–4" and train a severity classifier
- **Option B**: Keep type-based classification for synthetic data, test denoising on real data without classification alignment (denoising-only evaluation on real data)
- **Recommendation**: Option B — use real data purely for denoising quality validation (NCC/RMSE), NOT for segmentation accuracy. The segmentation head is validated on synthetic data only.

---

### 0.2 — Pulse Generation Critique & Corrections

#### ✅ What is CORRECT
- DOP model `A·exp(-t/τ)·sin(2πfct+φ)` is the correct physical model for AE pulses
- The concept of generating clean pulses then adding noise is correct
- Using segmentation masks alongside signals is correct

#### ❌ What is WRONG — Fix These

**Error 1: Frequency range is MHz, should be kHz**
```python
# WRONG — currently in your code:
fc = 40e6 + np.random.rand() * 80e6   # 40–120 MHz — THIS IS UHF ELECTRICAL PD, NOT AE

# CORRECT for AE:
fc = 50e3 + np.random.rand() * 200e3  # 50–250 kHz — AE band
```
This is the single most important fix. AE sensors operate at 20–500 kHz. Using MHz frequencies means your synthetic pulses bear zero resemblance to real AE signals.

**Error 2: Decay constant τ is too short**
```python
# WRONG — if you're using nanoseconds:
tau = 10e-9 + np.random.rand() * 20e-9  # ns scale

# CORRECT — from image analysis, tails last ~500–1500 µs:
tau = 200e-6 + np.random.rand() * 1000e-6  # 200–1200 µs
```

**Error 3: Missing transfer function (multipath / reverberation)**
The images clearly show that real pulses have a secondary oscillation ~100–300 µs AFTER the main arrival. This is acoustic multipath from tank wall reflections. You must model this:
```python
def apply_acoustic_transfer_function(signal, fs, num_reflections=2):
    """
    Convolve clean AE pulse with a simple impulse response modeling:
    - Direct path (main pulse)
    - 1-3 reflections arriving later, attenuated
    """
    h = np.zeros(int(0.005 * fs))  # 5ms impulse response
    h[0] = 1.0  # direct path
    for i in range(num_reflections):
        delay = int((100e-6 + i * 150e-6 + np.random.rand() * 100e-6) * fs)
        if delay < len(h):
            h[delay] = (0.3 - i * 0.1) * (0.8 + np.random.rand() * 0.4)
    return np.convolve(signal, h, mode='same')
```

**Error 4: The 1.0mm class is structurally different**
From Image 2, the 1.0mm signal looks like CONTINUOUS low-amplitude discharge (surface streamer), not sparse impulses. Your generator must produce a different morphology for this class — dense overlapping short pulses rather than sparse large ones.

---

### 0.3 — Type G (Treeing) Analysis

**Is treeing detectable via AE?** Yes — electrical treeing in solid insulation does produce AE signals. However:
- It occurs in SOLID insulation (epoxy, XLPE cable insulation), not primarily in oil-filled GIS
- The Q.Lin dataset models surface discharge with metal particles in a GIS-like environment — treeing is a different failure mode
- Your code uses `fc = 40e6 to 120e6 Hz` which is completely wrong for AE treeing signals

**Recommendation:** Remove Type G (Treeing) from your model OR correct it:
```python
# If keeping treeing for AE context:
# Treeing AE characteristics:
# - Lower amplitude than surface discharge
# - Higher repetition rate (hundreds of events per second)
# - Carrier: 80–200 kHz
# - Short τ (~50–200 µs, shorter tails than surface discharge)
fc = 80e3 + np.random.rand() * 120e3   # 80–200 kHz
tau = 50e-6 + np.random.rand() * 150e-6
```
But since Q.Lin has no treeing class, treeing signals will never appear in your test evaluation. Either drop it or clearly mark it as "extended future class — not evaluated on Q.Lin."

---

### 0.4 — Noise Generation Critique

#### ✅ Appropriate for AE
| Noise Type | AE Relevance | Keep? |
|---|---|---|
| White Gaussian Noise | Sensor thermal noise, electronic noise floor | ✅ Keep |
| Bernoulli-Gaussian Impulsive | Mechanical impacts, rain, loose hardware | ✅ Keep |
| Powerline 50/60 Hz | Magnetostriction from core, nearby cables | ✅ Keep |
| Narrowband Interference | Cooling fan resonance, specific mechanical frequencies | ✅ Keep |

#### ❌ NOT needed / Wrong
| Noise Type | Issue |
|---|---|
| "Acoustic Reflection Noise" | Reflections are part of the SIGNAL physics (transfer function), not noise. Model via convolution in signal generation, not as additive noise |
| "AEPD Background Noise" | Redundant with Bernoulli-Gaussian impulsive noise |
| Any MHz-range narrowband | Wrong frequency range for AE |

**One Noise to ADD — Magnetostrictive Hum:**
```python
def generate_magnetostrictive_noise(t, fs, amplitude=0.01):
    """
    GIS transformers produce 100/120 Hz hum from magnetostriction.
    This is a dominant real-world AE background noise source.
    """
    noise = amplitude * np.sin(2 * np.pi * 100 * t)  # 100 Hz for 50Hz grid
    noise += (amplitude * 0.3) * np.sin(2 * np.pi * 200 * t)  # 2nd harmonic
    return noise
```

---

### 0.5 — Activation Function Recommendations for Denoising

| Layer Type | Current (likely ReLU) | Recommended | Reason |
|---|---|---|---|
| CNN encoder/decoder | ReLU | **GELU** | Smoother gradients, better for continuous signal regression |
| BiGRU | tanh/sigmoid (fixed) | Keep default | Standard RNN activations are correct here |
| Swin Transformer | GELU | Keep GELU | Already optimal |
| Final reconstruction layer | Linear/ReLU | **Linear (no activation)** | Signal can be negative — never use ReLU on output |
| Attention gates | Sigmoid | Keep sigmoid | Correct for gate values in [0,1] |

---

## PART 1: PIPELINE ARCHITECTURE

```
Q.Lin Real Data (1MHz, 4 bead sizes)
         │
         ├──[90% Test Only]──────────────────────────────────────────────┐
         │                                                                 │
         └──[10% → WGAN-GP noise extractor]                               │
                    │                                                      │
Synthetic Generator (corrected DOP, AE frequencies)                      │
         │                                                                 │
         └──[+ WGAN noise augmentation]                                   │
                    │                                                      │
         ┌──────────┴──────────┐                                          │
  Training Set             Validation Set                                  │
  (synthetic+augmented)   (synthetic+augmented)                           │
         │                                                                 │
  ┌──────┴──────┐                                                         │
  │   MLflow    │  ← tracks all experiments                               │
  │  Orchestrator│                                                         │
  └──────┬──────┘                                                         │
         │                                                                 │
  ┌──────┴──────┐                                                         │
  │   Optuna   │  ← hyperparameter search                                │
  │   Study    │                                                          │
  └──────┬──────┘                                                         │
         │                                                                 │
  ┌──────┴──────────────────────────────────────────────────┐            │
  │          Curricular Trainer (3 phases)                   │            │
  │  Phase1: SNR [5,15]dB → Phase2: [-5,5]dB → Phase3: [-20,-5]dB       │
  └──────┬──────────────────────────────────────────────────┘            │
         │                                                                 │
  ┌──────┴──────────────────────────────────────────────────┐            │
  │        Model Zoo (all architectures share this trainer)  │            │
  │  MR-TAE-FULL, noBiGRU, noSwin, noAttn, noMTL, noWavelet │            │
  │  MWCNN-BiGRU, MWCNN-Swin, UNet-BiGRU-Swin, UNet-Attn   │            │
  └──────┬──────────────────────────────────────────────────┘            │
         │                                                                 │
  ┌──────┴──────┐                                                         │
  │  Benchmark  │◄────────────────── Real Data Test ─────────────────────┘
  │  Runner     │
  └──────┬──────┘
         │
  ┌──────┴──────┐
  │   SHAP/     │  ← explainability on best model
  │   LIME      │
  └─────────────┘
```

---

## PART 2: FILE STRUCTURE

```
project_root/
├── .gitignore
├── requirements.txt
├── README.md
├── config/
│   ├── signal_config.yaml          # All signal generation parameters
│   ├── noise_config.yaml           # Noise type params and SNR ranges
│   ├── model_registry.yaml         # Model IDs, architectures, flags
│   ├── training_config.yaml        # Epochs, LR, batch size per phase
│   └── hardware_config.yaml        # RTX 4070 memory/heat limits
│
├── data/
│   ├── generators/
│   │   ├── pulse_generator.py      # CORRECTED: AE-accurate DOP pulses
│   │   ├── noise_generator.py      # WGN, BG-Impulse, Powerline, NBI, Magnetostrictive
│   │   ├── transfer_function.py    # Acoustic multipath / tank wall reflection
│   │   └── dataset_builder.py      # Assembles training windows (2048 samples @ 1MHz)
│   ├── wgan/
│   │   ├── wgan_noise.py           # Learns real-world noise distribution
│   │   └── wgan_pulse.py           # Conditional pulse morphology augmentation
│   ├── splitter.py                 # 90/10 real data split (90% test-only)
│   └── qlin_loader.py              # Load and preprocess Lin et al. .mat files
│
├── models/
│   ├── components/
│   │   ├── mwcnn_blocks.py         # DWT/IDWT encoder-decoder blocks (shared)
│   │   ├── attention_gate.py       # Attention gate (shared)
│   │   ├── bigru_module.py         # BiGRU bottleneck (shared)
│   │   ├── swin_1d.py              # 1D Swin Transformer (shared)
│   │   └── seg_head.py             # Segmentation head (shared)
│   ├── base.py                     # BaseDenoiser with AblationConfig
│   ├── pure_cnn.py
│   ├── unet_1d.py
│   ├── mwcnn.py
│   ├── mr_tae.py                   # Full proposed model
│   └── ablations/                  # Auto-generated from flags — not separate files
│
├── training/
│   ├── trainer.py                  # Unified trainer with AMP, gradient checkpointing
│   ├── curricular_scheduler.py     # Phase-aware SNR scheduler
│   ├── loss_functions.py           # Charbonnier, Focal, Dice, Homoscedastic MTL
│   ├── memory_manager.py           # RTX 4070 OOM prevention
│   └── heat_monitor.py             # GPU temperature watchdog
│
├── hpo/
│   ├── optuna_study.py             # Optuna integration with MLflow backend
│   └── search_space.yaml           # Hyperparameter search bounds
│
├── evaluation/
│   ├── benchmark_runner.py         # Master eval: synthetic + real test sets
│   ├── metrics.py                  # SNR, NCC, RMSE, IoU, Dice
│   └── explainability/
│       ├── shap_analysis.py        # SHAP on best model
│       └── lime_analysis.py        # LIME on signal windows
│
├── pipeline/
│   ├── mlflow_orchestrator.py      # Top-level experiment runner
│   ├── auto_retrain.py             # Drift detection + Noise2Void self-supervision
│   └── experiment_tracker.py      # Snapshot images every N epochs
│
├── traditional_baselines/
│   ├── wavelet_threshold.py
│   ├── bayesshrink.py
│   └── savitzky_golay.py
│
├── results/                        # .gitignored
│   └── {experiment_name}/
│       ├── checkpoints/
│       ├── plots/
│       └── mlflow/
│
└── docs/
    ├── SIGNAL_PHYSICS.md
    ├── ARCHITECTURE_SUMMARY.md
    ├── EXAMINER_QA.md
    └── RESULTS_COMPARISON.md
```

---

## PART 3: CORRECTED SIGNAL GENERATION SPECIFICATION

### Window Size Decision
From images: real Q.Lin data is sampled at **1 MHz**. The thesis uses 2048-sample windows.
At 1 MHz, 2048 samples = **2.048 ms** per window.
From Image 4, a full AE pulse (onset + tail) takes ~1500 µs → fits in 2048 samples ✅

### Corrected DOP Parameters (AE-accurate)

```yaml
# config/signal_config.yaml
pulse_types:
  surface_discharge:          # Matches Q.Lin 1.8mm / 2.0mm / 2.5mm
    fc_range_hz: [50000, 200000]    # 50–200 kHz
    tau_range_us: [200, 1200]       # 200–1200 µs decay
    amplitude_range: [0.3, 5.0]
    polarity_bias: 0.6              # Mostly negative onset (from images)
    
  surface_streamer_dense:    # Matches Q.Lin 1.0mm (continuous small events)
    fc_range_hz: [80000, 300000]
    tau_range_us: [50, 200]
    amplitude_range: [0.01, 0.08]
    events_per_second: [500, 2000]  # Dense overlapping
    
  corona:                    # Synthetic only
    fc_range_hz: [20000, 80000]
    tau_range_us: [100, 500]
    amplitude_range: [0.05, 0.5]
    
  internal:                  # Synthetic only
    fc_range_hz: [100000, 400000]
    tau_range_us: [50, 300]
    amplitude_range: [1.0, 8.0]

transfer_function:
  num_reflections: [1, 3]
  reflection_delays_us: [80, 400]
  reflection_attenuation: [0.15, 0.45]
```

---

## PART 4: LOSS FUNCTION STRATEGY

Use all three loss configurations as separate experimental conditions, tracked by MLflow:

### Loss Config A — Current (MTL Charbonnier + Focal)
```python
L_total = (1/2σ²_recon)*L_charb + (1/2σ²_seg)*L_focal + log(σ_recon) + log(σ_seg)
```

### Loss Config B — MTL Charbonnier + Dice
Replace Focal Loss with Dice Loss for segmentation:
```python
L_dice = 1 - (2·TP + ε) / (2·TP + FP + FN + ε)
```
Better than Focal for highly imbalanced temporal masks (PD pulse occupies <5% of signal).

### Loss Config C — Reconstruction Only (Charbonnier + SSIM)
No segmentation head. Add Structural Similarity (SSIM) to Charbonnier:
```python
L_recon = α·L_charb + (1-α)·L_SSIM   # α=0.7
```
SSIM captures perceptual signal structure better than pure L1/L2 — relevant for NCC metric.

### Loss Config D — Perceptual Loss (Feature Matching)
Use the encoder's intermediate features as a perceptual loss term:
```python
L_perceptual = Σ ||encoder_feat_i(clean) - encoder_feat_i(denoised)||₂
```
Prevents over-smoothing by penalizing divergence in learned feature space.

**Test all 4 on same train/test split. MLflow tracks which is best for NCC and RMSE.**

---

## PART 5: RTX 4070 LAPTOP TRAINING SPECIFICATION

### Hardware Profile
- VRAM: 12 GB
- TDP: ~80–115W laptop variant
- Thermal limit: ~83°C before throttle
- Memory bandwidth: ~288 GB/s

### Memory Budget per Model

| Model | Params | Batch32 VRAM | Batch64 VRAM | Safe? |
|---|---|---|---|---|
| Pure CNN | ~2M | ~1.5 GB | ~2.8 GB | ✅ |
| 1D U-Net | ~31M | ~3.5 GB | ~6.5 GB | ✅/⚠️ |
| MWCNN | ~45M | ~4.8 GB | ~9.2 GB | ⚠️ |
| MR-TAE Full | ~65M est. | ~6.5 GB | ~11.8 GB | ⚠️ batch32 |
| MR-TAE + Swin large | ~90M+ | OOM at batch32 | OOM | ❌ |

### Mandatory RTX 4070 Safeguards

```python
# training/memory_manager.py

class RTX4070Manager:
    MAX_VRAM_GB = 11.0          # Leave 1GB headroom
    TEMP_WARNING_C = 78         # Warn at 78°C
    TEMP_THROTTLE_C = 82        # Pause training at 82°C
    TEMP_RESUME_C = 72          # Resume when cooled to 72°C
    
    TRAINING_CONFIG = {
        'pure_cnn':    {'batch': 128, 'grad_ckpt': False, 'amp': True},
        'unet_1d':     {'batch': 64,  'grad_ckpt': False, 'amp': True},
        'mwcnn':       {'batch': 32,  'grad_ckpt': False, 'amp': True},
        'mr_tae_full': {'batch': 32,  'grad_ckpt': True,  'amp': True},
        'ablations':   {'batch': 32,  'grad_ckpt': True,  'amp': True},
    }
```

**Key techniques:**
1. **AMP (Automatic Mixed Precision)**: `torch.cuda.amp.autocast()` — cuts VRAM ~40%, speeds up ~1.5×
2. **Gradient Checkpointing**: For MR-TAE — trades compute for memory in Swin blocks
3. **`pin_memory=True`** on DataLoader for faster CPU→GPU transfer
4. **`torch.backends.cudnn.benchmark = True`** — optimize CUDA kernels for fixed input size
5. **Clear cache between models**: `torch.cuda.empty_cache()` + `gc.collect()` when switching models
6. **Persistent workers**: `num_workers=4, persistent_workers=True`

---

## PART 6: AUTOMATED RETRAINING PIPELINE (Noise2Void Integration)

### When to Retrain
Trigger retraining when:
1. NCC on held-out real validation batch drops >5% from baseline
2. New Q.Lin-format data is placed in `data/incoming/`
3. Weekly scheduled check (cron job or MLflow scheduled run)

### Noise2Void Self-Supervision Strategy
Since real clean ground truth is unavailable for new unlabeled data:
```
For each new unlabeled AE signal window:
1. Mask a random 10% of samples as "blind spots"
2. Train the model to predict the blind spot value from surrounding context
3. This is valid because AE noise is spatially uncorrelated (independent at each sample)
   while the signal (pulse) is spatially correlated (smooth decay)
4. Loss: MSE on masked positions only
```
This allows continual fine-tuning on real unlabeled AE data without requiring clean ground truth.

---

## PART 7: EXPERIMENT SNAPSHOT SCHEDULE

Every **10 epochs**, auto-save:
1. `training_curves_epoch{N}.png` — RMSE + NCC vs. epoch so far
2. `signal_snapshot_epoch{N}.png` — 3-panel: noisy | denoised | clean at -15 dB, -5 dB, +5 dB
3. `seg_snapshot_epoch{N}.png` — segmentation mask prediction vs. ground truth
4. Checkpoint: `checkpoint_epoch{N}.pth`
5. MLflow log: all metrics for that epoch

At training end:
- `snr_sweep_final.png`
- `efficiency_scatter_final.png`
- SHAP summary plot (best model only — compute-expensive)
- `benchmark_final.csv` — all models, all metrics, side by side

---

## PART 8: EVALUATION PROTOCOL — SAME TEST SET FOR ALL

**Both evaluation sets are fixed before any training begins. Never retouched.**

### Test Set 1 — Synthetic AE (Primary)
- 10,000 windows, 2048 samples each
- SNR ∈ {-15, -10, -5, 0, 5} dB, 2000 per SNR level
- All PD types represented
- Generated once, saved as `data/test_synthetic_ae.pt`
- Metrics: SNR_improvement, RMSE, NCC, Dice (segmentation)

### Test Set 2 — Real Q.Lin Data (Domain Transfer)
- 90% of Q.Lin data (all 4 bead sizes, all SNR levels)
- Saved as `data/test_real_qlin.pt`
- Metrics: RMSE, NCC only (no segmentation ground truth available for real data)
- **Primary metric for viva: NCC on this test set**

### Baseline Comparison (Required)
All of the following must appear in final benchmark table:
- Wavelet db4 Soft Thresholding
- Wavelet BayesShrink
- Savitzky-Golay
- Pure CNN (Arch 1)
- 1D U-Net (Arch 2)  
- MWCNN (Arch 3)
- MR-TAE-FULL (Arch 4)
- All ablation variants
- All cross-combination variants

---

## PART 9: KNOWN OPEN QUESTIONS (Agent Must Resolve From Codebase)

The following cannot be answered without reading your actual code:

1. **What is the actual sampling rate used in `distance_signals.py`?** If it's not 1 MHz, the entire window sizing and frequency parameterization needs revision.
2. **Do Types A-F in your generator map to Corona/Internal/Surface specifically?** Show the mapping table.
3. **Is DWT using `pywt` (CPU) or `WaveTF` (GPU TF)?** If TF, migrate to `torch-wavelets` or `pywt` with PyTorch wrapper for unified framework.
4. **What is the actual window size used in training?** 2048 samples was stated in thesis — confirm it matches your generator output.
5. **Does the WGAN currently train on raw signal windows or on extracted noise segments?** If on full windows, that is likely causing the overfitting issue.

The agent should resolve all 5 questions BEFORE implementing any new code.
