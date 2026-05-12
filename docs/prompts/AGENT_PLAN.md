# MR-TAE Research Agent Plan
## Intelligent PD Denoising — Architecture Ablation, Expansion & Documentation
**For: Claude Opus 4.6 Agentic Coding System with Subagents**  
**Project:** Intelligent Partial Discharge Denoising with Adaptive Capabilities  
**Researcher:** Oo Jian Hong, University of Malaya, 2026  

---

## PRE-FLIGHT: Questions Answered vs. Code-Dependent

Before the agent begins coding, the following examiner Q&A has been pre-analyzed.

### ✅ ANSWERABLE FROM THESIS + DOMAIN KNOWLEDGE

**AE Physics (for viva prep):**  
PD events create a rapid pressure surge in transformer oil. This launches a longitudinal acoustic wave that travels through oil (~1400 m/s) and couples into the steel tank wall, where it converts partially to shear waves. At each interface (oil→steel, steel→air) there is a large acoustic impedance mismatch, causing significant energy reflection. The wave attenuates due to geometric spreading (1/r²), material damping (viscoelastic loss in oil), and mode conversion losses. Standard piezoelectric AE sensors detect in the 20 kHz–500 kHz band — far below the MHz/GHz range of UHF/HFCT electrical methods. This means AE is immune to electromagnetic interference (EMI) but susceptible to mechanical noise (magnetostriction hum at 100/120 Hz harmonics, cooling fans, rain impacts, loose hardware rattles).

**Sensors (Q1):** In field deployments, sensors are coupled to the tank exterior using a viscous couplant (vacuum grease/silicone gel) to maximize acoustic transmission. Magnetic hold-downs or adhesive waveguides are used for permanent installation. The Lin et al. (2022) experimental setup differs from this — it uses direct contact on an insulating structure in a lab, meaning real tank reverb, paint/rust attenuation layers, and weld-scattering are absent. This gap between lab and field is a legitimate examiner challenge.

**AE vs. Electrical Detection (Q2):**  
| Property | AE (Acoustic) | Electrical (HFCT/UHF) |
|---|---|---|
| Propagation | Sound ~1400 m/s (oil) | Electromagnetic ~3×10⁸ m/s |
| Attenuation | Physical damping, distance-dependent | Impedance mismatch, cable loss |
| EMI immunity | ✅ High | ❌ Susceptible |
| Sensitivity to internal voids | Lower (signal must travel to tank wall) | Higher (direct electrical coupling) |
| Standard frequency band | 20 kHz – 500 kHz | 3 MHz – 3 GHz |

**Nature of Simulated Data (Q3):**  
The thesis generates PD signals as mathematical decaying sinusoids (damped exponentials), which are a reasonable first-order approximation of real PD pulses. However, this model does NOT account for: (a) reverberation / multipath reflections from tank walls and internal structures, (b) dispersion — different frequency components of the AE wave arriving at different times because group velocity is frequency-dependent in steel plates, (c) mode conversion — P-waves converting to S-waves at interfaces. Real AE signals have complex oscillatory tails that last far longer than the idealized decaying sinusoid. This is a known and citable limitation that should be explicitly acknowledged in the viva.

**Examiner Curveball Pre-Answers:**
1. **Source Location Ambiguity:** The current model makes no explicit distance/location assumption — it classifies by PD type (Corona/Internal/Surface), not source location. The thesis uses copper beads of different sizes at uniform gaps to represent severity. An examiner may challenge whether the model can distinguish severity from discharge type — this is a valid gap; the segmentation head could theoretically be extended to a severity-regression head, but it has not been implemented.
2. **Mechanical vs. Electrical Noise:** The model implicitly learns this discrimination through training on PD pulses vs. AWGN/NBI/Impulse backgrounds. However, it has NOT been trained on true mechanical transients (rain, vibration, magnetostriction bursts) as negative examples. This is the biggest examiner vulnerability — the WGAN augmentation helps but doesn't specifically model mechanical noise profiles. Recommend: add Spectral Kurtosis (SK) as a pre-screening stage for future work.
3. **AE sensitivity vs. UHF/HFCT:** This is a legitimate sensitivity tradeoff. AE requires the pressure wave to survive the oil→steel→sensor path. Internal winding voids deep inside the transformer core may produce signals too attenuated to detect reliably via AE. The correct answer is: AE and electrical methods are complementary, not competing. AE excels at localization in oil-insulated equipment; UHF excels at sensitivity in GIS.
4. **WGAN Overfitting (addressed in detail in Phase 2 of this plan).**
5. **Dispersion robustness:** The current architecture has not been validated on signals with artificial dispersion modeling. This should be flagged as future work.

### ⚠️ REQUIRES CODE INSPECTION BEFORE ANSWERING

The agent must read these files before the following questions can be answered:
- `mr_tae_ultimate.py` — Is the BiGRU actually bidirectional? What are the GRU hidden dims?
- `train_extended.py` — What is the actual curricular phase split logic?
- `mr_tae_fusion/data/distance_signals.py` — What noise types are actually implemented vs. documented?
- `run_training.py` — Are checkpoints being saved? What's the actual epoch/lr schedule?
- Any existing model checkpoint files — Which models have actually been trained?

---

## PHASE 0: Repository Hygiene (Do This First)

### Task 0.1 — Create .gitignore

```
# Environments
venv/
.venv/
env/
__pycache__/
*.pyc
*.pyo
.Python

# Data (heavy, do not upload)
data/real/
data/mock/
data/wgan_generated/
*.mat
*.csv
datasets/

# Model checkpoints (large binary files)
checkpoints/
*.pth
*.pt
*.h5
*.keras
saved_models/

# Experiment outputs (regeneratable)
results/plots/
results/raw/
benchmark_results.json

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb  # optional — decide if notebooks should be tracked

# Logs
*.log
runs/  # TensorBoard logs
```

### Task 0.2 — Create requirements.txt

Extract from codebase imports. Expected dependencies include:
```
torch>=2.0.0
torchvision
torchaudio
numpy
scipy
pywavelets
scikit-learn
matplotlib
seaborn
pandas
tqdm
tensorboard
WaveTF  # if TF-based MWCNN is used
tensorflow>=2.12.0  # only if TF used alongside PyTorch
```

**Important:** If the project mixes TensorFlow (MWCNN WaveTF) and PyTorch (MR-TAE), document this clearly in README.md. Suggest migrating all to PyTorch using `pywt` for DWT.

### Task 0.3 — Audit and catalog all .py files

Before any training, the agent must produce a `FILE_INVENTORY.md` with:
- File name, purpose, model(s) defined, training status (untrained / partially trained / fully trained)
- Parameter count for each model
- Whether checkpoints exist

---

## PHASE 1: Architecture Ablation Studies

The goal is to isolate the contribution of each MR-TAE component. Based on the thesis (Section 6.2, Future Work), these ablations were planned but not yet executed.

### Ablation Naming Convention
```
MR-TAE-FULL      = All components (baseline — the proposed model)
MR-TAE-noBiGRU   = Remove BiGRU, keep Swin Transformer in bottleneck
MR-TAE-noSwin    = Remove Swin Transformer, keep BiGRU in bottleneck  
MR-TAE-noAttn    = Remove Attention Gates on skip connections (plain U-Net skips)
MR-TAE-noMTL     = Remove segmentation head, train denoising-only with Charbonnier loss
MR-TAE-noWavelet = Replace DWT/IDWT with standard MaxPool/TransposeConv (plain 1D-UNet with BiGRU-Swin)
```

### Task 1.1 — Implement each ablation variant as a class
Each variant should:
- Inherit from a base `MRTAEBase` class where possible
- Have a `MODEL_NAME` string attribute for logging
- Accept an `AblationConfig` dataclass to toggle components on/off cleanly
- NOT duplicate model code — use flags, not copy-paste

### Task 1.2 — Cross-combination variants (new architectures)
```
MWCNN-BiGRU      = MWCNN encoder/decoder + BiGRU bottleneck (no Transformer)
MWCNN-Swin       = MWCNN encoder/decoder + Swin only (no BiGRU)
UNet-BiGRU-Swin  = Standard MaxPool U-Net + full BiGRU-Swin bottleneck
UNet-BiGRU       = Standard MaxPool U-Net + BiGRU only
UNet-Attn        = Standard U-Net + Attention Gates only (Attention U-Net baseline)
```

This creates a 3×3 grid of feature importance:
```
Bottleneck\Encoder   | Plain MaxPool U-Net | MWCNN (DWT)
---------------------|---------------------|-------------
CNN only             | 1D-UNet (existing)  | MWCNN (existing)
BiGRU only           | UNet-BiGRU          | MWCNN-BiGRU
Swin only            | UNet-Swin           | MWCNN-Swin
BiGRU + Swin (full)  | UNet-BiGRU-Swin     | MR-TAE-FULL
```
Plus MTL and Attention Gate as orthogonal switches on top.

### Task 1.3 — Think Outside the Box: Additional Candidate Architectures

The agent should also implement and test the following novel ideas:
1. **MR-TAE + Mamba (SSM) bottleneck** — Replace BiGRU-Swin with a State Space Model (Mamba/S4). SSMs have linear complexity like Swin but model long-range temporal dependencies natively without attention windows. Hypothesis: may outperform BiGRU-Swin on longer sequences.
2. **Conformer bottleneck** (Conv + Self-Attention interleaved) — Used in speech processing, highly relevant for 1D temporal signals.
3. **WaveNet-style decoder** — Replace transposed conv decoder with dilated causal convolutions to better reconstruct high-frequency pulse edges.
4. **Dual-Encoder** — Parallel wavelet branch (MWCNN) + raw temporal branch (CNN), fused at bottleneck. Inspired by dual-stream architectures in speech enhancement.
5. **Denoising Diffusion Model (DDPM) for PD** — Exploratory. Frame denoising as conditional score matching. Hypothesis: very high NCC but slow inference — benchmark latency carefully.

---

## PHASE 2: WGAN Overfitting — Correct Strategy

### The Problem
The real dataset (Lin et al. 2022) has only ~800 vectors per SNR level (200 seconds × 4 SNR levels). Training a WGAN-GP directly to generate clean PD signals from this will overfit — the discriminator will memorize the 800 samples.

### Recommended Strategy
**Do NOT train WGAN to generate complete PD+noise signals from scratch.**

Instead, use WGAN in a targeted augmentation role:
1. **Mode 1 — Noise-Only WGAN**: Train WGAN on noise-only segments (extracted from blank gaps between PD pulses in the real data). The WGAN learns the real-world noise distribution (magnetostriction, mechanical, etc.). Use this to augment synthetic clean pulses → realistic noisy training samples.
2. **Mode 2 — Pulse Morphology WGAN**: Train a conditional WGAN conditioned on discharge type label (Corona/Internal/Surface) to generate diverse pulse shapes beyond the 4 copper-bead sizes tested. Prevents the denoiser from overfitting to specific pulse morphologies.
3. **Add Feature Matching Loss**: Instead of just Wasserstein loss, add feature matching on intermediate discriminator layers to prevent mode collapse.
4. **Spectral Normalization**: Apply to all discriminator layers (already part of WGAN-GP spec, but verify in code).
5. **Data holdout**: Keep 90% of real data strictly for testing (as planned in original benchmark_runner.py spec). Only augment the 10% training split.
6. **Early stopping on FID**: Monitor Fréchet Inception Distance (or its 1D signal equivalent — Maximum Mean Discrepancy) on a held-out real validation set. Stop WGAN training when MMD plateaus.

The agent should implement `wgan_noise_aug.py` and `wgan_pulse_aug.py` as separate modules.

---

## PHASE 3: Unified Training Pipeline

### Task 3.1 — Unified trainer `train_all_ablations.py`

Single script that:
- Accepts a `--model` flag from the model registry
- Uses the same data split, same random seed (42), same augmentation pipeline
- Saves: checkpoint, training curves, per-epoch metrics to `results/{model_name}/`
- Supports resuming from checkpoint
- Logs to TensorBoard

### Task 3.2 — Curricular Training (fix/verify)

Per the thesis (Section 3.5.2), training is in 3 phases:
- Phase 1: SNR [5, 15] dB — easy
- Phase 2: SNR [-5, 5] dB — medium  
- Phase 3: SNR [-20, -5] dB — hard

Verify this is correctly implemented. Common bugs to check:
- Are validation metrics computed on the **full SNR range** even during Phase 1 training? (They must be, for fair epoch-curve plotting)
- Is the scheduler reset between phases or continuous?
- Is the segmentation head activated from epoch 1 or only after Phase 1?

### Task 3.3 — Benchmarking `benchmark_runner.py`

Single evaluation script for ALL models on the SAME test set:
```
Metrics per model: SNR_out, SNR_improvement, MSE, RMSE, NCC, 
                   Classification Accuracy (Corona/Internal/Surface/Non-PD),
                   False Positive Rate,
                   Parameter Count (#M),
                   Training Time (minutes),
                   Inference Latency (ms/sample, batch=1 and batch=32)
```

Output: `benchmark_results.json`, `results_summary.csv`, comparison bar charts.

---

## PHASE 4: Documentation & Result Storage

### Task 4.1 — Recommended File Hierarchy

```
project_root/
├── README.md                          # Overview, setup instructions, results summary
├── requirements.txt                   # Pip dependencies
├── .gitignore                         # See Phase 0
│
├── models/                            # All model class definitions
│   ├── __init__.py
│   ├── base.py                        # MRTAEBase, AblationConfig
│   ├── pure_cnn.py                    # Architecture 1
│   ├── unet_1d.py                     # Architecture 2
│   ├── mwcnn.py                       # Architecture 3
│   ├── mr_tae.py                      # Architecture 4 (full proposed model)
│   ├── ablations/
│   │   ├── mr_tae_no_bigru.py
│   │   ├── mr_tae_no_swin.py
│   │   ├── mr_tae_no_attn.py
│   │   ├── mr_tae_no_mtl.py
│   │   └── mr_tae_no_wavelet.py
│   └── experimental/
│       ├── mwcnn_bigru.py
│       ├── unet_bigru_swin.py
│       └── (diffusion, mamba, etc.)
│
├── data/
│   ├── generators/
│   │   ├── distance_signals.py        # Synthetic PD signal generator
│   │   ├── noise_injection.py         # All noise types (AWGN, NBI, Impulse, Powerline)
│   │   └── wgan_augmentation/
│   │       ├── wgan_noise_aug.py
│   │       └── wgan_pulse_aug.py
│   ├── splitter.py                    # 10/90 real data split
│   └── dataset.py                     # PyTorch Dataset class
│
├── training/
│   ├── train_all_ablations.py         # Unified trainer
│   ├── curricular_trainer.py          # Phase management
│   ├── loss_functions.py              # Charbonnier, Focal, Homoscedastic
│   └── schedulers.py
│
├── evaluation/
│   ├── benchmark_runner.py            # Master evaluation script
│   ├── metrics.py                     # SNR, MSE, RMSE, NCC
│   └── classifier_eval.py             # Post-denoising classification test
│
├── traditional_baselines/
│   ├── wavelet_threshold.py
│   ├── bayesshrink.py
│   └── savitzky_golay.py
│
├── results/                           # gitignored for raw data, but structure is tracked
│   ├── .gitkeep
│   └── {model_name}/
│       ├── training_curves.png
│       ├── snr_vs_rmse.png
│       ├── snr_vs_ncc.png
│       └── metrics.json
│
├── docs/
│   ├── ARCHITECTURE_SUMMARY.md        # Full summary of all models
│   ├── EXAMINER_QA.md                 # Pre-prepared viva answers
│   ├── MODEL_COMPARISON.md            # Results table + analysis
│   └── FILE_INVENTORY.md              # What agent produces in Phase 0
│
└── scripts/
    ├── run_all_ablations.sh
    └── generate_report_figures.py
```

### Task 4.2 — ARCHITECTURE_SUMMARY.md

The agent must produce a markdown file documenting:
- Every model: architecture diagram (ASCII or mermaid), parameter count, input/output shape
- Training method used (MTL vs. single-task, curricular vs. standard)
- Test set used (synthetic only vs. synthetic + real)
- Final metric table (reproducing and extending Table 4.1 from thesis)
- Clear verdict: which model is most robust, which is most efficient, which is best for deployment

### Task 4.3 — Result images

For every trained model, produce and save:
1. `training_curves.png` — RMSE and NCC vs. epoch
2. `snr_sweep.png` — RMSE and NCC at SNR ∈ {-15, -10, -5, 0, 5} dB
3. `signal_comparison.png` — side-by-side: noisy | denoised | clean ground truth, at -15 dB and 0 dB
4. `benchmark_bar.png` — grouped bar chart of all models across all metrics

---

## PHASE 5: Final Robustness Verdict (Agent to Conclude)

After all training and evaluation, the agent must write `FINAL_VERDICT.md` answering:

1. **Most robust model overall** — highest SNR_improvement + NCC on real-world data
2. **Most efficient model** — best NCC/parameter ratio and inference latency
3. **Best ablation insight** — which single component contributes most to MR-TAE's performance?
4. **Recommended training method** — Curricular + MTL vs. standard single-task?
5. **Is the WGAN augmentation actually helping?** — Compare with and without WGAN data
6. **Should BiGRU be kept?** — Latency vs. accuracy tradeoff quantified
7. **Cross-application opportunity** — Can this denoiser be applied to audio/vibration signal denoising in other domains? What would need to change?

---

## EXECUTION ORDER

```
Phase 0 → Phase 1.1 (read code) → Phase 0 (gitignore + requirements) →
Phase 1.2 (ablation classes) → Phase 1.3 (cross-combos) →
Phase 2 (WGAN fix) → Phase 3.1 (unified trainer) → Phase 3.2 (verify curricular) →
Phase 3.3 (benchmark runner) → [TRAIN ALL] → Phase 4 (docs + figures) →
Phase 5 (verdict)
```

**Subagent allocation suggestion:**
- Subagent A: Code audit + gitignore + file hierarchy restructuring
- Subagent B: Ablation model implementations
- Subagent C: WGAN augmentation correction
- Subagent D: Unified training + benchmark pipeline
- Subagent E: Documentation generation + figure generation

---

## KEY CONSTRAINTS FOR AGENT

1. **Never duplicate model code** — use flags and inheritance
2. **All models trained on identical splits** — use `random.seed(42)` and a fixed DataLoader
3. **Real data is never touched for training** — 100% of Lin et al. 2022 data used for testing only (or at most 10% for augmentation as originally planned)
4. **Save all plots as PNG** at minimum 150 DPI
5. **Every file must have a module docstring** explaining its role
6. **requirements.txt must be tested** — run `pip install -r requirements.txt` in a clean venv and verify no conflicts
7. **The thesis noise types are AWGN + Powerline + NBI + Impulse** — not "acoustic reflection" as mentioned in the planning doc; the planning doc misidentified the noise model. The real signal is AE, but the noise added in training is synthetic electrical/electronic noise (AWGN, NBI, Impulse) per Chapter 3.3 of the thesis. Do not confuse acoustic physics noise with the mathematical noise injected during training.
