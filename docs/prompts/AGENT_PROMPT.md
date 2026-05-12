# AGENT PROMPT — MR-TAE Research Expansion
## For: Claude Opus 4.6 with Subagents, Tools, and Code Skills

---

## YOUR MISSION

You are a senior AI research engineer and signal processing expert. You are working on a Final Year Thesis project: **"Intelligent Partial Discharge Denoising with Adaptive Capabilities"** by Oo Jian Hong, University of Malaya 2026.

The researcher has an existing codebase with 4 deep learning architectures for denoising Acoustic Emission (AE) signals from Partial Discharge (PD) events in high-voltage equipment. The research paper is complete, but the code needs to be:
1. Cleaned up and properly organized for reproducibility
2. Extended with ablation studies the paper planned but did not run
3. Extended with new candidate architectures worth testing
4. Benchmarked rigorously so the researcher can defend architectural choices in their viva

**Read AGENT_PLAN.md first. It is your primary reference for all tasks.**

---

## STEP 1 — MANDATORY FIRST ACTION: Code Audit

Before writing a single line of new code, use your tools to:

1. List all `.py` files in the project
2. Read every model-definition file and produce `docs/FILE_INVENTORY.md` with:
   - File name and purpose
   - Every model class defined, with parameter count
   - Training status (has checkpoint? partially trained? untrained?)
   - Whether it matches the thesis architecture or is a variant

Pay special attention to:
- Is the BiGRU in `mr_tae_ultimate.py` actually bidirectional? What are hidden dims?
- Does `train_extended.py` implement the 3-phase curricular training correctly?
- Does `distance_signals.py` have AWGN + NBI + Impulse + Powerline all implemented?
- Are there any models in the codebase NOT described in the thesis? Document them.

---

## STEP 2 — Git Hygiene

Create `.gitignore` and `requirements.txt` per the AGENT_PLAN.md Phase 0 spec.

**Requirements.txt rules:**
- Pin major versions (e.g., `torch>=2.0.0,<3.0.0`)
- Separate PyTorch and TensorFlow if both are used
- Add a comment block at the top explaining the dual-framework situation if it exists
- Test that it installs cleanly

---

## STEP 3 — Ablation Study Implementation

Implement the following model variants as described in AGENT_PLAN.md Phase 1:

**Ablations of MR-TAE (remove one component):**
- `MR-TAE-noBiGRU` — bottleneck has only Swin Transformer (no BiGRU)
- `MR-TAE-noSwin` — bottleneck has only BiGRU (no Swin Transformer)
- `MR-TAE-noAttn` — attention gates on skip connections replaced with plain concatenation
- `MR-TAE-noMTL` — denoising only, no segmentation head, Charbonnier loss only
- `MR-TAE-noWavelet` — replace DWT downsampling with MaxPool, IDWT upsampling with TransposeConv

**Cross-combination new architectures:**
- `MWCNN-BiGRU` — MWCNN encoder/decoder + BiGRU-only bottleneck
- `MWCNN-Swin` — MWCNN encoder/decoder + Swin-only bottleneck
- `UNet-BiGRU-Swin` — Standard MaxPool 1D U-Net + full BiGRU-Swin bottleneck
- `UNet-BiGRU` — Standard MaxPool 1D U-Net + BiGRU only
- `UNet-Attn` — Standard U-Net + attention gates (Attention U-Net)

**Implementation rules:**
- All variants must share a `BaseDenoiser` parent class with common forward-pass boilerplate
- Use an `AblationConfig` dataclass with boolean flags to toggle components
- No copy-paste of model blocks — shared components live in `models/components/`
- Each variant must have a `get_parameter_count()` method and a `MODEL_ID` string

**Experimental (implement if time allows — flag as experimental):**
1. Conformer bottleneck (Conv + Multi-Head Self-Attention interleaved)
2. Mamba/S4 SSM bottleneck as BiGRU-Swin replacement

---

## STEP 4 — WGAN Augmentation Fix

The current WGAN likely overfits because the real dataset (Lin et al. 2022) has only ~800 vectors.

Implement two separate WGAN modules per AGENT_PLAN.md Phase 2:
1. `data/wgan_augmentation/wgan_noise_aug.py` — learns the real-world NOISE distribution from blank gap segments between PD pulses
2. `data/wgan_augmentation/wgan_pulse_aug.py` — conditional WGAN on PD type label to augment pulse morphology diversity

Both must include:
- Gradient penalty (already part of WGAN-GP)
- Spectral normalization on discriminator
- Feature matching loss
- MMD-based early stopping criterion
- The training split constraint: only the 10% real-data training subset is used for WGAN fitting

---

## STEP 5 — Unified Training Pipeline

Create `training/train_all_ablations.py` with:
- `--model` flag accepting any MODEL_ID from the registry
- `--phases` flag for [1,2,3] curricular phases (default: all 3)
- Checkpoint saving every 10 epochs + best model
- TensorBoard logging
- Resume from checkpoint support
- Identical data pipeline (same seed, same split) for ALL models

Verify the curricular training logic. Common bugs to fix if found:
- Validation must always use the full SNR range regardless of training phase
- Scheduler should either reset or use a phase-aware warmup at phase boundaries
- The segmentation head loss weight should ramp up — not be constant from epoch 1

---

## STEP 6 — Benchmark Runner

Create `evaluation/benchmark_runner.py` that evaluates ALL trained models on the same test set and outputs:

```json
{
  "model_name": {
    "snr_improvement_mean": float,
    "snr_improvement_std": float,
    "mse_mean": float,
    "rmse_mean": float,
    "ncc_mean": float,
    "classification_accuracy": {
      "corona": float,
      "internal": float,
      "surface": float,
      "false_positive_rate": float
    },
    "parameter_count_M": float,
    "training_time_minutes": float,
    "inference_latency_ms_batch1": float,
    "inference_latency_ms_batch32": float
  }
}
```

Also generate:
1. `results/benchmark_bar.png` — grouped bar chart (SNR_improvement, NCC, FPR, latency)
2. `results/ablation_heatmap.png` — heatmap of NCC across the 3×3 encoder/bottleneck grid
3. `results/snr_sweep.png` — line chart of NCC vs. input SNR for all models at [-15, -10, -5, 0, 5] dB
4. `results/efficiency_scatter.png` — scatter: inference latency (x) vs. NCC (y), bubble size = parameter count

---

## STEP 7 — Documentation

After all training and benchmarking, produce:

**`docs/ARCHITECTURE_SUMMARY.md`** — For every model:
- ASCII architecture diagram or Mermaid flowchart
- Parameter count
- What makes it different from the others
- When to use it

**`docs/EXAMINER_QA.md`** — Pre-written viva answers for:
- AE Physics (propagation, attenuation, impedance mismatch)
- AE sensor installation and field acquisition
- AE vs. electrical detection comparison
- Nature and limitations of the simulated data
- WGAN overfitting concern and the mitigation strategy implemented
- Source location ambiguity (how the model handles it or doesn't)
- Mechanical vs. PD noise discrimination

**`docs/FINAL_VERDICT.md`** — Your empirical verdict:
- Most robust model and why
- Most efficient model and why  
- Which single ablation component matters most
- Whether WGAN augmentation demonstrably helps (quantified)
- Whether BiGRU or Swin contributes more (from ablation data)
- Recommendation for the researcher's future work

**`docs/MODEL_COMPARISON.md`** — Extended version of thesis Table 4.1 covering all trained models

---

## STEP 8 — File Hierarchy Restructure

Restructure the repo to match the hierarchy in AGENT_PLAN.md Phase 4.1. When moving files:
- Update all import paths
- Verify the codebase runs after restructuring
- Produce `docs/FILE_INVENTORY.md` in FINAL state (post-restructure)

---

## CRITICAL CONSTRAINTS

1. **Real data (Lin et al. 2022) is held out** — never used for training, only testing. If any current code violates this, fix it.
2. **All models use the same random seed (42)** and identical train/test split
3. **The noise types added during training are MATHEMATICAL noise** (AWGN, NBI, Impulse, Powerline per Chapter 3.3 of thesis), NOT acoustic physics noise. Do not confuse these.
4. **The signal type IS acoustic** — the raw signals are AE signals from piezoelectric sensors. The noise is added synthetically.
5. **Do not duplicate model code** — shared components belong in `models/components/`
6. **Save every result figure** to `results/{model_name}/` before moving on
7. **Document every architectural decision** — if you make a design choice the thesis didn't specify (e.g., number of Mamba layers), explain it in a comment
8. **Flag anything uncertain** — if a component's behavior doesn't match the thesis description, write a `# NOTE:` comment explaining the discrepancy

---

## DOMAIN CONTEXT FOR THE AGENT

- **Signal**: 1D time-series, length 2048, represents ~1 second of AE signal from a piezoelectric sensor on a high-voltage transformer/switchgear tank
- **PD types**: Corona (repetitive, low amplitude), Internal (high amplitude, short), Surface (irregular, broad)
- **Noise**: AWGN (SNR -20 to +10 dB), NBI (narrowband from power electronics), Stochastic Impulse, Powerline harmonics (50/60 Hz)
- **Real data**: Lin et al. 2022 — copper bead defect models (1.0, 1.8, 2.0, 2.5 mm), 200s per class, segmented into 1s windows = 800 vectors per class per SNR level
- **Primary metrics**: SNR improvement (dB), NCC (shape fidelity), False Positive Rate (background misclassified as PD)
- **MTL**: Joint Charbonnier (denoising) + Focal Loss (segmentation) with homoscedastic uncertainty weighting

---

## DELIVERABLES CHECKLIST

By the end of this task, the following must exist:

- [ ] `.gitignore` covering envs, data, checkpoints, logs
- [ ] `requirements.txt` tested and clean
- [ ] `docs/FILE_INVENTORY.md` — audit of original codebase
- [ ] All ablation model classes implemented and unit-tested
- [ ] All cross-combination model classes implemented
- [ ] WGAN noise augmentation module with anti-overfitting measures
- [ ] Unified training script with curricular phases
- [ ] Benchmark runner producing JSON + 4 comparative figures
- [ ] `docs/ARCHITECTURE_SUMMARY.md`
- [ ] `docs/EXAMINER_QA.md`
- [ ] `docs/FINAL_VERDICT.md`
- [ ] `docs/MODEL_COMPARISON.md`
- [ ] Restructured file hierarchy per AGENT_PLAN.md
- [ ] Updated `FILE_INVENTORY.md` reflecting final state
- [ ] `README.md` with setup instructions and results summary

---

*Reference: AGENT_PLAN.md for full specification of each phase.*
*Thesis: "Intelligent Partial Discharge Denoising with Adaptive Capabilities", Oo Jian Hong, University of Malaya, 2026.*
