# AGENT PROMPT v2 — AE-PD Denoising Full Pipeline
## For: Claude Opus 4.6 + Subagents + Tools + Code Skills
## Hardware: RTX 4070 Laptop 12GB VRAM | Framework: PyTorch

---

## YOUR ROLE

You are a senior deep learning research engineer specializing in signal processing and acoustic emission diagnostics. You are implementing a complete, automated, reproducible research pipeline for a Final Year thesis on Partial Discharge denoising via Acoustic Emission signals.

**Read FULL_RESEARCH_PLAN.md first. It contains all domain physics, corrections, and specifications.**  
**Read AGENT_PLAN.md for architecture ablation specifications.**  
Both documents are your authoritative reference.

---

## MANDATORY FIRST ACTION — CODEBASE AUDIT (5 Questions)

Before ANY code generation, use your tools to read the existing codebase and answer these 5 questions in `docs/CODEBASE_AUDIT.md`:

1. **Sampling rate**: What is `fs` in `distance_signals.py`? Is it 1,000,000 Hz (1 MHz)?
2. **Frequency range**: What are the actual `fc` values used in each pulse type? Are they in kHz or MHz?
3. **Window size**: Confirm 2048 samples is the actual training window size
4. **Framework**: Is there mixed TF/PyTorch usage? List all `import tensorflow` and `import torch` occurrences
5. **Type mapping**: What is the `type_to_class` dictionary? Show the A–F/G type names and their integer class labels

**If the audit reveals MHz-range frequencies, flag this as CRITICAL CORRECTION REQUIRED before proceeding.**

After audit, use your web search tool to verify: "acoustic emission PD detection frequency range piezoelectric sensor" — confirm the 20–500 kHz AE band is correct per IEEE/IEC standards.

---

## STEP 1 — SIGNAL & NOISE CORRECTIONS

### 1a — Fix pulse_generator.py

Rewrite the pulse generator with these AE-accurate parameters. **Do not preserve old frequency values.**

```python
"""
pulse_generator.py — AE-accurate Partial Discharge pulse synthesis

PHYSICS BASIS:
- AE sensors detect mechanical pressure waves, NOT electrical signals
- Frequency band: 20 kHz – 500 kHz (NOT MHz — MHz is for UHF/HFCT electrical PD)
- Carrier frequency from Q.Lin waveform analysis: 50–200 kHz
- Decay constant from Q.Lin tail analysis: 200–1200 µs
- Transfer function modeling: 1–3 acoustic reflections from tank walls

SIGNAL TYPES RETAINED (AE-valid):
- surface_discharge: sparse impulses, high amplitude (maps to Q.Lin 1.8mm/2.0mm/2.5mm)  
- surface_streamer_dense: continuous low-amplitude activity (maps to Q.Lin 1.0mm)
- corona: repetitive low-amplitude bursts (synthetic only)
- internal: high-amplitude short-decay (synthetic only)

SIGNAL TYPES REMOVED OR CORRECTED:
- Type G (treeing): fc corrected to 80–200 kHz; flagged as synthetic-only, not in Q.Lin
- Any type with fc in MHz range: REMOVED
"""
```

Key implementation requirements:
- All time vectors must be **continuous** (not sample-index based): `t = np.arange(n_samples) / fs`
- Add `apply_acoustic_transfer_function(signal, fs)` to every generated pulse
- For the dense-streamer type (1.0mm equivalent): generate overlapping short pulses at high event rate, not single isolated pulses
- Segmentation mask: mark each pulse region with the correct class integer
- Include a `verify_frequency_content(signal, fs)` function that plots the PSD and asserts peak is in [20kHz, 500kHz]

### 1b — Fix noise_generator.py

Keep: WGN, Bernoulli-Gaussian impulsive, Powerline (50 Hz + harmonics), Narrowband.

Add: **Magnetostrictive hum** (100 Hz + 200 Hz) as a new noise type.

Remove: Any "acoustic reflection noise" or "AEPD background" noise — reflections belong in the signal transfer function, not as additive noise.

Fix: All narrowband frequencies must be in the sub-MHz range. No `fc = N * 1e6` in noise_generator.py.

### 1c — Add transfer_function.py

New file. Implements the acoustic multipath model:
```python
def generate_tank_transfer_function(
    fs: float,
    num_reflections: int = None,    # random 1-3 if None
    reflection_delays_us: tuple = (80, 400),
    reflection_attenuation: tuple = (0.15, 0.45)
) -> np.ndarray:
    """
    Returns impulse response h of length = 5ms * fs.
    Convolve with clean pulse to get physically realistic AE waveform.
    """
```

Apply this to EVERY generated pulse. The resulting signal should show:
- Sharp initial onset
- Oscillatory tail lasting 500–1500 µs
- Subtle secondary oscillation packet ~100–300 µs after main arrival

### 1d — Validate against Q.Lin visuals

After implementing corrections, generate 6 random synthetic pulses and create a comparison figure `docs/signal_validation.png` showing:
- Your synthetic pulse vs. a real Q.Lin pulse of the same approximate type
- The PSD of both (should both peak in 50–200 kHz range)
- **Do not proceed to model training until this visual match is satisfactory**

---

## STEP 2 — UNIFIED MODEL FRAMEWORK

### 2a — AblationConfig

```python
@dataclass
class AblationConfig:
    use_wavelet_pooling: bool = True    # DWT/IDWT vs MaxPool/TransposeConv
    use_bigru: bool = True              # BiGRU in bottleneck
    use_swin: bool = True               # Swin Transformer in bottleneck
    use_attention_gates: bool = True    # Attention Gates on skip connections
    use_mtl: bool = True                # Multi-task segmentation head
    loss_config: str = 'A'             # 'A', 'B', 'C', or 'D' — see PLAN
    model_id: str = 'MR-TAE-FULL'      # Used for MLflow run naming
```

### 2b — Model Registry

Define all models as entries in `config/model_registry.yaml`, not as hardcoded Python:
```yaml
models:
  MR-TAE-FULL:
    use_wavelet_pooling: true
    use_bigru: true
    use_swin: true
    use_attention_gates: true
    use_mtl: true
    
  MR-TAE-noBiGRU:
    use_wavelet_pooling: true
    use_bigru: false
    use_swin: true
    use_attention_gates: true
    use_mtl: true
    
  # ... all other ablations and cross-combinations from AGENT_PLAN.md
```

### 2c — Fix Activation Functions

In all encoder/decoder CNN blocks:
- Replace `ReLU` with `GELU` 
- The FINAL output convolution (reconstruction head) must use **no activation** (linear)
- Never apply ReLU after the final reconstruction layer — AE signals are bipolar

---

## STEP 3 — RTX 4070 TRAINING FRAMEWORK

### 3a — memory_manager.py

```python
class RTX4070Manager:
    """
    Prevents OOM and thermal throttling on RTX 4070 laptop.
    
    Rules:
    - Always use AMP (torch.cuda.amp) 
    - Always use gradient checkpointing for models >40M params
    - Monitor GPU temp every 30 seconds
    - Pause training if temp > 82°C, resume at 72°C
    - Auto-reduce batch size by 50% on OOM, retry
    - Log VRAM peak usage to MLflow after each epoch
    """
    
    def get_safe_batch_size(self, model_param_count_M: float) -> int:
        """Return safe batch size based on parameter count."""
        if model_param_count_M < 5:    return 128
        elif model_param_count_M < 20: return 64
        elif model_param_count_M < 50: return 32
        else:                          return 16  # grad checkpointing required too
    
    def check_thermal(self) -> bool:
        """Returns True if safe to continue, False if need to pause."""
        # Use pynvml or subprocess nvidia-smi to get temp
        
    def handle_oom(self, trainer, current_batch_size: int) -> int:
        """OOM recovery: halve batch size, clear cache, return new batch size."""
        torch.cuda.empty_cache()
        gc.collect()
        new_bs = max(4, current_batch_size // 2)
        print(f"OOM caught — reducing batch size: {current_batch_size} → {new_bs}")
        return new_bs
```

### 3b — Unified Trainer (trainer.py)

Requirements:
- Accepts any model from the registry by `model_id`
- Uses AMP for all models
- Uses gradient checkpointing if `params > 40M` or `use_gradient_ckpt=True` in config
- Curricular phase handling: `--phase 1|2|3|all`
- Saves checkpoint every 10 epochs + best checkpoint (tracked by val NCC)
- Saves snapshot figure every 10 epochs (3-panel: noisy/denoised/clean at -15dB, -5dB, +5dB)
- Logs ALL metrics to MLflow every epoch
- Handles `KeyboardInterrupt` gracefully: saves checkpoint before exiting
- OOM recovery: wraps training step in try/except, calls `memory_manager.handle_oom()`

### 3c — heat_monitor.py

Background thread watching GPU temperature:
```python
class HeatMonitor(threading.Thread):
    """
    Runs in background. Sets a threading.Event when temp is too high.
    Trainer checks this Event at each batch iteration.
    """
    WARN_C = 78
    PAUSE_C = 82
    RESUME_C = 72
    CHECK_INTERVAL_S = 30
```

---

## STEP 4 — MLFLOW ORCHESTRATOR

### orchestrator.py

This is the TOP-LEVEL script the researcher runs. It:
1. Reads `config/model_registry.yaml` — gets list of all models to train
2. For each model (in order of parameter count, smallest first):
   a. Creates MLflow experiment: `AE-PD-Denoising/{model_id}`
   b. Runs Phase 1 training → saves checkpoint
   c. Runs Phase 2 training (resumes from Phase 1) → saves checkpoint
   d. Runs Phase 3 training (resumes from Phase 2) → saves best checkpoint
   e. Runs benchmark evaluation on both test sets
   f. Logs all metrics, figures, and checkpoint paths to MLflow
3. After all models trained: runs `generate_comparison_report()`

### MLflow Logging Schema
```python
# Every epoch:
mlflow.log_metrics({
    "train/loss": ..., "train/rmse": ..., "train/ncc": ...,
    "val/loss": ..., "val/rmse": ..., "val/ncc": ...,
    "gpu/vram_used_gb": ..., "gpu/temp_c": ...,
    "train/phase": ...,   # 1, 2, or 3
})

# Every 10 epochs:
mlflow.log_artifact(f"snapshot_epoch{N}.png")

# At end:
mlflow.log_metrics({
    "test_synthetic/snr_improvement": ...,
    "test_synthetic/ncc": ...,
    "test_synthetic/rmse": ...,
    "test_real/ncc": ...,     # PRIMARY METRIC
    "test_real/rmse": ...,
    "inference_latency_ms": ...,
    "param_count_M": ...,
})
mlflow.log_artifact("benchmark_results.json")
```

---

## STEP 5 — OPTUNA HYPERPARAMETER OPTIMIZATION

### hpo/optuna_study.py

Run HPO **on MR-TAE-FULL only** (most expensive model, most hyperparameters). Use 30 trials.

**Search space:**
```python
search_space = {
    # Architecture
    "swin_num_heads":    trial.suggest_categorical([4, 8, 16]),
    "swin_window_size":  trial.suggest_categorical([16, 32, 64]),
    "bigru_hidden_dim":  trial.suggest_categorical([128, 256, 512]),
    "encoder_base_ch":   trial.suggest_categorical([16, 32, 64]),
    
    # Training
    "lr":                trial.suggest_float(1e-5, 3e-3, log=True),
    "weight_decay":      trial.suggest_float(1e-6, 1e-3, log=True),
    "dropout":           trial.suggest_float(0.0, 0.3),
    
    # Loss
    "loss_config":       trial.suggest_categorical(["A", "B", "C", "D"]),
    "charb_epsilon":     trial.suggest_float(1e-4, 1e-2, log=True),
    
    # Curriculum
    "phase1_epochs":     trial.suggest_int(20, 50),
    "phase2_epochs":     trial.suggest_int(20, 50),
    "phase3_epochs":     trial.suggest_int(20, 50),
}
```

**Objective function**: maximize `test_real/ncc` (NCC on Q.Lin real data — the primary viva metric).

**Pruning**: Use `optuna.integration.MLflowCallback` + `optuna.pruners.HyperbandPruner` to kill bad trials early.

**RTX 4070 constraint**: Only run 1 trial at a time (`n_jobs=1`). Each trial = Phase 1 only (30 epochs) for speed, then retrain best config for full curriculum.

---

## STEP 6 — EXPLAINABILITY (Post-Training)

Run on the BEST model only (highest test_real/ncc from MLflow).

### shap_analysis.py

Use SHAP's `DeepExplainer` on a batch of 100 test windows:
```python
# Explain which time steps contributed most to the denoising decision
explainer = shap.DeepExplainer(best_model.encoder, background_batch)
shap_values = explainer.shap_values(test_batch)

# Save:
# 1. shap_summary.png — SHAP value distribution across time steps
# 2. shap_pulse_example.png — overlay SHAP values on an individual noisy signal
```

### lime_analysis.py

Use LIME to explain individual predictions:
- Segment the signal into 32-sample "super-pixels"
- Perturb segments and observe NCC change
- Identify which segments the model relies on most for pulse reconstruction

Save: `lime_window_examples.png` — 5 examples showing which input regions matter most.

---

## STEP 7 — BENCHMARK RUNNER

### benchmark_runner.py

Called automatically at end of each model's training. Evaluates on BOTH fixed test sets.

```python
def run_full_benchmark(model_id: str, checkpoint_path: str):
    """
    Evaluates model on:
    1. Synthetic AE test set (data/test_synthetic_ae.pt)
    2. Real Q.Lin test set (data/test_real_qlin.pt)
    
    Returns dict of all metrics.
    Saves comparison figures.
    Updates docs/RESULTS_COMPARISON.md
    """
    results = {
        "synthetic": {
            "snr_improvement_mean": ...,
            "snr_improvement_std": ...,
            "rmse_mean": ...,
            "rmse_std": ...,
            "ncc_mean": ...,       # NCC = Normalized Cross-Correlation
            "ncc_std": ...,
            "dice_mean": ...,      # Segmentation (synthetic only)
            "false_positive_rate": ...,
        },
        "real_qlin": {
            "rmse_mean": ...,
            "rmse_std": ...,
            "ncc_mean": ...,       # PRIMARY METRIC FOR VIVA
            "ncc_std": ...,
            # No segmentation metrics — real data has no temporal masks
        },
        "efficiency": {
            "param_count_M": ...,
            "inference_latency_ms_b1": ...,
            "inference_latency_ms_b32": ...,
            "training_time_min": ...,
        }
    }
```

### Generated Figures (auto-saved for every model)
1. `snr_sweep_{model_id}.png` — NCC and RMSE vs. SNR at [-15,-10,-5,0,5] dB
2. `signal_samples_{model_id}.png` — 3×2 grid: 3 SNR levels × (noisy/denoised/clean)
3. After all models: `comparison_bar_all_models.png`
4. After all models: `ablation_heatmap.png` — NCC matrix across bottleneck×encoder grid
5. After all models: `efficiency_scatter.png` — latency vs. NCC, bubble=params

---

## STEP 8 — AUTOMATED RETRAINING (Noise2Void)

### auto_retrain.py

```python
class AutoRetrainer:
    """
    Monitors model drift and triggers self-supervised fine-tuning.
    
    Drift detection:
    - Maintain a rolling NCC on a fixed 500-window "canary" real data set
    - If rolling NCC drops >0.05 from baseline: trigger retraining
    
    Self-supervised method: Noise2Void
    - Works on NEW UNLABELED real data (placed in data/incoming/)  
    - No clean ground truth needed
    - Randomly mask 10% of samples per window
    - Train to predict masked values from context
    - Valid because: PD signal is locally correlated; noise is i.i.d.
    - Loss: MSE on masked positions only
    
    Safety gate:
    - New model must achieve NCC >= current_baseline - 0.01 on test_real set
    - Otherwise reject new weights and keep old model
    - Log all retraining events to MLflow
    """
```

---

## STEP 9 — DOCUMENTATION AUTO-GENERATION

After all training complete, generate:

### docs/RESULTS_COMPARISON.md (auto-updated by benchmark_runner)
Full table format:
```
| Model | Params(M) | NCC-Synth | NCC-Real | RMSE-Real | FPR | Latency(ms) |
|---|---|---|---|---|---|---|
| Wavelet db4 | — | 0.135 | 0.345 | 0.143 | — | ~0.1 |
| BayesShrink | — | 0.220 | 0.375 | 0.136 | — | ~0.2 |
...
```

### docs/SIGNAL_PHYSICS.md
Document the corrected AE signal model with:
- Frequency justification (with citations)
- Transfer function parameter derivation from Q.Lin pulse images
- Why MHz-range was wrong and what was corrected

### docs/EXAMINER_QA.md (from AGENT_PLAN.md spec — keep)

---

## STEP 10 — WALKTHROUGH GUIDE FOR RESEARCHER

Create `WALKTHROUGH.md` — a step-by-step supervision guide:

```markdown
# Researcher Walkthrough

## First Run (~15 minutes setup)
1. pip install -r requirements.txt
2. Place Q.Lin .mat files in data/raw/qlin/
3. python data/qlin_loader.py --verify  # check signal plots
4. python data/generators/pulse_generator.py --verify  # check synthetic signal vs real
5. Review docs/signal_validation.png  # MUST look similar to Q.Lin pulses

## Training All Models (~24-72 hours total)
python pipeline/mlflow_orchestrator.py --run-all
# Progress visible at: mlflow ui (localhost:5000)
# GPU temp monitored automatically — safe to leave overnight

## Monitoring
- Open MLflow UI: mlflow ui --port 5000
- Check heat_monitor logs: tail -f logs/heat_monitor.log
- Snapshots auto-saved every 10 epochs to results/{model_id}/plots/

## After Training
python evaluation/benchmark_runner.py --compare-all
# Generates comparison figures and RESULTS_COMPARISON.md

## HPO (Optional, ~8-12 hours additional)
python hpo/optuna_study.py --n-trials 30
# Best hyperparameters saved to config/best_hparams.yaml
# Retrain best model: python pipeline/mlflow_orchestrator.py --model MR-TAE-FULL --hparams config/best_hparams.yaml

## Explainability (on best model)
python evaluation/explainability/shap_analysis.py --model-id MR-TAE-FULL
```

---

## CRITICAL CONSTRAINTS (READ BEFORE WRITING ANY CODE)

1. **AE frequency range: 20 kHz – 500 kHz. Any fc value above 500,000 Hz is WRONG.**
2. **Real Q.Lin test data is never used for training.** It is 100% held out. Loading it for training is a critical error.
3. **Output layer has NO activation function.** AE signals are bipolar — ReLU on output produces physically wrong results.
4. **All models trained and evaluated on identical test sets** (fixed seeds, saved as .pt files before any training begins).
5. **OOM must be handled gracefully.** Wrap all `trainer.train_step()` calls in try/except OOM, call `memory_manager.handle_oom()`, never crash.
6. **MLflow must be running before training starts.** Check `mlflow.get_tracking_uri()` at startup.
7. **Optuna pruning must be enabled.** 30 trials on an RTX 4070 without pruning = ~30 hours. With Hyperband pruning = ~8 hours.
8. **Gradient checkpointing is NOT optional for MR-TAE-FULL on 12GB VRAM.** Enable it unconditionally for models >40M params.
9. **Q.Lin class labels are bead sizes (1.0mm, 1.8mm, 2.0mm, 2.5mm) = surface discharge SEVERITY, not PD type.** Do not use these for segmentation head training. Real data is denoising-quality evaluation only.
10. **The segmentation head is validated on synthetic data only.** NCC on real data is the primary deployability metric.

---

## TECHNOLOGY STACK

```
Deep Learning:     PyTorch 2.x (unified — no TensorFlow unless unavoidable)
DWT:               pywt + custom PyTorch autograd wrapper (or torch-wavelets)
Experiment Track:  MLflow 2.x (local file store)
HPO:               Optuna 3.x with MLflowCallback
Explainability:    SHAP (DeepExplainer) + LIME
GPU Monitoring:    pynvml (NVML bindings for temperature/VRAM)
Signal Processing: scipy, numpy, pywt
Visualization:     matplotlib, seaborn
Config:            PyYAML + dataclasses
```

---

## SUBAGENT ALLOCATION (suggested)

- **Subagent A** — Codebase audit + signal/noise corrections + signal_validation.png
- **Subagent B** — Model framework (AblationConfig, components, all model variants)
- **Subagent C** — Training framework (trainer, memory_manager, heat_monitor, curricular scheduler)
- **Subagent D** — MLflow orchestrator + Optuna HPO integration
- **Subagent E** — Benchmark runner + SHAP/LIME + auto_retrain + documentation

**Subagent B must complete before D. Subagent C must complete before D. A must complete before B and C.**

---

*Reference documents: FULL_RESEARCH_PLAN.md (physics, corrections, pipeline) | AGENT_PLAN.md (architecture ablations)*  
*Thesis: "Intelligent Partial Discharge Denoising with Adaptive Capabilities", Oo Jian Hong, University of Malaya 2026*
