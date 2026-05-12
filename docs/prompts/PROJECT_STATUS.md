# Project Status — Intelligent PD Denoising
## AE Signal Denoising with MR-TAE | University of Malaya 2026

---

## ✅ COMPLETED

### Infrastructure & Environment
- [x] Virtual environment (`.venv`) with PyTorch 2.9.0 + CUDA 13.0 (RTX 4070 confirmed)
- [x] All 18 core dependencies verified importing correctly
- [x] `.gitignore` covering envs, data, checkpoints, logs, heavy files
- [x] `config/model_registry.yaml` — registry-driven model definition
- [x] `config/signal_config.yaml` — AE-accurate signal parameters

### Signal Physics — CRITICAL CORRECTIONS APPLIED
- [x] Sampling rate fixed: `fs = 1 MHz` (was `1 GHz` — a 1000× error)
- [x] Signal length fixed: `2048` samples = 2.048 ms per window
- [x] Pulse frequencies corrected to AE band: `20–500 kHz` (was MHz — wrong physics)
- [x] Decay constants corrected: `τ = 200–1200 µs` (was nanoseconds — wrong physics)
- [x] `transfer_function.py` — tank wall acoustic multipath model (1–3 reflections)
- [x] `pulse_generators.py` — AE-accurate DOP model, continuous time vector, applies transfer function
- [x] `noise_generators.py` — WGN, BG-Impulsive, Powerline, NBI, + new Magnetostrictive (100/200 Hz)
- [x] `verify_frequency_content()` — PSD assertion in [20kHz, 500kHz]

### Model Architecture
- [x] `AblationConfig` dataclass — flags for wavelet, BiGRU, Swin, attention gates, MTL
- [x] Shared components in `models/components/` (MWCNN blocks, attention gate, BiGRU, Swin 1D, seg head)
- [x] All model variants loadable from registry (6 variants confirmed)
- [x] GELU activation replacing ReLU in CNN encoder/decoder blocks
- [x] Linear output layer (no activation) — correct for bipolar AE signals
- [x] MR-TAE-noSwin confirmed building at 3.05M params

### Training Pipeline
- [x] `training/trainer.py` — unified trainer with AMP, gradient checkpointing, curricular phases
- [x] `training/memory_manager.py` — RTX 4070 VRAM budgets, OOM auto-recovery (halve batch + retry)
- [x] `training/heat_monitor.py` — background thread, pause >82°C, resume <72°C
- [x] `training/loss_functions.py` — Charbonnier, Focal, Dice, Homoscedastic MTL weighting
- [x] `training/curricular_scheduler.py` — Phase 1/2/3 SNR ranges, scheduler reset between phases
- [x] Checkpoint saving every 10 epochs + best-by-val-NCC
- [x] Snapshot figures (noisy/denoised/clean 3-panel) every 10 epochs
- [x] `KeyboardInterrupt` handled gracefully with checkpoint save

### Orchestration
- [x] `pipeline/mlflow_orchestrator.py` — trains all 6 models sequentially, smallest-first
- [x] MLflow logging schema: per-epoch metrics, VRAM, GPU temp, phase, artifacts
- [x] Training LIVE: `MR-TAE-noSwin` epoch 1 Phase 1 confirmed running
- [x] Batch size 128, AMP enabled, GPU at 39°C

---

## 🔴 INCOMPLETE — MUST DO BEFORE THESIS DEFENCE

### Signal Validation (Blocking)
- [ ] `docs/signal_validation.png` — synthetic vs Q.Lin real pulse overlay (needs scipy .mat loader)
- [ ] `data/qlin_loader.py` — load and preprocess Lin et al. `.mat` files
- [ ] Visual confirmation that synthetic pulse PSD peaks in 50–200 kHz (matching Q.Lin images)

### HPO (High Priority)
- [ ] `hpo/optuna_study.py` — Optuna with Hyperband pruning, MR-TAE-FULL only, 30 trials
- [ ] `config/search_space.yaml` — HPO bounds (lr, wd, Swin heads, BiGRU hidden dim, loss config)
- [ ] Best hyperparameter retraining after HPO completes
- [ ] `config/best_hparams.yaml` — saved best config

### Benchmarking (Critical for viva)
- [ ] `evaluation/benchmark_runner.py` — full evaluation on BOTH fixed test sets
- [ ] Fixed test sets saved as `.pt` before any training: `data/test_synthetic_ae.pt`, `data/test_real_qlin.pt`
- [ ] `results/comparison_bar_all_models.png`
- [ ] `results/ablation_heatmap.png` — NCC across bottleneck×encoder grid
- [ ] `results/efficiency_scatter.png` — latency vs NCC, bubble=params
- [ ] `results/snr_sweep_all.png` — NCC vs SNR at [-15,-10,-5,0,5] dB for all models

### Explainability
- [ ] `evaluation/explainability/shap_analysis.py` — SHAP DeepExplainer on best model
- [ ] `evaluation/explainability/lime_analysis.py` — LIME on signal window segments
- [ ] `results/shap_summary.png`, `results/lime_examples.png`

### WGAN Augmentation (Planned, Not Built)
- [ ] `data/wgan/wgan_noise.py` — learns real noise distribution from Q.Lin blank gaps
- [ ] `data/wgan/wgan_pulse.py` — conditional WGAN on PD type for morphology diversity
- [ ] Validate WGAN: MMD-based early stopping, feature matching loss, spectral normalization

### Automated Retraining
- [ ] `pipeline/auto_retrain.py` — Noise2Void self-supervised fine-tuning on unlabelled real data
- [ ] Drift detection: rolling NCC canary set, trigger on >0.05 drop

### Documentation (Required for thesis)
- [ ] `docs/ARCHITECTURE_SUMMARY.md` — all models with ASCII diagrams + param counts
- [ ] `docs/EXAMINER_QA.md` — pre-written viva answers (AE physics, sensor installation, etc.)
- [ ] `docs/RESULTS_COMPARISON.md` — extended Table 4.1 covering all trained models
- [ ] `docs/FINAL_VERDICT.md` — empirical verdict: most robust, most efficient, best ablation

### MLOps Platform & Frontend
- [ ] Local research dashboard (Flask + React) — live training monitor, file browser, results viewer
- [ ] Docker containerisation — reproducible environment for any machine
- [ ] Kubernetes manifest — scalable multi-GPU future deployment
- [ ] W&B / Neptune.ai integration as MLflow supplement for richer experiment UI

---

## 🟡 IN PROGRESS

| Item | Status |
|------|--------|
| Training MR-TAE-noSwin | Running (epoch 1, PID 21528) |
| MLflow logging | Active (localhost:5000) |
| remaining 5 model variants | Queued in orchestrator |

---

## 📋 PRIORITY ORDER FOR NEXT SESSION

1. Build and freeze both fixed test sets BEFORE training finishes (urgent — must exist before benchmark)
2. `data/qlin_loader.py` + `docs/signal_validation.png`
3. `evaluation/benchmark_runner.py`
4. Local dashboard + Docker
5. `hpo/optuna_study.py` (run after all 6 models trained)
6. SHAP/LIME (run on best model from HPO)
7. WGAN augmentation
8. Auto-retrain pipeline
9. Final documentation generation

---

## 🗂 FILES TO CLEAN UP (Your Decision)

| File/Folder | Size | Action |
|---|---|---|
| `Try Model/myenv/` | ~6 GB | Delete — `.venv` replaces it |
| `mr_tae_ultimate.py` (root) | small | Archive or delete — superseded |
| `compare_methods.py`, `train_extended.py`, `run_training.py` | small | Archive |
| Root `.m` MATLAB files (7 files) | small | Keep for reference |
| Root `.png` files | small | Delete — will be regenerated |
