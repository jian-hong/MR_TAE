# MLOps Forward Plan — AE-PD Denoising
## From Training to Production-Ready Research System

---

## PHASE A — COMPLETE CURRENT TRAINING RUN (This Week)

### A1 — Fix Test Sets IMMEDIATELY (Do Before Training Finishes)

Both test sets must be frozen before any model finishes training. If you wait until after, there's a risk of data leakage.

```bash
# Run now while training is ongoing in background
python data/build_test_sets.py --seed 42 \
    --synthetic-out data/test_synthetic_ae.pt \
    --real-out data/test_real_qlin.pt
```

`data/build_test_sets.py` must:
- Generate 10,000 synthetic AE windows (2048 samples, SNR ∈ {-15,-10,-5,0,5} dB, 2000 per level)
- Load Q.Lin `.mat` files, split 90% into test (never train), save as `.pt`
- Print SHA256 hash of both files — log these hashes in your thesis for reproducibility

### A2 — Monitor Training

```bash
# Terminal 1: MLflow UI
cd project_root && mlflow ui --port 5000

# Terminal 2: Live tail of training log
tail -f logs/training.log

# Terminal 3: GPU monitor
watch -n 10 nvidia-smi
```

Expected timeline (RTX 4070, batch=128, AMP):
- MR-TAE-noSwin (3M params): ~45 min per 100 epochs
- MR-TAE-noAttn (~20M): ~90 min
- MR-TAE-noBiGRU (~40M): ~2.5 hrs
- MR-TAE-noMTL (~65M): ~3.5 hrs
- MR-TAE-noWavelet (~35M): ~2 hrs
- MR-TAE-FULL (~65M): ~3.5 hrs
**Total: ~14–18 hours continuous**

---

## PHASE B — BENCHMARKING & ANALYSIS (After Training)

### B1 — Run Full Benchmark

```bash
python evaluation/benchmark_runner.py --compare-all \
    --test-synthetic data/test_synthetic_ae.pt \
    --test-real data/test_real_qlin.pt \
    --output-dir results/benchmark/
```

Outputs:
- `results/benchmark/comparison_table.csv` — all models × all metrics
- `results/benchmark/comparison_bar.png`
- `results/benchmark/ablation_heatmap.png` — NCC matrix
- `results/benchmark/efficiency_scatter.png`
- `results/benchmark/snr_sweep.png`
- `docs/RESULTS_COMPARISON.md` — auto-updated

### B2 — HPO on Best Performing Model

After benchmark, identify best model (highest NCC on real Q.Lin data):

```bash
python hpo/optuna_study.py \
    --model MR-TAE-FULL \
    --n-trials 30 \
    --objective test_real_ncc \
    --output config/best_hparams.yaml
```

HPO strategy:
- Hyperband pruner: kills bad trials at epoch 15, 30 of Phase 1
- Storage: SQLite `hpo/optuna.db` (survives crashes, resumable)
- 30 trials × ~35 min each ≈ ~17 hrs (pruning cuts this to ~8 hrs)
- Best config auto-saved to `config/best_hparams.yaml`

```bash
# Retrain best model with optimal hyperparameters
python pipeline/mlflow_orchestrator.py \
    --model MR-TAE-FULL \
    --hparams config/best_hparams.yaml \
    --run-name "MR-TAE-FULL-HPO-BEST"
```

### B3 — Explainability

```bash
python evaluation/explainability/shap_analysis.py \
    --model-checkpoint results/MR-TAE-FULL-HPO-BEST/best.pth \
    --test-data data/test_real_qlin.pt \
    --n-samples 200

python evaluation/explainability/lime_analysis.py \
    --model-checkpoint results/MR-TAE-FULL-HPO-BEST/best.pth \
    --n-windows 20
```

---

## PHASE C — WGAN DATA AUGMENTATION

### C1 — Extract Real Noise Segments

From Q.Lin signals, extract "blank gap" segments (between PD events):
```python
# data/wgan/extract_noise_segments.py
# 1. Load Q.Lin .mat files
# 2. Run amplitude threshold to find inter-pulse gaps
# 3. Extract segments of 2048 samples where no PD event occurs
# 4. These are your REAL NOISE samples for WGAN training
```

### C2 — Train WGAN-Noise

```bash
python data/wgan/wgan_noise.py \
    --noise-segments data/real_noise_segments.pt \
    --output-model data/wgan/noise_generator.pth \
    --epochs 500 --batch 64
```

Training stops when MMD (Maximum Mean Discrepancy) on held-out noise segments plateaus.

### C3 — Augmented Retraining

After WGAN is trained, retrain MR-TAE-FULL with WGAN noise augmentation:
- 50% of training batches: synthetic pulse + mathematical noise (WGN/NBI/Impulse)
- 50% of training batches: synthetic pulse + WGAN-generated real noise

This bridges the synthetic→real domain gap.

---

## PHASE D — LOCAL DASHBOARD & MLOps STACK

### D1 — Research Dashboard (Flask Backend + React Frontend)

**Architecture:**
```
Flask API (port 8080)
├── GET /api/experiments          → list all MLflow runs
├── GET /api/experiment/{id}      → metrics, params, artifacts for one run
├── GET /api/training/live        → SSE stream of live training metrics
├── GET /api/benchmark            → comparison table JSON
├── GET /api/files/{path}         → serve checkpoint/figure files
├── POST /api/training/start      → launch a training run
└── POST /api/training/stop       → graceful interrupt + checkpoint save

React Frontend (port 3000, proxied through Flask in production)
├── Dashboard: live training curves, GPU stats, phase indicator
├── Experiments: sortable table of all MLflow runs with filter by metric
├── Benchmark: interactive comparison bar charts + heatmap
├── Files: file browser for checkpoints, figures, logs
└── Config: edit model_registry.yaml and training_config.yaml in browser
```

### D2 — Experiment Tracking Stack Comparison

| Platform | Best For | Your Use Case |
|---|---|---|
| **MLflow** (already using) | Local, simple, no cloud required | ✅ Keep as primary |
| **W&B** | Rich visualizations, team collab, sweep UI | Add for HPO visualization |
| **Neptune.ai** | Heavy artifact logging (signal plots, .mat files) | Optional |
| **TensorBoard** | Quick loss curve viewing | Already integrated in trainer |
| **Comet.ml** | Cloud-based, good for remote access | Not needed for local thesis |

**Recommendation:** Keep MLflow as primary + add W&B for HPO (Optuna has native W&B integration for beautiful sweep visualizations). Two tools is enough.

```python
# Add to optuna_study.py:
import wandb
wandb.init(project="ae-pd-denoising", name=f"hpo-trial-{trial.number}")
```

---

## PHASE E — CONTAINERISATION

### E1 — Docker (Single Machine)

**Purpose:** Anyone can reproduce your exact environment on any machine in one command.

```bash
# Build
docker build -t ae-pd-denoising:latest .

# Train (GPU passthrough)
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    -p 5000:5000 -p 8080:8080 \
    ae-pd-denoising:latest \
    python pipeline/mlflow_orchestrator.py --run-all

# Dashboard only
docker-compose up dashboard
```

### E2 — Kubernetes (Multi-GPU Future Scaling)

For thesis, Docker is sufficient. Kubernetes is for:
- Running 6 models in PARALLEL on 6 different GPUs (university compute cluster)
- Each model = one Kubernetes Job
- Results aggregated by a coordinator pod

```bash
kubectl apply -f k8s/training-job-mr-tae-full.yaml
kubectl get jobs -n ae-pd-denoising
```

---

## PHASE F — DATA ACQUISITION & MANAGEMENT

### F1 — Current Data Sources

| Dataset | Samples | Status |
|---|---|---|
| Q.Lin et al. (2022) .mat files | ~800 vectors × 4 SNR × 4 classes | Available |
| Synthetic AE (generated) | Unlimited | Implemented |
| WGAN-augmented | TBD | Not yet |

### F2 — Acquiring More Real Data

For a stronger thesis and real-world validation:
1. **Lin et al. IEEE TDEI 2022** — request full dataset from authors (cite in thesis)
2. **EPRI PD database** — US Electric Power Research Institute, apply for access
3. **IEC 60270 compliant test bench data** — contact supervisor about UM lab measurements
4. **ORNL/IEC open datasets** — search IEEE Dataport for "partial discharge acoustic emission"

### F3 — Data Versioning with DVC

```bash
# Initialize DVC
dvc init
dvc remote add -d local_store /external_drive/dvc_cache

# Track datasets
dvc add data/raw/qlin/
dvc add data/test_synthetic_ae.pt
dvc add data/test_real_qlin.pt
git add data/.gitignore data/*.dvc .dvc/config
git commit -m "Track datasets with DVC"
```

This means your git repo stays small, but anyone can pull the exact data version used for your thesis results.

---

## PHASE G — THESIS FINALISATION

### G1 — Auto-Generate Thesis Figures

```bash
python scripts/generate_thesis_figures.py \
    --benchmark results/benchmark/ \
    --format pdf \  # PDF figures for LaTeX/Word
    --dpi 300
```

Figures needed for each chapter:
- Chapter 3 (Methodology): architecture diagrams for all 6 models
- Chapter 4 (Results): Table 4.1 extended, Figure 4.1 extended, SNR sweep plots
- Chapter 5 (Discussion): ablation heatmap, efficiency scatter, SHAP overlay

### G2 — Final Checklist Before Viva

- [ ] Test set hashes logged in thesis (SHA256)
- [ ] All 6 models benchmarked on SAME test sets
- [ ] HPO completed and best model retrained
- [ ] Signal validation figure showing synthetic ≈ real AE pulse shape
- [ ] SHAP explanation figure ready for examiner "how does it work?" question
- [ ] `docs/EXAMINER_QA.md` reviewed and memorised
- [ ] Real Q.Lin data never touched during training (verify in DVC lineage)
- [ ] MLflow UI screenshot as evidence of systematic experiment tracking

---

## COMPLETE DEPENDENCY TIMELINE

```
Week 1:  Build test sets → Finish 6-model training → Run benchmark
Week 2:  HPO (30 trials) → Retrain best model → SHAP/LIME
Week 3:  WGAN noise augmentation → WGAN-augmented retraining
Week 4:  Dashboard + Docker → DVC data versioning → W&B sweep viz
Week 5:  Final documentation → Thesis figures → Submit
```
