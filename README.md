# Intelligent PD Denoising with Adaptive Capabilities

Final Year Thesis codebase for AE-based partial discharge denoising, ablation analysis, and benchmarking.

This repository focuses on:
- reproducible denoiser architecture experiments
- controlled ablation studies around MR-TAE components
- unified evaluation for model selection and viva defense

## 1) Project Goal

Given a noisy 1D AE signal (length 2048), denoise it while preserving PD morphology for downstream discrimination of:
- Corona
- Internal
- Surface

Primary evaluation priorities:
- SNR improvement
- NCC (shape fidelity)
- false positive rate
- efficiency (params and latency)

## 2) Repository Layout

```text
models/
  base.py                    # BaseDenoiser, AblationConfig
  variants.py                # MR-TAE ablations and cross-combination variants
  registry.py                # MODEL_ID -> class mapping
  components/blocks.py       # Shared reusable blocks (no copy-paste design)

training/
  train_all_ablations.py     # Unified trainer entrypoint

evaluation/
  benchmark_runner.py        # Unified benchmark + figures

data/
  wgan_augmentation/
    wgan_noise_aug.py        # Noise-distribution WGAN module
    wgan_pulse_aug.py        # Conditional pulse morphology WGAN module

docs/
  FILE_INVENTORY.md
  ARCHITECTURE_SUMMARY.md
  MODEL_COMPARISON.md
  EXAMINER_QA.md
  FINAL_VERDICT.md
```

## 3) Environment Setup

### Prerequisites
- Python 3.10+
- CUDA-enabled PyTorch recommended for training speed

### Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Framework Note

- PyTorch is the primary framework for model training and inference.
- TensorFlow is included only for compatibility with legacy/experimental workflows.

## 4) Quick Start

Run a single smoke training pass:

```bash
python training/train_all_ablations.py --model MR-TAE-FULL --phases 1 --epochs-per-phase 1 --batch-size 2 --length 256
```

Run benchmark generation:

```bash
python evaluation/benchmark_runner.py
```

Outputs are saved under `results/`.

## 5) Model IDs (for `--model`)

- `MR-TAE-FULL`
- `MR-TAE-noBiGRU`
- `MR-TAE-noSwin`
- `MR-TAE-noAttn`
- `MR-TAE-noMTL`
- `MR-TAE-noWavelet`
- `MWCNN-BiGRU`
- `MWCNN-Swin`
- `UNet-BiGRU-Swin`
- `UNet-BiGRU`
- `UNet-Attn`

## 6) Training Protocol (Target)

Curriculum phases:
- Phase 1 (easy): SNR [5, 15]
- Phase 2 (medium): SNR [-5, 5]
- Phase 3 (hard): SNR [-20, -5]

Reproducibility constraints:
- fixed seed = 42
- same split strategy across all model variants
- same evaluation set for fair comparison

## 7) Benchmark Outputs

`evaluation/benchmark_runner.py` produces:
- `results/benchmark_results.json`
- `results/benchmark_bar.png`
- `results/ablation_heatmap.png`
- `results/snr_sweep.png`
- `results/efficiency_scatter.png`

## 8) Data and Ethics Constraints

- Real Lin et al. data is reserved for testing/evaluation policy per thesis rules.
- Synthetic corruption types are mathematical noise injections:
  - AWGN
  - narrowband interference
  - impulse
  - powerline harmonics
- Do not conflate synthetic injected noise with physical acoustic propagation effects.

## 9) Git Hygiene

This repo intentionally ignores:
- environments (`myenv`, `.venv`, etc.)
- raw datasets and heavy binaries
- checkpoints and generated training outputs
- transient experiment artifacts

See `.gitignore` for details.

## 10) Suggested Next Steps

1. Finalize real-data split enforcement in the unified pipeline.
2. Run all ablation models with identical settings.
3. Export final benchmark JSON and regenerate all figures.
4. Fill `docs/MODEL_COMPARISON.md` with measured values.
5. Write the final empirical conclusions in `docs/FINAL_VERDICT.md`.
6. Tag a reproducible release commit for viva.
