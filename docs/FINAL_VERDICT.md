# Final Verdict

*Updated from the latest automated benchmark aggregation in `results/benchmark/comparison_table.json`. Re-run `python -m evaluation.benchmark_runner` (or your orchestrator hook) after all `results/<MODEL_ID>/checkpoints/best.pt` exist for a complete leaderboard.*

## Snapshot (partial run on file system)

The committed table included **classical baselines** (`wavelet_db4_soft`, `savitzky_golay`) and **one DL row** (`MR-TAE-noMTL`). Many MR-TAE checkpoints were absent or incomplete during that run, so **`MR-TAE-noMTL` reported negative NCC** — consistent with a bad/partial weight load (see training logs for `model_state` warnings).

### Synthetic test set (primary for ranking DL)

| Model | NCC mean ↑ | SNR impr. (dB) |
|--------|------------|----------------|
| wavelet_db4_soft | **0.689** | **6.45** |
| savitzky_golay | 0.625 | 5.27 |
| MR-TAE-noMTL | -0.132 | 4.64 |

### Real Q.Lin (reference-based; see `real_note` in JSON)

Interpret cautiously: metrics use a **Savitzky–Golay reference** when no clean GT exists — high NCC for SG-like methods is expected.

## Most robust model (tentative)

Until all checkpoints are trained: **wavelet_db4_soft** on this slice of the table. After full training, expect **`MR-TAE-FULL`** or **`MR-TAE-noSwin`** to lead on synthetic NCC — confirm numerically.

## Most efficient model

**Classical methods** — ~0 param, ~0.5 ms batch-1 in the stub latency model. Among DL rows with valid checkpoints, compare `inference_ms_b1` / `param_count_M` in `comparison_table.json`.

## Most impactful ablation

Hypothesis: **`MR-TAE-noMTL`** vs full MTL — segmentation auxiliary task stabilizes the shared representation. *Verify after checkpoints are fixed.*

## WGAN benefit

**Not yet quantified** in the JSON snapshot. After `scripts/run_wgan_pipeline.ps1` and retraining with mixed noise, add a row to `docs/RESULTS_COMPARISON.md` with before/after NCC at SNR -10 dB.

## BiGRU vs Swin

Compare **`MR-TAE-noBiGRU`** vs **`MR-TAE-noSwin`** on synthetic SNR sweep (`ncc_snr_*` columns) once both are trained.

## Recommendation

1. Complete `--run-all` training (`docs/TRAINING_LAUNCH.md`).  
2. Re-run benchmark with all `best.pt` present.  
3. Run Optuna (`hpo/optuna_study.py`, 30 trials) and save `config/best_hparams.yaml`.  
4. Refresh this file from the new `comparison_table.json` and per-model `benchmark_results.json`.
