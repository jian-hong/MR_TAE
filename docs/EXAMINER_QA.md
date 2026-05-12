# Examiner Q&A Prep (15 prompts)

1. **What problem does this pipeline solve?**  
   Unsupervised / weakly supervised denoising of Acoustic Emission (AE) partial-discharge–like pulses in high noise, with optional segmentation for pulse vs background.

2. **Why MR-TAE (multi-resolution temporal + attention/encoder) vs a plain U-Net?**  
   Long-range correlations (BiGRU) and local periodic structure (Swin-like) help separate impulsive pulses from colored WGN + impulse noise typical in AE.

3. **How is synthetic data built?**  
   Controlled pulse generators + noise injection (`PDSignalDataset`, `SignalConfig`); curriculum varies SNR and noise type.

4. **How is real Q.Lin data used?**  
   Sliding 2048-sample windows, z-score per window, labels by source file (defect size proxy); 90% fixed evaluation / 10% WGAN pool — see `mr_tae_fusion/data/qlin_loader.py`.

5. **What are the frozen test artifacts?**  
   `data/test_synthetic_ae.pt` and `data/test_real_qlin.pt` with SHA256 in `docs/TEST_SET_HASHES.md` for reproducibility.

6. **Why can’t real-set NCC be interpreted like synthetic NCC?**  
   No clean ground truth on disk; benchmark optionally scores against a Savitzky–Golay reference and documents the caveat in JSON/`RESULTS_COMPARISON.md`.

7. **How do you measure efficiency?**  
   `param_count_M`, batch-1 and batch-32 inference ms on GPU, training wall time from MLflow where available.

8. **What is the resilience design?**  
   `training/resilience.ResilienceManager` + session files + `scripts/resume.ps1` so long runs survive interrupts.

9. **What does MLOps add here?**  
   MLflow metrics/artifacts, dashboard for monitoring, Optuna SQLite study (`hpo/optuna.db`), benchmark aggregation under `results/benchmark/`.

10. **How does HPO interact with training?**  
    `hpo/optuna_study.py` optimizes e.g. `encoder_base_ch`, lr, weight decay, **`charbonnier_eps`** (wired into `MultiTaskLoss`), with curriculum SNR driven into the dataset each epoch.

11. **Explainability?**  
    SHAP DeepExplainer and segment LIME-style perturbation on encoder features (`evaluation/explainability/`).

12. **WGAN noise augmentation goal?**  
    Learn distribution of low-RMS inter-pulse noise from Q.Lin pool; train generator in `data/wgan/wgan_noise.py`; pipeline script `scripts/run_wgan_pipeline.ps1`.

13. **Auto-retrain (N2V-style)?**  
    `pipeline/auto_retrain.py` masks inputs, self-supervised MSE on masked samples; **safety gate** rejects weights if NCC vs test-real regression exceeds tolerance.

14. **Biggest limitation?**  
    Domain gap synthetic→field; label semantics on real data are equipment-specific; classical baselines can be competitive when DL checkpoints are missing or mis-loaded.

15. **How would you deploy?**  
    Export best checkpoint + `config/model_registry.yaml`; run `evaluation/benchmark_runner.py` on frozen `.pt` sets; serve `dashboard/app.py` read-only for demos; document hardware (e.g. RTX 4070) and env in `requirements.txt` / venv.

---

## Short-topic answers (from earlier drafts)

- **AE physics:** Pulses attenuate and disperse through oil/structures; boundaries cause reflections.  
- **Sensors:** Coupling and mounting affect bandwidth and repeatability.  
- **AE vs electrical:** AE resists EMI; electrical methods often more sensitive but EMI-prone.  
- **Sim limitations:** Synthetic pulses lack full structural reverberation.  
- **WGAN mitigations:** SN, GP, feature-matching term, MMD plateau heuristic in `wgan_noise.py`.  
- **Localization:** Pipeline targets denoising/classification, not geometry inversion.  
- **Mechanical noise:** Future work — add field mechanical transients to training.
