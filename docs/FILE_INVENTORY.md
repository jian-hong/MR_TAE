# File Inventory (Pre-Restructure Audit)

This document audits the current repository state before architecture restructuring.

## Scope

- **Primary active codebase audited:** `mr_tae_fusion/`, `examples/`, `mr_tae_gem/`
- **Archived/legacy snapshot also checked for thesis questions:** `Data/OneDrive_2025-12-13/`
- **Excluded from audit:** `myenv/`, duplicated site-packages content

## Direct Answers to Mandatory Questions

- **BiGRU in `mr_tae_ultimate.py`:** yes, it is configured as bidirectional (`bidirectional=True`) with `hidden_size=bottleneck_ch // 2` in the archived file.
- **`train_extended.py` curricular 3-phase training:** not implemented in `examples/train_extended.py`; it uses epoch-based training with dataset `update_epoch()` and cosine warm restarts, but no explicit thesis 3-phase controller.
- **`distance_signals.py` noise types (archived):** archived `Data/.../mr_tae_fusion/data/distance_signals.py` includes AWGN + impulse and AEPD/reflection-like components; it does **not** cleanly implement thesis AWGN + NBI + Impulse + Powerline as separate composable components.
- **Models not in thesis baseline set:** `TGAN` noise generator/discriminator, `DistanceClassifier`, `DistanceClassifierWithDenoising`, and GEM-related training pipeline (`mr_tae_gem`).

## Model Definition Inventory

### `mr_tae_fusion/models/mr_tae_fusion.py`

- **Purpose:** Main MR-TAE-Fusion denoiser + segmentation model.
- **Classes:**
  - `HybridBottleneck` — **2,437,104** params (default instantiation)
  - `MRTAEFusion` — **3,053,468** params (default `ModelConfig`)
- **Training status:** no local project checkpoints found under primary repo (`*.pth`/`*.pt` absent).
- **Thesis match:** closest to proposed thesis architecture (MWCNN + BiGRU + Swin + attention + MTL).

### `mr_tae_fusion/models/mwcnn.py`

- **Purpose:** Wavelet encoder/decoder components.
- **Classes:**
  - `MWCNNEncoder` — **127,392** params
  - `MWCNNDecoder` — **324,576** params
  - `ConvBlock`, `MWCNNBlock`, `MWCNNDecoderBlock` — building blocks (require constructor args; not counted standalone)
- **Training status:** used as submodules, no independent checkpoint.
- **Thesis match:** architecture component (wavelet branch) aligns with thesis.

### `mr_tae_fusion/models/swin_transformer.py`

- **Purpose:** 1D Swin Transformer blocks.
- **Classes:**
  - `SwinTransformer1D` — **1,581,040** params (default)
  - `WindowAttention1D`, `MLP`, `SwinTransformerBlock` — components (arg-dependent)
- **Training status:** only as submodule.
- **Thesis match:** part of thesis bottleneck.

### `mr_tae_fusion/models/attention.py`

- **Purpose:** Skip attention gating.
- **Classes:** `AttentionGate`, `MultiHeadAttentionGate`, `GatedSkipConnection` (arg-dependent components).
- **Training status:** only as submodule.
- **Thesis match:** aligns with thesis attention-gated skip strategy.

### `mr_tae_fusion/models/fusion_head.py`

- **Purpose:** Reconstruction and segmentation heads.
- **Classes:**
  - `ReconstructionHead` — **1,601** params
  - `DualStreamFusionHead`, `AttentionFusionHead` — arg-dependent heads
- **Training status:** only as submodule.
- **Thesis match:** MTL head aligns with thesis design intent.

### `mr_tae_fusion/models/tgan.py`

- **Purpose:** GAN-based noise generation augmentation.
- **Classes:**
  - `NoiseGenerator` — **2,046,977** params
  - `NoiseDiscriminator` — **853,697** params
  - `TGANNoiseLoader`, `TGAN` utility wrappers
- **Training status:** no local checkpoint found in active root.
- **Thesis match:** variant/extension; not one of the core four denoiser architectures.

### `Data/OneDrive_2025-12-13/Try Model/mr_tae_fusion/models/distance_classifier.py` (archived)

- **Purpose:** distance/classification model family.
- **Classes:**
  - `DistanceClassifier` — **8,281,540** params
  - `DistanceClassifierWithDenoising` — arg-dependent (not default-instantiated)
- **Training status:** archived script references saved checkpoint generation.
- **Thesis match:** not core denoiser architecture from thesis table; auxiliary classifier track.

### `Data/OneDrive_2025-12-13/mr_tae_ultimate.py` (archived)

- **Purpose:** older "ultimate" training/model script.
- **Classes:** contains BiGRU-Swin style architecture (module import execution required unavailable dependency on local shell for automatic param extraction).
- **Training status:** archived.
- **Thesis match:** variant/legacy implementation of thesis concept.

## Training / Pipeline Script Notes

### `examples/train_extended.py`

- Uses standard epoch loop, mixed precision, cosine warm restarts.
- Not explicit thesis 3-phase curricular script (`--phases` style absent).

### `mr_tae_fusion/training/trainer.py`

- Has implicit phase scheduling via dataset `update_epoch()`.
- Saves best and periodic checkpoints every 10 epochs.
- Validation currently tied to dataset mode/update flow; explicit full-range validation guardrail is not enforced in code comments/arguments.

### Checkpoint Status (active root)

- No project checkpoints discovered in active root (`*.pt`, `*.pth` not found, excluding environment `.pth` files).
- Therefore active models are currently treated as **untrained** in the primary workspace state.

## Final State Addendum (Post-Restructure in This Task)

Newly added structure and files:

- `models/base.py`
- `models/variants.py`
- `models/registry.py`
- `models/components/blocks.py`
- `training/train_all_ablations.py`
- `evaluation/benchmark_runner.py`
- `data/wgan_augmentation/wgan_noise_aug.py`
- `data/wgan_augmentation/wgan_pulse_aug.py`
- `docs/ARCHITECTURE_SUMMARY.md`
- `docs/EXAMINER_QA.md`
- `docs/MODEL_COMPARISON.md`
- `docs/FINAL_VERDICT.md`

These additions provide the requested ablation-ready scaffold, unified training entrypoint, benchmark script, and documentation framework.

