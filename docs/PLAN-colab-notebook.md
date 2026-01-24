
# Plan: EchoFind Colab Pro "Max Performance" Notebook

## Context
- **User Goal:** Maximize results (Semantics + Robustness) using Google Colab Pro.
- **Constraints:** Self-contained `.ipynb` (no git clone), > 50 Epochs.
- **Strategy:** Scale up everything (Model size, Batch size, Training time).

## The "Best Result" Config
To achieve the "Best Result Ever", we will upgrade:
1.  **Backbone:** `ResNet18` -> `ResNet50` (Deeper features, higher capacity).
2.  **Projection:** `128 dim` -> `256 dim` (Standard SimCLR).
3.  **Batch Size:** `64` -> `256` (Critical for Contrastive Learning).
4.  **Epochs:** `50` -> `200` (SimCLR converges slowly).
5.  **Precision:** `FP32` -> `FP16` (AMP) (Faster training, less memory).

## Notebook Structure (`EchoFind_Pro.ipynb`)

### Cell 1: Setup & Dependencies
-   Install `torchaudio`, `librosa`, `tqdm`.
-   Mount Google Drive for persistent storage.

### Cell 2: Data Pipeline (Inline)
-   `wget` FMA Small dataset (Direct link).
-   `unzip` to local Colab disk (fast I/O).
-   Inline `AugmentationPipeline` (The "Goldilocks" Version: Noise 0.1-0.4, Mixup 0.4).
-   Inline `FMAContrastiveDataset` (Single Crop logic).

### Cell 3: Model Architecture (Inline)
-   Define `SimCLR` class.
-   Replace `resnet18` with `resnet50`.
-   Add `InstanceNorm` (Optional: Standard SimCLR uses BatchNorm, but Audio often prefers Instance. We will stick to InstanceNorm for safety based on our experiments).

### Cell 4: Training Engine (Inline)
-   `train()` loop.
-   **Mixed Precision (AMP):** `torch.cuda.amp.GradScaler`.
-   **Scheduler:** `CosineAnnealingLR` (Vital for 200 epochs).
-   **Checkpointing:** Save `best_model.pt` to Drive every 10 epochs.

### Cell 5: Verification (Inline)
-   Inline `Acid Test` (Robustness Check).
-   Inline `Linear Probe` (Semantics Check).

## Verification Strategy
-   The user runs the notebook.
-   We expect Semantics F1 > 60% (due to ResNet50 + 200 Epochs).
