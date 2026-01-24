
# Plan: Asymmetric "Denoising" Contrastive Learning

## The Insight
- **Current State (38%):** We train `Aug(Crop1) <-> Aug(Crop2)`. The model learns to match two noisy versions. It might settle on "Noise Pattern A matches Noise Pattern B" rather than "Song Content".
- **The Task:** Match `Noisy Query` <-> `Clean Center Crop`.
- **The Gap:** We never explicitly train the model to map Noisy -> Clean.

## The Fix: Asymmetric Pairs
We will modify `src/dataset.py` to produce:
1.  **View 1 (Anchor):** Clean Random Crop (Simulates the Database entry).
2.  **View 2 (Positive):** Augmented Random Crop (Simulates the Acid Test Query).
    - Augmentations: Brute Noise (0.1-0.4), Mixup (simulating background chatter).

This forces the encoder `f(x)` to satisfy: `f(Noisy) ≈ f(Clean)`.

## Steps
1.  **Modify `src/dataset.py`**:
    -   Crop `spec` once.
    -   Set `view1 = spec.clone()` (Clean).
    -   Set `view2 = augment(spec, noise_spec)` (Dirty).
    -   Ensure `view1` has no augmentations (except maybe crop).
    -   **Important:** `AugmentationPipeline` has `time_shift`. We CANNOT use `time_shift` on `view2` if `view1` is static, unless `view1` is shifted too?
    -   **Decision:** Remove `time_shift` from `augmentations.py` or apply it to *both*?
    -   If `view2` is shifted 1s, and `view1` is not, they are misaligned. CNN might fail.
    -   **Correction:** We should disable `time_shift` for this experiment OR apply the same shift to both.
    -   Given Acid Test simulates `time_shift` via "Random Split", the cropping handles the shift naturally. We don't need `torch.roll` shift.

2.  **Verification**: 5 Epoch Train -> Index -> Acid Test.
