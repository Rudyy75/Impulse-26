
# Plan: Temporal Invariance (The "Intro-Chorus" Bridge)

## The Root Cause (38% Ceiling)
- **Database:** Stores **Deterministic Center Crop** (e.g., 2:30-2:35).
- **Query:** Picked **Randomly** (e.g., 0:00-0:05).
- **Model:** Trained to map `Aug(A) -> Aug(A)`. It never learned that `Intro == Center`.
- **Result:** If the query is temporally distant from the center, the embedding distance is huge. Retrieval fails.

## The Solution: Temporal Contrastive Learning
We modified `src/dataset.py` to generate **two independent random crops** for each training step.
- `View 1`: Intro (Augmented)
- `View 2`: Chorus (Augmented)
- `Loss`: `Distance(Intro, Chorus) -> 0`.

This forces the model to learn a **Global Song Fingerprint** that ignores "When" the clip is from.

## Execution
1.  **Config:** Set `EPOCHS = 5`.
2.  **Train:** Run `src/train.py`.
3.  **Verify:** Index & Test.
4.  **Expectation:** Accuracy should jump significantly (>50%) because we are finally solving the retrieval geometry.
