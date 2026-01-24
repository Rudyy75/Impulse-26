
# Plan: The Final Polish (Norm + Scheduler)

## Goal
Ensure the 50-epoch training run yields maximum possible accuracy.

## 1. Input Normalization (The "Fix")
**Problem:** Our spectrograms are in Decibels (roughly -80 to 0). Standard Neural Networks expect inputs around Mean 0, Std 1.
**Fix:** Add `nn.InstanceNorm2d` at the start of the model.
- This creates "Adaptive Contrast".
- It makes the model invariant to global volume differences (e.g., a quiet recording vs a loud one).

## 2. Learning Rate Scheduler (The "Boost")
**Problem:** Constant LR works, but decays help finetune.
**Fix:** Use `CosineAnnealingLR`.
- Start: High (`3e-4`) to explore.
- End: Low (`0`) to settle.

## 3. The FAISS Question
**User asked:** "Why not FAISS?"
**Answer:** Scale.
- **FAISS (IVF/HNSW):** Designed for 1 Million+ vectors. It uses *approximation* (clusters) to search fast ($O(\log N)$). It trades 5% accuracy for speed.
- **Our Database (8,000 vectors):** This is tiny.
    - Matrix Multiplication ($O(N)$) takes ~2ms on GPU.
    - FAISS overhead would actually make it *slower*.
    - Plus, Brute Force is **100% accurate**. We don't need approximation.

## Execution Steps
1.  **Modify `src/model.py`:** Insert `InstanceNorm2d`.
2.  **Modify `src/train.py`:** Insert `lr_scheduler`.
3.  **Start Training.**
