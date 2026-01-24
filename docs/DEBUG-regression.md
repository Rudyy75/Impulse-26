
# 🚨 Emergency Debug Plan: Regression to 18%

## The Issue
We added:
1.  **Self-Mixup** (Overlaying random songs).
2.  **Bandpass** (Zeroing frequencies).
3.  **InstanceNorm2d** (Adaptive contrast).
4.  **Scheduler** (Cosine Decay).

Result: **18% Accuracy** (Worse than 30%).
Previous Baseline: 18% (Clean) -> 30% (Simple Noise).

## Hypotheses

### 1. The "Signal Destruction" Hypothesis (InstanceNorm)
- **Problem:** `InstanceNorm2d(1)` normalizes *each sample* to Mean=0, Std=1.
- **Why it's bad:** In Log-Mel Spectrograms, "Silence" is -80dB. "Loud" is 0dB. Normalization forces Silence to become Mean 0 (gray), and Loud to become Mean 0. It destroys the "Energy Profile" of the song. A quiet piano intro now looks like a loud wall of noise.
- **Action:** **REMOVE InstanceNorm.**

### 2. The "Label Noise" Hypothesis (Mixup)
- **Problem:** We mixed Target with Noise at `alpha` up to 0.4.
- **Why it's bad:** If `alpha=0.4`, the background song is nearly half as loud as the foreground. The Contrastive Loss might be pulling the embedding towards the *Background Song's* cluster (Confusion).
- **Action:** Reduce Alpha to `[0.05, 0.2]`.

## Corrective Steps
1.  **Revert:** Remove `input_norm` from `src/model.py`.
2.  **Soften:** Reduce mixup strength in `src/augmentations.py`.
3.  **Retrain:** 50 Epochs (This is painful but necessary).
