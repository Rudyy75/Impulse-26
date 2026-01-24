
# Enhancement Plan: Robust Training for Acid Test Survival

## Goal
Improve model retrieval accuracy on the 5dB SNR "Acid Test" from 18% to >80%.

## Problem
The current model was trained with mild adjustments (clean room). The test simulates heavy street noise and reverb (construction zone). The model fails because it over-relies on clean frequency details.

## Proposed Changes

### 1. `src/augmentations.py` [MODIFY]
We will turn the "gentle" pipeline into a "rugged" pipeline.
- **Increase Gaussian Noise:** `std` from `0.03` to range `[0.05, 0.15]`.
- **Add RIR Simulation:** A crude reverb simulation (decaying echo).
- **Add Bandpass Filter:** Simulate crappy phone microphones (keep only 300Hz-3400Hz).
- **Increase Masking:** Make SpecAugment harsher (bigger masks).

### 2. `src/train.py` [RUN]
- Retrain from scratch (or finetune) for 50 epochs.
- Since the task is harder, the loss might be higher, but the *utility* will be better.

## Verification Plan

### Automated
1.  **Run Acid Test:** `python -m src.test_acid`
    - Success Condition: Accuracy > 60% (Acceptable), > 80% (Excellent).

### Manual
1.  **Visual Check:** inspect `phase1_verification.png` after updating augmentations to ensure the spectrograms are still recognizable as music (not just pure static).
