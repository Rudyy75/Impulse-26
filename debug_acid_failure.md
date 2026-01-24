
## 🔍 Debug: Acid Test Failure (18.0% Accuracy)

### 1. Symptom
The model simulates a "Shazam" retrieval task with 5dB SNR (Street Noise) and Heavy Reverb.
- **Expected:** >80% accuracy (robust to noise).
- **Actual:** 18.0% accuracy (fragile).

### 2. Information Gathered
- **Training Loss:** Dropped to 0.022 (Very low).
- **Phase 2 Verification:** 100% on "Clean" augmentations.
- **Phase 3 Verification:** 99.9% on Exact Match.
- **Acid Test:** 18.0% on 5dB SNR/Reverb.

### 3. Hypotheses
1.  🎯 **Weak Training Augmentations:** Our training pipeline (`src/augmentations.py`) likely used mild noise (`std=0.03`) and standard masking. It never saw "heavy reverb" or "street noise" during training, so it learned to rely on fragile frequency details that get destroyed by real noise.
2.  ❓ **Domain Shift:** The "Street Noise" simulation in `test_acid.py` is mathematically different from the `AugmentationPipeline` noise.
3.  ❓ **Overfitting:** The model memorized the exact training clips (including their specific augmentations) but didn't learn general invariant features.

### 4. Investigation
**Check `src/augmentations.py`:**
- Current `noise_std = 0.03`.
- `test_acid.py` uses `snr_db=5`.
- **Math Check:** 5dB SNR implies the signal is only ~1.7x stronger than noise. `std=0.03` is likely ~20-30dB SNR (very clean).
- **Conclusion:** The model was trained in a distinctively "Cleaner" environment than the test.

### 5. Root Cause
**The "Campfire" Fallacy.**
We trained the model in a quiet room (mild noise), but we are testing it in a construction zone (5dB noise). It never learned to ignore massive disruption because it never needed to.

### 6. Fix Plan (Enhancement)
We must **RETRAIN** (Phase 2) with "Acid-Level" augmentations.
1.  **Enhance `src/augmentations.py`:**
    - Increase `noise_std` to range `[0.05, 0.2]`.
    - Add `RIR / Reverb` simulation to the training pipeline.
    - Add `Bandpass Filter` (simulate phone microphone).
2.  **Retrain:** Run `src/train.py` for 50 epochs again using the new tough data.

### 7. Prevention
Always train with augmentations *stronger* than the expected test conditions.
