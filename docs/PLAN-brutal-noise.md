
# Plan: "Brutal Noise" Robustness (Matching the Acid Test)

## The Problem (36% Ceiling)
Both 5-epoch and 50-epoch models hit exactly 36% accuracy.
- **Cause:** The model is overfitting to the "Easy Noise" regime (0.1-0.4 strength, 80% prob).
- **The Acid Test:** Uses ~0.3-0.5 strength (5dB SNR) with 100% probability.
- **Result:** When the model sees a "Very Noisy" sample in testing, it fails because it never learned to handle that level of entropy during training.

## The Solution: "Train Hard, Fight Easy"
 We must make training *harder* than the test.

### 1. Augmentation Boost (Implemented)
- **Gaussian Noise:** Range boosted to `[0.3, 0.6]`.
- **Probability:** Boosted to `95%` (Almost always noisy).
- **Mixup:** Boosted to `50%` chance.

### 2. Validation Loop (5 Epochs)
- We run `train.py` (5 epochs).
- We index and test.
- **Success Criteria:** Score > 40%. (If it beats 36% in just 5 epochs, the hypothesis is proven).

### 3. Production
- If successful, we revert to 50 epochs and train the final model.
