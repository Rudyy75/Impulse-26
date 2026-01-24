# Phase 1: Input Pipeline - Completed

## 1. Concept Explanation
The goal of this phase was to build a robust data pipeline that transforms raw audio into "views" suitable for Contrastive Learning.

**Key Concepts:**
- **Spectrograms:** We convert raw waveforms (1D) into Mel-Spectrograms (2D) to reveal frequency content over time.
- **Log Compression:** We use decibels (Log scale) to match human hearing and compress the dynamic range for neural networks.
- **SpecAugment:** We mask random time and frequency strips to force the model to learn context, not just local texture.
- **Generic Noise:** Essential for the "Shazam Test" - simple Gaussian noise teaches the model to ignore background static.

## 2. Implementation Journey
We implemented the pipeline in three modular files:

1.  **`src/spectrogram.py`**:
    - Handles loading with `torchaudio`.
    - Resamples to 22050Hz (Nyquist limit ~11kHz).
    - Converts Stereo to Mono.
    - Implemented `audio_to_melspec` with `n_fft=2048`, `hop_length=512`.

2.  **`src/augmentations.py`**:
    - Created a stochastic `AugmentationPipeline`.
    - `add_noise`: Robustness to static.
    - `time_mask`: Robustness to dropouts.
    - `freq_mask`: Robustness to EQ changes.
    - `time_shift`: Temporal invariance.

3.  **`src/dataset.py`**:
    - Implemented `FMAContrastiveDataset`.
    - Returns `(view1, view2)` for every index.
    - Uses the stochastic pipeline to ensure every epoch sees different variations.

## 3. Verification Results
We verified the pipeline using `src/test_phase1.py`.

- **Dataset Size:** 8000 tracks found.
- **Tensor Shape:** Correctly `(1, 128, 216)` (1 channel, 128 mels, ~5 seconds).
- **Augmentation Check:** Confirmed that `view1` and `view2` are different tensors (augmentations are active).

## 4. Key Code Sections

### The Spectrogram Transform
```python
mel_transform = T.MelSpectrogram(
    sample_rate=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)
log_mel = torch.log(mel_spec + 1e-9) # Epsilon prevents -inf
```

### The Stochastic Augmentation
```python
def __call__(self, spec):
    out = spec.clone() # Critical: don't modify original
    if random.random() < 0.8: out = self.add_noise(out)
    if random.random() < 0.5: out = self.time_mask(out)
    return out
```
