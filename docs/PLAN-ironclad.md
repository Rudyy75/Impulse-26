
# Plan: "Ironclad" Robustness Upgrade (Self-Mixup + Bandpass)

## Goal
Achieve >70% on Acid Test (Current: 30%).

## Strategy
1.  **Background Noise Injection (Self-Mixup):**
    - The Acid Test uses Street Traffic/Cafe noise (Structured Sound).
    - We will simulate this by mixing the target song with *another random song* from the dataset at low volume.
    - **Math:** $x_{aug} = x_{target} + \alpha \cdot x_{noise}$, where $\alpha \in [0.1, 0.4]$.
    - **Safety:** We limit $\alpha < 0.5$ so the *Target* is always the dominant signal.

2.  **Bandpass Filtering:**
    - Real "Shazam" queries happen on phones.
    - Phone mics kill Sub-bass (<300Hz) and Highs (>4000Hz).
    - We will mask out the top and bottom rows of the spectrogram to force the model to focus on the Mids.

## Code Path
1.  **`src/augmentations.py`**:
    - Add `mix_background(spec, noise_spec)`.
    - Add `bandpass_filter(spec)`.
2.  **`src/dataset.py`**:
    - In `__getitem__`, load a *second* random index `noise_idx`.
    - Pass this `noise_spec` to the augmentor.
3.  **`src/train.py`**:
    - Retrain for 50 epochs.
