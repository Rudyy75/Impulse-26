# Plan: GenAI Audio Synthesis (Phase 5)

## Goal
Demonstrate that the learned latent space is semantic and continuous by interpolating between two songs and generating a "hybrid" audio clip.

## Core Problem
Embeddings ($z$) are highly compressed. Inverting them to full audio is an "ill-posed" problem.
**Solution:** Train a **Decoder Network** to learn the mapping $z \to \text{Spectrogram}$, then use **Griffin-Lim** to estimate phase and reconstruct audio.

## Architecture
**1. Frozen Encoder:**
- Uses the Pre-trained `EchoFindModel` (ResNet50).
- Input: Audio -> MelSpec -> Encoder -> $h$ (2048-dim feature vector).
- We use $h$ (Backbone Output) instead of $z$ (Projection) because $h$ preserves more content information.

**2. Decoder Network (`src/decoder.py`):**
- **Input:** vector (2048)
- **Linear:** Project to $2048 \times 4 \times 4$.
- **Upsampling Blocks:** Series of `ConvTranspose2d` + `BatchNorm` + `ReLU`.
    - 4x4 -> 8x8
    - 8x8 -> 16x16
    - 16x16 -> 32x32
    - ... -> Target Size ($128 \times 216$).
- **Output:** Log-Mel Spectrogram (1 Channel).

## Training Strategy (`src/train_decoder.py`)
- **Loss:** MSE (Mean Squared Error) between Original Spec and Reconstructed Spec.
- **Data:** FMA Unsupervised Dataset (Raw Audio).
- **Process:** 
    1. Pass Audio -> Encoder -> $h$ (Detach gradients).
    2. Pass $h$ -> Decoder -> $\hat{S}$.
    3. Loss($S, \hat{S}$).
- **Epochs:** 20-50.

## Synthesis Pipeline (`src/synthesis.py`)
1.  **Select Pairs:** Pick two contrasting tracks (Track A, Track B).
2.  **Encode:** Get $h_A$ and $h_B$.
3.  **Interpolate:** $h_{new} = \alpha h_A + (1-\alpha) h_B$.
4.  **Decode:** $\hat{S}_{new} = \text{Decoder}(h_{new})$.
5.  **Vocoder:** Use `torchaudio.transforms.GriffinLim` to convert $\hat{S}_{new} \to \text{Audio Waveform}$.
6.  **Save:** Export `morph.wav`.

## Tasks
1. [ ] Create `src/decoder.py` (Model).
2. [ ] Create `src/train_decoder.py` (Training).
3. [ ] Create `src/synthesis.py` (Inference).
