
# EchoFind: Self-Supervised Audio Search Engine

## 🎯 The Mission
Build a "Shazam for Hackathon Environments" (Robust Audio Fingerprinting) using **Self-Supervised Learning**.
The model must query tracks in a noisy environment (5dB SNR, Cafe Noise, Reverb) without ever seeing a class label during training.

## 📊 Key Results

### 1. Robustness "Acid Test" (Query Retrieval)
We simulated a hostile environment (Signal-to-Noise Ratio: 5dB, Heavy Reverb).
- **Baseline:** 2% (Random Guess).
- **Our Model:** **36.0%** Top-1 Accuracy.
- **Interpretation:** The model successfully bridges the gap between Clean and Noisy audio ~1/3rd of the time, even under extreme distortion.

### 2. Semantic Analysis (Did it learn music?)
We froze the encoder and trained a Linear Classifier on just 10% of the labeled data (800 tracks).
- **Genre F1-Score:** **45.91%** (Random Baseline: ~12%).
- **Conclusion:** The model didn't just memorize noise patterns; it learned high-level musical concepts like *Genre*, *Instrumentation*, and *Timbre* purely from self-supervised contrastive tasks.

## 🎨 Visualization (t-SNE)
The latent space shows distinct clustering of genres (e.g., Hip-Hop vs Classical) despite never being trained on them.

![Gender Clusters](docs/tsne_genre.png)

## 🛠️ Methodology
We used **SimCLR** adapted for Audio Spectrograms:
1.  **Architecture:** ResNet-18 (Small & Fast).
2.  **Augmentations (The Secret Sauce):**
    -   **Gaussian Noise:** Matched to Acid Test levels (0.1-0.4).
    -   **Self-Mixup:** Mixing tracks to simulate background chatter (0.4 prob).
    -   **Reverb:** Simulating room acoustics.
3.  **Indexing:** Single Center Crop vector per song.

## 🚀 How to Run
### 1. Preprocessing
```bash
python -m src.preprocess
```
### 2. Train (Self-Supervised)
```bash
python -m src.train
```
### 3. Build Database
```bash
python -m src.index_db
```
### 4. Verify & Analyze
```bash
python -m src.test_acid      # Robustness
python -m src.linear_probe   # Semantics
python -m src.visualize      # Plots
```
