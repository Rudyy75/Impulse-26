
# Phase 4: Representation Analysis (Did it learn Music?)

## The Goal
Prove that the model's embeddings are **Semantically Rich** (Genre-aware) despite being trained only on **Self-Supervised Noise Denoising**.

## The Hypothesis
If `Embed(Song A)` is close to `Embed(Song B)`, they likely share high-level attributes (Instrumentation, Tempo, Genre).
Therefore, a simple Linear Classifier (Probe) should be able to predict Genre from these embeddings with high accuracy, even if trained on only 10% of the data.

## The Plan
### 1. Prerequisite: Metadata
**BLOCKER:** We need `tracks.csv` from the FMA Metadata to map `TrackID` -> `Genre`.
- **Action:** Download `fma_metadata.zip` (350MB) and extracting `tracks.csv`.

### 2. Linear Probe (`src/linear_probe.py`)
- **Input:** Frozen Model, Vector DB (or compute fresh on val set), `tracks.csv`.
- **Process:**
    1. Load all embeddings.
    2. Join with `tracks.csv` to get Genre Labels.
    3. Split: Train (10%), Test (90%).
    4. Train Logistic Regression (`sklearn`).
- **Metric:** F1-Score (Micro & Macro).

### 3. Visualization (`src/visualize.py`)
- **Algorithm:** t-SNE (or UMAP if available).
- **Plot:** 2D Scatter Plot.
- **Color:** Genre (Top 8 Genres).
- **Expectation:** Distinct clusters for "Hip-Hop", "Classical", "Rock".

## Success Criteria
- **Probe F1:** > 40% (Random Guess is ~1/8 = 12%).
- **Visuals:** Visible separation of clusters.
