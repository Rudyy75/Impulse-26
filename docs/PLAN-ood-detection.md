# Plan: Zero-Shot OOD Detection (Phase 6)

## Goal
Implement an automated "Gatekeeper" that rejects non-music audio (e.g., Speech, Noise, Silence) by analyzing the density of the embedding space.

## Concepts
- **In-Distribution (ID):** The FMA Music Dataset (Manifold of Music).
- **Out-of-Distribution (OOD):** Anything else.
- **Hypothesis:** Music embeddings cluster together. OOD embeddings will land in "sparse" regions of the latent space.

## Approach
We will implement **Distance-Based Anomaly Detection**.

### 1. The Density Estimator (`src/ood.py`)
We will use **k-Nearest Neighbors (kNN) Distance**.
- **Fit:** Load all 8000 training embeddings ($E_{train}$).
- **Score($z$):** Calculate mean distance to the $k=50$ nearest neighbors in $E_{train}$.
- **Decision:**
    - If `mean_dist > threshold`, return `IS_OOD = True`.
    - Else, return `IS_OOD = False`.
- **Thresholding:** We will set the threshold automatically using the **95th percentile** of the training distances. (Anything further than 95% of known music is suspicious).

### 2. Integration (`src/query.py`)
Modify `query_audio` to check OOD score before searching.
```python
if ood_detector.is_anomaly(h):
    print("⚠️ Anomaly Detected: Input does not sound like music.")
    return []
```

### 3. Verification
- We will manually test with:
    1. A valid music file (Should pass).
    2. A random noise file or speech file (Should fail).

## Implementation Steps
1. [ ] Create `src/ood.py`:
    - Class `OODDetector`
    - Method `fit(embeddings)`
    - Method `score(embedding)`
    - Serialize fitted detector to `weights/ood_model.pkl`.
2. [ ] Train OOD Detector:
    - Run `python -m src.ood` (Loads vector db, fits KNN, saves model).
3. [ ] Integrate into `src/query.py`.
