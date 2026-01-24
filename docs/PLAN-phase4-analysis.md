
# Plan: Representation Analysis (Linear Probe & t-SNE)

## The Objective
Prove that the Self-Supervised Model learned "Music Concepts" (Genre) without ever seeing a label.

## The Data
- **Inputs:**
    - `data/vector_db.pt`: Contains 8000 embeddings (Pre-computed).
    - `data/fma_metadata/tracks.csv`: Contains Genre labels.

## Task 1: Linear Probe (`src/linear_probe.py`)
1.  **Load Embeddings:** Read `vector_db.pt`.
    -   Parse `Track ID` from filename (e.g., `000123.mp3` -> `123`).
2.  **Load Labels:** Read `tracks.csv` (using `pandas`).
    -   Header logic: Row 0 is type, Row 1 is sub-header.
    -   Filter: `subset == 'small'`.
    -   Target: `track['genre_top']`.
3.  **Align:** Match Embedding ID to Genre Label.
4.  **Split:**
    -   **Train:** 10% of data (800 samples).
    -   **Test:** 90% of data (7200 samples).
    -   *Why?* To prove "Few-Shot" capability.
5.  **Train:** `sklearn.linear_model.LogisticRegression`.
6.  **Report:** Accuracy, F1-Score, Classification Report.

## Task 2: Visualization (`src/visualize.py`)
1.  **Reduce:** Use `sklearn.manifold.TSNE` (or `UMAP` if installed).
    -   Project 512D -> 2D.
2.  **Plot:** `matplotlib` scatter plot.
    -   Color by: Genre (8 classes).
    -   Save to `docs/tsne_genre.png`.

## Verification
- Run `python -m src.linear_probe`.
- Check if F1 > 40%.
- Check `docs/tsne_genre.png` for clusters.
