
# Plan: Multi-View Indexing (The "Brute Force" Solution)

## The Insight
- **Model Capability:** ResNet18 is good at matching texture (Denoising) but bad at matching long-range temporal structure (Intro vs Chorus).
- **The Bottleneck:** Our Database (Phase 2) only stores the **Center 5s**.
- **The Fix:** Don't ask the model to bridge the gap. **Bridge the gap in the Database.**

## The Method
1.  **Re-Baseline Training:** Restore `dataset.py` to "Same Crop" (done). Train 5 epochs to get a solid 36% Denoising model.
2.  **Modify `src/index_db.py`**:
    -   Instead of 1 crop (Center), extract **3 Crops**:
        -   **Start:** `0:00 - 0:05` (actually padding start if needed).
        -   **Center:** `Mid - 2.5s` to `Mid + 2.5s`.
        -   **End:** `Dur - 5.0s`.
    -   Store all 3 vectors with the same `filename`.
3.  **Search Logic:**
    -   Query vector `q`.
    -   Mathmul `q @ DB.T`.
    -   Find Top 1.
    -   If Top 1 matches any of the 3 vectors for Song X, it's a hit!

## Effort
- **Code:** Medium (Rewrite `index_db.py`).
- **Compute:** Indexing takes 3x longer (~60s vs 20s). Negligible.
- **Storage:** DB grows 3x. Still tiny (MBs).

## Verification
- Train 5 Epochs.
- Index Multi-View.
- Acid Test.
- **Expectation:** Score > 50%.
