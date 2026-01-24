
# Plan: The Final Push (Mixup + 50 Epochs)

## The Evolution
1.  **Gaussian Noise Only:** 36% (Plateau).
2.  **Boosted Noise (5 epochs):** 36%.
3.  **Mixup (5 epochs):** 38% (Slight improvement).
4.  **Multi-View Indexing:** 36% (Regression).

## The Conclusion
- **Indexing:** Single Center Crop is optimal. Adding more views increases False Positives (Collisions).
- **Augmentation:** Mixup is the *only* thing that improved the score (from 36 -> 38).
- **Training Time:** Mixup is a tough regularizer. It makes the task harder. 5 epochs is likely underfitting.

## The Strategy
We commit to the **Long Training Run (50 Epochs)** with **Mixup**.
- **Model:** ResNet18 + InstanceNorm (Wait, removed? Yes, removed).
- **Augs:** Noise 0.1-0.4, Mixup 0.4.
- **Index:** Single Crop.

## Execution
1.  Set `EPOCHS = 50`.
2.  Run `src/train.py`.
3.  Run `src/index_db.py`.
4.  Run `src/test_acid.py`.
