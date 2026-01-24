
# Plan: Rapid Prototyping (Velocity Check)

## The Constraint
Training 50 epochs takes too long (~40 mins) to discover failure.
User wants to iterate fast.

## The Strategy: "Fail Fast, Fail Cheap"
1.  **Reduce Epochs to 5:** A healthy model should reach ~35-40% accuracy on the Acid Test by Epoch 5 (trajectory toward 70%).
    - If it's at 10%, we kill it.
    - If it's at 40%, we scale it.
2.  **Random 5 Epochs?** No, we must train from scratch (0-5). Pre-selecting random epochs doesn't make sense in optimization.
3.  **Verification Loop:**
    - Train (5 epochs) -> Index -> Test Acid.
    - Total Loop Time: ~5 minutes.

## Actions
1.  Modify `src/config.py`: Set `EPOCHS = 5`.
2.  Run `train.py`.
3.  Run `index_db.py`.
4.  Run `test_acid.py`.
5.  **Evaluate:** If Score > 35%, restore `EPOCHS = 50` and run full storage.
