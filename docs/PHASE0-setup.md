# Phase 0: Environment Setup

## Overview

This phase covers the foundational setup required before we can start building the EchoFind system. Getting the environment right is critical - ML projects often fail due to version mismatches, missing dependencies, or GPU issues that only surface hours into training.

---

## What We're Setting Up

### 1. Project Structure

The project follows a modular design where each phase has its own module:

```
echofind/
├── src/                  # Core source code
│   ├── __init__.py       # Package marker
│   ├── config.py         # All hyperparameters in one place
│   ├── spectrogram.py    # Audio to mel spectrogram (Phase 1)
│   ├── augmentations.py  # Data augmentation (Phase 1)
│   ├── dataset.py        # Contrastive dataset (Phase 1)
│   ├── model.py          # Encoder + projection head (Phase 2)
│   ├── losses.py         # NT-Xent loss (Phase 2)
│   ├── train.py          # Training loop (Phase 2)
│   ├── retrieval.py      # Vector search (Phase 3)
│   └── evaluate.py       # Linear probe + visualization (Phase 4)
├── scripts/              # Utility scripts
│   └── verify_gpu.py     # Environment verification
├── submission.py         # Entry point for organizers
├── requirements.txt      # Dependencies
├── data/                 # Dataset goes here (gitignored)
├── weights/              # Trained models
├── notebooks/            # Jupyter notebooks for exploration
└── docs/                 # Learning documentation
```

**Why this structure?**
- **Separation of concerns**: Each module does one thing well
- **Testability**: We can test each component independently
- **Organizer requirements**: They need specific files (submission.py, weights/, notebooks/)

### 2. Configuration Centralization

All hyperparameters live in `config.py`. This is important because:
- When experimenting, you only change one file
- Results are reproducible (save the config with your runs)
- No "magic numbers" scattered through the code

Key parameters we set:

| Parameter | Value | Why |
|-----------|-------|-----|
| SAMPLE_RATE | 22050 Hz | Standard for music, half CD quality |
| DURATION | 5.0 sec | Long enough for musical structure |
| N_MELS | 128 | Good frequency resolution |
| N_FFT | 2048 | ~93ms window for harmonics |
| BATCH_SIZE | 64 | Fits in 8GB GPU, enough negatives |
| TEMPERATURE | 0.1 | SimCLR recommended value |

### 3. Dependencies

The requirements.txt includes:
- **torch, torchaudio**: Deep learning framework
- **librosa**: Audio analysis (for mel spectrograms)
- **faiss-cpu**: Fast similarity search
- **scikit-learn**: For linear probe evaluation
- **matplotlib, umap-learn**: Visualization

---

## Setup Steps

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

Virtual environments isolate project dependencies. Important because:
- Different projects might need different PyTorch versions
- Avoids polluting system Python
- Makes reproduction easier

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Verify GPU
```powershell
python scripts\verify_gpu.py
```

The script checks:
1. **PyTorch version**: Need 2.0+ for modern features
2. **CUDA availability**: Training on CPU would take hours instead of minutes
3. **GPU memory**: Need at least 4GB, 8GB preferred
4. **Package imports**: Catch missing dependencies early

---

## Problems Encountered

### Problem 1: [To be filled in during execution]

**Error:**
```
[paste error here]
```

**Root Cause:**
[explanation]

**Solution:**
[what fixed it]

---

## Key Takeaways

1. **Always verify GPU first** - Don't waste hours training on CPU by accident
2. **Centralize configuration** - Makes experiments reproducible
3. **Structure for the end goal** - We knew the submission requirements, so we designed around them
4. **Virtual environments** - Essential for ML projects with complex dependencies

---

## Next Steps

Once GPU verification passes:
1. Download FMA-Small dataset (7.2 GB)
2. Extract to `data/fma_small/`
3. Proceed to Phase 1: Input Pipeline

---

## Dataset Download Instructions

The FMA (Free Music Archive) dataset:
- URL: https://github.com/mdeff/fma
- We need: fma_small.zip (7.2 GB)
- Contains: 8,000 tracks, 30 seconds each, 8 genres

Download options:
1. Direct download from GitHub releases
2. Kaggle: https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium

After download:
```powershell
# Unzip to data folder
# Final structure should be:
# data/fma_small/000/000002.mp3
# data/fma_small/000/000005.mp3
# ... (8000 total mp3 files)
```
