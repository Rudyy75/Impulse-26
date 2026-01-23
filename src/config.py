"""
Configuration file for EchoFind.

All hyperparameters and paths are centralized here. This makes it easy to:
1. Experiment with different settings
2. Reproduce results
3. Understand the model configuration at a glance
"""

import os
from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================

# Get the project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset paths
DATA_DIR = PROJECT_ROOT / "data"
FMA_DIR = DATA_DIR / "fma_small"

# Output paths
WEIGHTS_DIR = PROJECT_ROOT / "weights"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure output directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# AUDIO PARAMETERS
# ==============================================================================

# Sample rate: 22050 Hz is standard for music (half of CD quality 44100)
# This reduces computation while preserving relevant frequency content
SAMPLE_RATE = 22050

# Duration in seconds - we crop or pad all audio to this length
# 5 seconds captures enough musical structure while keeping batches manageable
DURATION = 5.0

# Total number of samples per audio clip
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 110250 samples

# ==============================================================================
# SPECTROGRAM PARAMETERS
# ==============================================================================

# FFT window size: 2048 samples = ~93ms at 22050 Hz
# This captures harmonic structure well for music
N_FFT = 2048

# Hop length: 512 samples = ~23ms
# Overlap = 1 - (512/2048) = 75%, good for smooth spectrograms
HOP_LENGTH = 512

# Number of mel bands: 128 gives good frequency resolution
# without excessive computation
N_MELS = 128

# Frequency range for mel filterbank
F_MIN = 20      # Human hearing lower limit
F_MAX = 8000    # Most musical content below this

# The output spectrogram shape will be:
# (1, N_MELS, TIME_FRAMES) = (1, 128, ceil(110250/512)) = (1, 128, 216)

# ==============================================================================
# MODEL PARAMETERS
# ==============================================================================

# Embedding dimension from encoder (before projection)
# 512 is standard for ResNet-18 final layer
EMBEDDING_DIM = 512

# Projection head output dimension
# 128 is common for SimCLR - small enough for efficient loss computation
PROJECTION_DIM = 128

# ==============================================================================
# TRAINING PARAMETERS
# ==============================================================================

# Batch size: 64 fits comfortably in 8GB GPU memory
# Larger batches give more negatives per positive (good for contrastive learning)
BATCH_SIZE = 64

# Number of training epochs (minimum 50 per problem statement)
EPOCHS = 50

# Learning rate: 3e-4 is Adam default, works well empirically
LEARNING_RATE = 3e-4

# Temperature for NT-Xent loss
# Lower = model treats negatives more harshly
# 0.1 is recommended in SimCLR paper
TEMPERATURE = 0.1

# Weight decay for regularization (light)
WEIGHT_DECAY = 1e-6

# Number of dataloader workers
# Set based on your CPU cores, 4 is safe default
NUM_WORKERS = 4

# ==============================================================================
# AUGMENTATION PARAMETERS
# ==============================================================================

# Time masking: how many consecutive time frames to mask
TIME_MASK_MAX = 30

# Frequency masking: how many consecutive mel bands to mask
FREQ_MASK_MAX = 20

# Gaussian noise standard deviation range
NOISE_STD_MIN = 0.01
NOISE_STD_MAX = 0.05

# Time shift: maximum fraction of duration to shift
TIME_SHIFT_MAX = 0.2

# Gain (volume) scaling range
GAIN_MIN = 0.8
GAIN_MAX = 1.2

# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================

import torch

# Automatically use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
