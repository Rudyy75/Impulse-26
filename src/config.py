import os

# -----------------------------------------------------------------------------
# Audio Processing Parameters
# -----------------------------------------------------------------------------
SAMPLE_RATE = 22050     # Hz, sufficient for music retrieval
DURATION = 5.0          # Seconds per clip
N_SAMPLES = int(SAMPLE_RATE * DURATION) # 110,250 samples

# Spectrogram Parameters
N_FFT = 2048            # Window size (~93ms)
HOP_LENGTH = 512        # Step size (~23ms, 75% overlap)
N_MELS = 128            # Number of frequency bins (Mel scale)

# -----------------------------------------------------------------------------
# Model & Training Parameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4    # Standard for Adam optimizer
TEMPERATURE = 0.1       # NT-Xent loss temperature

# Dimensions
EMBEDDING_DIM = 512     # Output of Encoder (h)
PROJECTION_DIM = 128    # Output of Projection Head (z)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Automatically detect project root relative to this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'fma_small')
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')

# Create weights directory if it doesn't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
