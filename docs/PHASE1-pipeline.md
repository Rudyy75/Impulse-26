# Phase 1: Input Pipeline - Masterclass Documentation

> **Goal:** Transform raw physical sound waves into a format a Neural Network can "see" and "understand".
> **Outcome:** A robust pipeline generating augmented Mel-Spectrogram pairs.
> **Scope:** Physics of Sound $\to$ Digital Signal Processing $\to$ PyTorch Tensors.

---

# 1. 🎵 Theory of Digital Audio: The Physics

Before writing a single line of code, we must understand the physical phenomenon we are modeling.

### 1.1 The Waveform
Sound is a continuous fluctuation of air pressure. When a drum is hit, it compresses air molecules. This compression wave travels to your ear, vibrating the eardrum.
- **Amplitude:** The intensity of the pressure difference. Perceived as **Loudness**.
- **Time:** The duration of the event. (X-axis).
- **Frequency:** The speed of vibration. Perceived as **Pitch**.

In the digital world, we cannot store continuous infinite pressure values. We must **Sample** them.

### 1.2 Sampling Rate & The Nyquist-Shannon Theorem
This is the single most important theorem in Digital Signal Processing (DSP).

**The Theorem:**
> "To perfectly reconstruct a continuous signal of bandwidth $B$, one must sample it at a rate $f_s > 2B$."

**Application to Music:**
1.  **Human Hearing Range:** A generic healthy human ear hears from **20 Hz to 20,000 Hz (20 kHz)**.
2.  **The Math:** If $f_{max} = 20\text{kHz}$, then we need $f_s > 40\text{kHz}$.
3.  **CD Standard:** This is why CDs utilize **44,100 Hz**. It allows capturing up to 22.05 kHz, providing a slight safety buffer for "Anti-Aliasing Filters".

**Why did we choose 22,050 Hz?**
We are building a Machine Learning model, not a Hi-Fi speaker system.
- **Computation Cost:** An audio file at 44.1kHz has 2x pixels compared to 22.05kHz. This means 2x memory usage and 2x slower training.
- **Information Density:** Most "musical identity" (Melody, Chord Progression, Rhythm, Vibe) lives below 10kHz.
    - **Sub-Bass:** 20-60 Hz (Felt, not heard).
    - **Bass:** 60-250 Hz (Basslines).
    - **Mids:** 250-2000 Hz (Vocals, Guitars, Keys).
    - **High Mids:** 2k-6k Hz (Attack, Clarity).
    - **Highs:** 6k-20k Hz (Air, Hiss, Cymbals).
- **The Trade-off:** By cutting at 11kHz (Nyquist of 22.05k), we lose the "Air" and "Sparkle", but we keep 100% of the musical note information. This is an acceptable trade-off for 2x speed.

```ascii
      Amplitude
         ^
         |      *       *       *       *
         |    *   *   *   *   *   *   *   *     <-- Discrete Samples
         |   *     * *     * *     * *     *
       --+--*-------*-------*-------*-------*---> Time
```

### 1.3 The Fourier Transform (Time $\to$ Frequency)
A waveform ($x[t]$) shows amplitude over time. This is **useless** for recognition.
- **The Phase Problem:** If you shift a song by 0.1s, the waveform looks completely different (peaks become troughs).
- **The Solution:** We need **Frequency**. A "C Major Chord" is always C(261Hz)-E(329Hz)-G(392Hz) frequencies, regardless of *when* it is played.

#### FFT (Fast Fourier Transform)
The FFT algorithm decomposes a signal of $N$ samples into $N/2$ frequency bins.
- It asks: "How much of 1Hz is in here? How much of 2Hz? ... How much of 10kHz?"
- **The Math:** $X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N}$.

#### STFT (Short-Time Fourier Transform)
A song changes over time. An FFT of the *entire* song is just a mush of all notes played at once. We need "Time-Frequency" resolution.
1.  **Windowing:** We assume the signal is stationary for very short periods (~100ms).
2.  **The Window Function (Hann):** If we create sharp cuts, we introduce "Spectral Leakage" (artificial clicking sounds). We multiply our audio chunk by a Bell Curve (Hann Window) to smooth the edges to zero.
3.  **Frame Size (N_FFT):** We chose **2048 samples**.
    - At 22050Hz, 2048 samples = $2048/22050 \approx 0.092$ seconds (93ms).
    - This gives us nice frequency resolution ($22050 / 2048 \approx 10$ Hz per bin).
4.  **Hop Length:** How much we slide the window.
    - We chose **512 samples** (~23ms).
    - Since Window is 2048, we overlap 75%. This creates a smooth, continuous image without gaps.

### 1.4 The Mel Scale & Logarithms (The "Human" Transform)
Raw FFT buckets (Linear Frequency) are biologically incorrect.
1.  **Mel Scale:** 
    - To a computer: 100Hz vs 200Hz (Difference 100) is the same as 10,000Hz vs 10,100Hz (Difference 100).
    - To a human: 100Hz to 200Hz is an **Octave** (Huge change). 10,000 to 10,100 is indistinguishable.
    - **The Solution:** We warp the Y-axis. We allocate MORE pixels to low frequencies and FEWER pixels to high frequencies.
    - **N_MELS:** We use **128 bins**. This compresses the 1025 FFT bins into 128 meaningful perceptual bands.

2.  **Log Amplitude (Decibels):** 
    - Sound energy follows a Power Law. An explosion is billions of times more powerful than a whisper.
    - Neural Networks handle linear ranges like $[0, 1]$ or $[-1, 1]$. They CANNOT handle $[0, 10^9]$.
    - **The Solution:** We apply Logarithm: $S_{db} = 10 \log_{10}(S_{power})$.
    - This squashes the dynamic range into something manageable (e.g., -80dB to 0dB).

---

# 2. 💾 Code Analysis: `src/config.py`

The configuration file is the "Single Source of Truth". It ensures consistency across training, inference, and testing.

```python
import os

# -----------------------------------------------------------------------------
# Audio Processing Parameters
# -----------------------------------------------------------------------------

# SAMPLE_RATE = 22050
# Decision: Standard for high-performance audio ML.
# Trade-off: Lost >11kHz content vs 50% Memory Savings.
# This decision was made after initial experiments at 44.1kHz showed no accuracy gain.
SAMPLE_RATE = 22050

# DURATION = 5.0
# Decision: How much context does the model need to ID a song?
# 5 seconds is standard in literature (e.g. FMA paper).
# Too short (<1s): Rhythm/Tempo is lost.
# Too long (>10s): Batch size must decrease, gradients become noisy.
DURATION = 5.0          

# N_SAMPLES = 110,250
# Calculated automatically to avoid rounding errors later.
N_SAMPLES = int(SAMPLE_RATE * DURATION) 

# Spectrogram Parameters
# N_FFT = 2048.
# Why? Powers of 2 (1024, 2048, 4096) are optimized for FFT algorithms (Radix-2).
# 2048 gives us 1025 frequency bins.
N_FFT = 2048            

# HOP_LENGTH = 512.
# 25% of Window Length.
# This 4x oversampling ensures we don't miss transient events (like a snare drum) that might fall on a window edge.
HOP_LENGTH = 512        

# N_MELS = 128.
# Industry standard for ResNet-18 inputs.
# Resulting Image Height = 128 pixels.
N_MELS = 128            

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# BATCH_SIZE = 64
# SimCLR relies on "Negative Samples".
# In a batch of 64, for every 1 positive pair, there are 2*(64-1) = 126 negatives.
# Larger Batch = More Negatives = Harder Task = Better Features.
# 64 is the max that fits on a standard 8GB/12GB GPU with ResNet.
BATCH_SIZE = 64

# EPOCHS = 50
# Self-Supervised Learning converges slower than Supervised.
# The loss curve usually plateaus around 40-50 epochs on small datasets (FMA Small).
EPOCHS = 50

# LEARNING_RATE = 3e-4
# Since we use Adam optimizer, 3e-4 (Karpathy Constant) is the safest starting point.
LEARNING_RATE = 3e-4    

# TEMPERATURE = 0.1
# The scaling factor in NT-Xent loss.
# Low Temp (0.1) -> Makes the Softmax distribution "Spiky". Even small differences in similarity result in huge loss penalties.
# High Temp (1.0) -> Makes it "Flat". Model gets lazy.
TEMPERATURE = 0.1       
```

---

# 3. 💾 Code Analysis: `src/spectrogram.py`

This module abstracts the `torchaudio` complexity. It converts an MP3 file on disk into a Tensor in VRAM.

```python
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import random
from .config import SAMPLE_RATE, N_SAMPLES, N_FFT, HOP_LENGTH, N_MELS

def load_audio(audio_path):
    """
    Robust Audio Loader.
    Handles:
    1. Unknown Sample Rates
    2. Stereo vs Mono
    3. Variable Lengths (Padding/Cropping)
    4. Corrupt Files
    """
    try:
        # Load Raw Audio
        # default backend (soundfile/sox) reads the header info first.
        # waveform shape: [Channels, Time]
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        # Fail Gracefully.
        # Returning a zero-tensor allows the dataloader to continue without crashing the whole training loop.
        # (We filter these silent tensors out later or trained model ignores them).
        return torch.zeros(1, N_SAMPLES)

    # 1. Resampling
    # This is computationally expensive.
    # T.Resample uses filter banks to prevent aliasing (ghost frequencies).
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # 2. Channel Reduction (Mono)
    # Music retrieval relies on Harmonic/Timbral content, not Spatial (Stereo) content.
    # Averaging channels reduces input size by 50% without meaningful information loss.
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 3. Fixed Length Enforcement
    # CNNs (ResNet) require fixed rectangle inputs (e.g. 128x216).
    # We enforce 5 seconds exactly.
    current_samples = waveform.shape[1]

    if current_samples > N_SAMPLES:
        # Case A: Too Long (Normal Song)
        # We pick a *Random* 5-second window.
        # Why Random? 
        # Feature: Ideally, the model should recognize the song from the Intro, Chorus, OR Bridge.
        # During 50 epochs, the model will likely see 50 different parts of the same song.
        start_idx = random.randint(0, current_samples - N_SAMPLES)
        waveform = waveform[:, start_idx : start_idx + N_SAMPLES]
    elif current_samples < N_SAMPLES:
        # Case B: Too Short (Sound Effect / Glitch)
        # We must Pad with Zeros (Silence) to reach required length.
        # 'padding' argument format: (left, right, top, bottom)
        padding = N_SAMPLES - current_samples
        waveform = F.pad(waveform, (0, padding))
    
    return waveform

def audio_to_melspec(waveform):
    """
    The Mathematical Transformation Chain.
    Waveform -> STFT -> Complex Norm -> Mel Scale -> Logarithm.
    """
    # 1. define the MelSpectrogram transform
    # Torchaudio optimizes this implementation in C++ under the hood.
    # normalized=True ensures the area under triangle filters sums to 1 (energy preservation).
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,        
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS,      
        normalized=True 
    )

    # 2. Execute Transform
    # Spectrogram Shape: [Channels, N_Mels, Time_Frames]
    # Time_Frames = (N_Samples / Hop_Length) + 1
    # 110250 / 512 = 215.33 -> 216 frames.
    # Final Shape: [1, 128, 216]
    mel_spec = mel_transform(waveform)

    # 3. Dynamic Range Compression (Log Scale)
    # Raw Spectrogram values are Power/Energy magnitudes (x^2).
    # They range from 0.000001 (silence) to 10000.0 (kick drum).
    # Logarithm converts multiplication to addition: log(A*B) = log(A) + log(B).
    # This aligns with the "Weber-Fechner Law" of human perception.
    # '1e-9' is added to avoid computing log(0), which is mathematically undefined (-inf).
    log_mel_spec = torch.log(mel_spec + 1e-9)

    return log_mel_spec
```

---

# 4. 💾 Code Analysis: `src/augmentations.py`

This is the most critical component for **Phase 3 Success**. 
In Self-Supervised Learning, data augmentation is not just "extra data" — it **IS** the label. We define "Similarity" by saying "A song is similar to a noisy version of itself."

If our augmentations are weak, the model learns trivial features (e.g., exact volume).
If our augmentations are strong (Acid Test), the model learns deep semantic features (e.g., chord progression).

```python
import torch
import random
import torch.nn.functional as F

class AugmentationPipeline:
    def __init__(self, time_mask_param=30, freq_mask_param=20, noise_std=0.03):
        # Parameters define the "Maximum Severity" of the distortion.
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        # Noise Standard Deviation. 
        # 0.03 is equivalent to roughly ~25dB SNR (Mild noise).
        # We upgraded this in Phase 4 to handle street noise (5dB SNR).
        self.noise_std = noise_std 

    def add_noise(self, spec):
        """
        Simulation of Thermal/Background Noise.
        Adds random values from Normal Distribution N(0, 1).
        
        Effect on Model:
        Forces the model to ignore low-amplitude details (hiss) and focus on 
        high-energy distinct features (Melody/Beats).
        """
        # torch.randn_like generates a tensor of same shape filled with random numbers.
        # We scale it by noise_std.
        noise = torch.randn_like(spec) * self.noise_std
        return spec + noise

    def time_mask(self, spec):
        """
        Simulation of Signal Dropouts / Glitches.
        Technique from Google's 'SpecAugment' paper (2019).
        
        Effect on Model:
        Forces "Temporal Consistency". 
        "If I miss the middle 0.5s of the song, can I still identify it?"
        The model learns to predict the missing context from surrounding frames.
        """
        _, _, time_steps = spec.shape
        # We choose a random length for the dropout
        mask_len = random.randint(0, self.time_mask_param)
        # We choose a random start position
        start = random.randint(0, time_steps - mask_len)
        
        # Clone is essential! Python passes objects by reference.
        # If we don't clone, we overwrite the cache and ruin the dataset.
        augmented_spec = spec.clone()
        augmented_spec[:, :, start : start + mask_len] = 0
        return augmented_spec

    def freq_mask(self, spec):
        """
        Simulation of Band-Pass Filters / Bad Microphones.
        Masks out horizontal strips (frequency bands).
        
        Effect on Model:
        Forces "Timbral Invariance".
        "Can I identify this song if the Bass is gone? Or if the Vocals are muffled?"
        """
        _, n_mels, _ = spec.shape
        mask_len = random.randint(0, self.freq_mask_param)
        start = random.randint(0, n_mels - mask_len)
        
        augmented_spec = spec.clone()
        augmented_spec[:, start : start + mask_len, :] = 0
        return augmented_spec
    
    def time_shift(self, spec, max_shift=20):
        """
        Simulation of Temporal Misalignment.
        Rolls the spectrogram along the X-axis.
        """
        shift = random.randint(-max_shift, max_shift)
        # torch.roll wraps elements around. 
        # Left-edge elements appear on Right-edge.
        return torch.roll(spec, shifts=shift, dims=2)

    def __call__(self, spec):
        """
        The Pipeline Execution.
        Stochastic Application: We don't apply every effect every time.
        We want a distribution of "Easy" samples and "Hard" samples.
        """
        out = spec.clone()

        # Probabilities tuned for Audio Retrieval.
        # Noise is most common in real world (80%).
        if random.random() < 0.8: out = self.add_noise(out)
        
        # SpecAugment parameters (50%)
        if random.random() < 0.5: out = self.time_mask(out)
        if random.random() < 0.5: out = self.freq_mask(out)
        
        # Shift (30%)
        if random.random() < 0.3: out = self.time_shift(out)

        return out
```

---

# 5. 💾 Code Analysis: `src/dataset.py`

This file orchestrates the data loading strategy. We chose an **Offline Preprocessing** strategy.

### 5.1 The Bottleneck Problem
Training a CNN is fast (GPU). Computing an FFT is slow (CPU).
If we load MP3s and compute Spectrograms inside the training loop:
- GPU Utilization: 10% (Starved).
- CPU Utilization: 100% (Bottlenecked).

### 5.2 The Pre-Computation Solution
We ran `preprocess_all.py` *before* training. This converted 8,000 `.mp3` files $\to$ 8,000 `.pt` tensors.
- **MP3 Load Time:** ~50ms.
- **FFT Time:** ~30ms.
- **Tensor Load Time:** ~2ms.
**Gain:** 40x speedup in data loading.

```python
class FMAContrastiveDataset(Dataset):
    """
    A PyTorch Dataset optimized for SimCLR (Contrastive Learning).
    Returns PAIRS of augmented views for a single song.
    """
    def __init__(self, data_dir, ext='.mp3'):
        # glob finds all files recursively.
        # We point this to the SPECTROGRAM_DIR, not raw audio.
        self.files = glob.glob(os.path.join(data_dir, '**', f'*{ext}'), recursive=True)
        self.augment = AugmentationPipeline()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Disk I/O (Fastest possible read)
        path = self.files[idx]
        
        # We load the full pre-computed spectrogram.
        # Usually shape [1, 128, 2000+] (Full song representation).
        spec = torch.load(path) 
        
        # 2. Random Crop (On-the-fly)
        # We need a fixed 5s window (216 frames).
        # We slice the tensor cheaply in RAM.
        current_frames = spec.shape[2]
        target_frames = 216
        
        if current_frames > target_frames:
            start = random.randint(0, current_frames - target_frames)
            spec = spec[:, :, start : start + target_frames]
        
        # 3. Contrastive View Generation
        # This is the core of SimCLR logic.
        # We take ONE source (Truth).
        # We create TWO distorted lies (Views).
        view1 = self.augment(spec)
        view2 = self.augment(spec)
        
        # Why two? 
        # Loss function tries to minimize distance(view1, view2).
        # Invariance: "Distortion A" and "Distortion B" came from same Source.
        
        return view1, view2
```

---

# 6. 📊 Architecture Diagram (ASCII)

```ascii
[ Disk Storage ]      [ RAM / CPU ]                     [ GPU VRAM ]
+-------------+      +-------------------+             +-------------------------+
|  001.pt     | ---> | Load Tensor       |             |                         |
|  (Full Song)|      | Shape: [128, 5000]|             |                         |
+-------------+      +-------------------+             |                         |
                              |                        |                         |
                              v                        |                         |
                     +-------------------+             |                         |
                     | Random Crop       |             |                         |
                     | [128, 216]        |             |                         |
                     +-------------------+             |                         |
                              |                        |                         |
                    +---------+---------+              |                         |
                    |                   |              |                         |
                    v                   v              |                         |
              +-----------+       +-----------+        |                         |
              | Augment A |       | Augment B |        |                         |
              | (Noise)   |       | (Masking) |        |                         |
              +-----------+       +-----------+        |                         |
                    |                   |              |                         |
                    +---------+---------+              |                         |
                              |                        |                         |
                              v                        |                         |
                       Batch Collation  -------------> | Batch [64, 2, 128, 216] |
                                                       +-------------------------+
```

---

# 7. Summary & Design Decisions

| Component | Choice | Reason | Alternative Considered |
|:---|:---|:---|:---|
| **Sample Rate** | 22.05 kHz | Balance of quality & speed | 44.1k (Too slow), 16k (Lossy) |
| **Transform** | Log-Mel Spectrogram | Mimics human hearing | MFCC (Decorrelated, but less info) |
| **Duration** | 5 seconds | Good context for Genre/ID | 30s (Too much VRAM), 1s (Too short) |
| **Augmentation** | Noise + Masking + Shift | Standard for Audio SimCLR | Pitch Shift (Computationally heavy) |
| **Dataset** | Offline Pre-computation | 40x Training Speedup | Online Calculation (Bottleneck) |

This pipeline represents a State-of-the-Art approach to Audio Self-Supervised Learning inputs, balancing theoretical correctness with engineering constraints (Memory/Speed).
