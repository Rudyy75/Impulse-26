import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import random
from .config import SAMPLE_RATE, N_SAMPLES, N_FFT, HOP_LENGTH, N_MELS

def load_audio(audio_path):
    """
    Loads an audio file and processes it into a fixed-length mono waveform.
    
    1. Load audio
    2. Resample to 22050 Hz
    3. Convert to Mono
    4. Randomly crop or pad to exactly 5 seconds
    """
    try:
        # 1. Load Audio
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        # Return a silent tensor of correct shape as fallback
        return torch.zeros(1, N_SAMPLES)

    # 2. Resample if necessary
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # 3. Convert to Mono (average across channels)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 4. Handle Length (Crop or Pad)
    current_samples = waveform.shape[1]

    if current_samples > N_SAMPLES:
        # Random Crop
        start_idx = random.randint(0, current_samples - N_SAMPLES)
        waveform = waveform[:, start_idx : start_idx + N_SAMPLES]
    elif current_samples < N_SAMPLES:
        # Zero Pad
        padding = N_SAMPLES - current_samples
        waveform = F.pad(waveform, (0, padding))
    
    # Ensure shape is (1, N_SAMPLES)
    return waveform

def audio_to_melspec(waveform):
    """
    Converts a waveform to a Log-Mel Spectrogram.
    """
    # Define Transform
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        normalized=True 
    )

    # Apply Transform
    mel_spec = mel_transform(waveform)

    # Log-Mel Spectrogram (add epsilon to avoid log(0))
    log_mel_spec = torch.log(mel_spec + 1e-9)

    return log_mel_spec
