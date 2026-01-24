
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

class GPUPipeline(nn.Module):
    """
    Centralized GPU Pipeline for Audio Preprocessing.
    Handles: Resampling -> MelSpectrogram -> AmplitudeToDB -> Normalization.
    Safe for Mixed Precision and handles NaNs/Silence.
    """
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, device='cuda'):
        super().__init__()
        self.device = device
        self.target_sr = sample_rate
        
        # Resamplers (Lazy init could be better, but fixed usually fine)
        # We assume input is 44.1k (Standard FMA)
        self.resampler_44k = T.Resample(44100, sample_rate).to(device)
        
        # Spectrogram
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ).to(device)
        
        self.amp_to_db = T.AmplitudeToDB().to(device)

    def forward(self, waveforms, original_sr=44100):
        # waveforms: (B, T)
        
        # 1. Resample
        if original_sr != self.target_sr:
            if original_sr == 44100:
                waveforms = self.resampler_44k(waveforms)
            else:
                # Fallback: Create dynamic resampler (Slow)
                resampler = T.Resample(original_sr, self.target_sr).to(self.device)
                waveforms = resampler(waveforms)
                
        # 2. MelSpec
        mels = self.melspec(waveforms)
        mels = self.amp_to_db(mels)
        
        # 3. Safety (NaN/Inf)
        mels = torch.nan_to_num(mels, nan=-80.0, posinf=0.0, neginf=-80.0)
        mels = torch.clamp(mels, min=-100.0, max=100.0)
        
        # 4. Normalize (Global Scale approx for FMA)
        # -80dB to 0dB -> [-1, 1]
        mels = (mels + 80) / 80
        mels = mels.unsqueeze(1) # (B, 1, F, T)
        mels = mels * 2 - 1
        
        return mels
