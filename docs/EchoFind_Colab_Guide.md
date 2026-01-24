# 🚀 EchoFind: Google Colab Pro Setup Guide

Here is the exact code to copy-paste into your Google Colab cells. This configuration is optimized for **Colab Pro (T4/A100 GPU)** with **ResNet50** and **Mixed Precision**.

---

## cell 01: Setup & Data Download
*Installs dependencies and downloads the FMA Small dataset (8GB).*

```python
# 1. Install Dependencies
!pip install -q torchaudio librosa soundfile pandas seaborn scikit-learn

# 2. Download Data
import os

DATA_ROOT = './data'
os.makedirs(DATA_ROOT, exist_ok=True)

# FMA URLs
FMA_SMALL_URL = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
FMA_META_URL = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'

def download_and_extract(url, target_dir):
    filename = url.split('/')[-1]
    path = os.path.join(target_dir, filename)
    
    # 1. Download
    if not os.path.exists(path):
        print(f'⬇️ Downloading {filename}...')
        !wget --progress=bar:force -P {target_dir} {url}
    else:
        print(f'✅ {filename} already downloaded.')
        
    # 2. Extract (System Unzip is FASTER)
    # -q is used because 8000 print statements will freeze the browser
    print(f'📦 Extracting {filename} (Please wait ~60s)...')
    !unzip -q -n {path} -d {target_dir}
    print(f'✅ Extracted {filename}')

download_and_extract(FMA_SMALL_URL, DATA_ROOT)
download_and_extract(FMA_META_URL, DATA_ROOT)
print("✅ Data Ready: ./data/fma_small")
```

---

## Cell 02: Imports & Configuration
*Sets up ResNet50, 200 Epochs, and GPU settings.*

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast # Mixed Precision

import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import glob
import time
import torchaudio.transforms as T
import random
from tqdm.notebook import tqdm

class Config:
    # Audio
    SAMPLE_RATE = 22050
    DURATION = 3.0
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    
    # Training (Colab Pro Specs)
    BATCH_SIZE = 128         # T4 safe. Try 256 for A100.
    EPOCHS = 200             # Deep training
    LEARNING_RATE = 3e-4
    TEMPERATURE = 0.5
    WEIGHT_DECAY = 1e-4
    EMBEDDING_DIM = 2048     # ResNet50 Output
    PROJECTION_DIM = 128
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_DIR = './data/fma_small'
    CHECKPOINT_DIR = './checkpoints'

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
print(f"🔥 Device: {Config.DEVICE}")
```

---

## Cell 03: Augmentations
*Implements the "Goldilocks" Protocol (Noise + Mixup).*

```python
class GPUAugmentations(nn.Module):
    def __init__(self):
        super().__init__()
        # GPU-Accelerated Augmentations
        
    def forward(self, spec):
        # Spec: (B, 1, F, T)
        if self.training:
            # 1. Time Masking
            if random.random() < 0.5:
                T = spec.shape[-1]
                mask_size = int(T * 0.1)
                t0 = random.randint(0, T - mask_size)
                spec[:, :, :, t0:t0+mask_size] = 0
                
            # 2. Freq Masking
            if random.random() < 0.5:
                F = spec.shape[-2]
                mask_size = int(F * 0.1)
                f0 = random.randint(0, F - mask_size)
                spec[:, :, f0:f0+mask_size, :] = 0
                
            # 3. Gaussian Noise (on Spectrogram)
            if random.random() < 0.5:
                noise = torch.randn_like(spec) * 0.05
                spec = spec + noise
                
        return spec
```

---

## Cell 04: Dataset
*Scanning and processing the files.*

```python
class FMAContrastiveDataset(Dataset):
    def __init__(self, data_dir, duration=3.0, sample_rate=22050):
        self.files = glob.glob(os.path.join(data_dir, '**/*.mp3'), recursive=True)
        self.duration = duration
        self.sample_rate = sample_rate
        self.n_samples = int(sample_rate * duration)
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        # FAST LOADING (Torchaudio uses C++ underneath)
        try:
            path = self.files[idx]
            # normalize=True creates float32 in [-1, 1]
            y, sr = torchaudio.load(path, normalize=True) 
            
            # Resample IF needed (Slow, but torchaudio is faster than librosa)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                y = resampler(y)
                
            y = y[0] # Mono
            
            # Pad/Crop
            if y.shape[0] < self.n_samples:
                pad = self.n_samples - y.shape[0]
                y = F.pad(y, (0, pad))
            elif y.shape[0] > self.n_samples:
                start = random.randint(0, y.shape[0] - self.n_samples)
                y = y[start:start+self.n_samples]
                
            return y # Return Raw Waveform (T,)
            
        except Exception:
             return torch.zeros(self.n_samples)
```

---

## Cell 05: Model (ResNet50)
*Bigger backbone for Colab Pro.*

```python
class SimCLR(nn.Module):
    def __init__(self, base_model='resnet50', embedding_dim=2048, projection_dim=128):
        super().__init__()
        
        # ResNet50 Backbone
        self.encoder = models.resnet50(weights=None)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        prev_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
            
        # Projection Head
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.ReLU(),
            nn.Linear(prev_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, -1e4) # -9e15 overflows FP16. -1e4 is safe & sufficient.
    labels = torch.cat([
        torch.arange(batch_size, device=z.device) + batch_size,
        torch.arange(batch_size, device=z.device)
    ], dim=0)
    return F.cross_entropy(sim_matrix, labels)
```

---

## Cell 06: Training Execution
*Mixed Precision Training Loop.*

```python
def train():
    # 1. GPU Transforms
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    ).to(Config.DEVICE)
    
    amp_to_db = T.AmplitudeToDB().to(Config.DEVICE)
    augmenter = GPUAugmentations().to(Config.DEVICE)
    
    # 2. Pipeline
    dataset = FMAContrastiveDataset(Config.DATA_DIR)
    dataloader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True, drop_last=True
    )
    
    model = SimCLR(base_model='resnet50').to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # AMP Scaler (Handle Torch Update)
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda') 
    else:
        scaler = GradScaler()
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    print(f'🚀 Starting Training: {Config.EPOCHS} Epochs | Batch: {Config.BATCH_SIZE}')
    
    loss_history = []
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
        
        for waveforms in pbar:
            # waveforms: (B, T) - Raw Audio
            waveforms = waveforms.to(Config.DEVICE)
            
            # --- GPU PROCESSING ---
            with torch.no_grad():
                # 1. Compute MelSpec (B, 128, T_frames)
                mels = mel_transform(waveforms)
                mels = amp_to_db(mels)
                
                # 2. Normalize [-1, 1] (Approx)
                # Global norm is faster than per-sample
                mels = (mels + 80) / 80 # FMA is roughly -80dB to 0dB
                mels = mels.unsqueeze(1) # Add Channel: (B, 1, F, T)
                
                # 3. Augment TWO Views
                spec1 = augmenter(mels.clone())
                spec2 = augmenter(mels.clone())
            
            # --- MODEL FORWARD ---
            with autocast(): # Mixed Precision
                _, z1 = model(spec1)
                _, z2 = model(spec2)
                loss = nt_xent_loss(z1, z2, temperature=Config.TEMPERATURE)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1} | Loss: {avg_loss:.4f}')
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'{Config.CHECKPOINT_DIR}/resnet50_e{epoch+1}.pt')
            
    return model

# START
model = train()
```
