
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import soundfile as sf
import random
import torchaudio
import torchaudio.transforms as T

# Imports
try:
    from config import *
    from model import EchoFindModel
except ImportError:
    from src.config import *
    from src.model import EchoFindModel

# ------------------------------------------------------------------------------
# 1. GPU Pipeline Components
# ------------------------------------------------------------------------------
class GPUPipeline(nn.Module):
    def __init__(self, target_sr=22050, device='cuda'):
        super().__init__()
        self.device = device
        self.target_sr = target_sr
        self.resampler_44k = T.Resample(44100, target_sr).to(device)
        self.melspec = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        ).to(device)
        self.amp_to_db = T.AmplitudeToDB().to(device)

    def forward(self, waveforms, original_sr=44100):
        if original_sr != self.target_sr:
            if original_sr == 44100:
                waveforms = self.resampler_44k(waveforms)
            else:
                resampler = T.Resample(original_sr, self.target_sr).to(self.device)
                waveforms = resampler(waveforms)
        mels = self.melspec(waveforms)
        mels = self.amp_to_db(mels)
        # FIX: Handle Silence/-Inf/NaN
        mels = torch.nan_to_num(mels, nan=-80.0, posinf=0.0, neginf=-80.0)
        mels = torch.clamp(mels, min=-100.0, max=100.0)
        mels = (mels + 80) / 80
        mels = mels.unsqueeze(1) 
        mels = mels * 2 - 1
        return mels

# ------------------------------------------------------------------------------
# 2. Dataset 
# ------------------------------------------------------------------------------
class FMAClassificationDataset(Dataset):
    def __init__(self, file_paths, labels, target_len_seconds=3.0, expected_sr=44100):
        self.files = file_paths
        self.labels = labels
        self.expected_sr = expected_sr
        self.crop_samples = int(expected_sr * target_len_seconds)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        try:
            y, sr = sf.read(path, dtype='float32') 
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            if len(y) < self.crop_samples:
                y = np.pad(y, (0, self.crop_samples - len(y)))
            elif len(y) > self.crop_samples:
                start = random.randint(0, len(y) - self.crop_samples) 
                y = y[start:start+self.crop_samples]
            return torch.from_numpy(y), torch.tensor(label).long(), sr
        except Exception:
            return torch.zeros(self.crop_samples), torch.tensor(label).long(), 44100

# ------------------------------------------------------------------------------
# 3. Helpers (MOVED UP)
# ------------------------------------------------------------------------------
def get_supervised_data():
    if not os.path.exists(os.path.join(METADATA_DIR, 'tracks.csv')): return None
    tracks = pd.read_csv(os.path.join(METADATA_DIR, 'tracks.csv'), index_col=0, header=[0, 1])
    small = tracks[tracks[('set', 'subset')] == 'small']
    genres = small[('track', 'genre_top')]
    
    file_paths = []
    labels = []
    all_files = glob.glob(os.path.join(DATA_DIR, '**/*.mp3'), recursive=True)
    id_to_path = {int(os.path.splitext(os.path.basename(f))[0]): f for f in all_files if os.path.splitext(os.path.basename(f))[0].isdigit()}
            
    for tid, genre in genres.items():
        if tid in id_to_path and isinstance(genre, str):
            file_paths.append(id_to_path[tid])
            labels.append(genre)
            
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    X_train, X_val, y_train, y_val = train_test_split(file_paths, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    return X_train, X_val, y_train, y_val, le

# ------------------------------------------------------------------------------
# 4. Training Loop
# ------------------------------------------------------------------------------
def run_finetuning():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {DEVICE}")
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 1e-4 
    
    data = get_supervised_data()
    if not data: return
    X_train, X_val, y_train, y_val, le = data
    
    gpu_prep = GPUPipeline(target_sr=22050, device=DEVICE)
    train_ds = FMAClassificationDataset(X_train, y_train)
    val_ds = FMAClassificationDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("🧠 Loading Model...")
    model = EchoFindModel(embedding_dim=128).to(DEVICE)
    weights = glob.glob(os.path.join(WEIGHTS_DIR, '*.pt'))
    if weights:
        latest = max(weights, key=os.path.getctime)
        print(f"⬇️ Loading Pretrained Weights: {latest}")
        try:
            checkpoint = torch.load(latest)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except:
             print("⚠️ Params mismatch. Learning from scratch.")
    else:
        print("⚠️ No weights found. Training from Scratch.")
    
    encoder = model.backbone
    encoder.fc = nn.Identity()
    classifier = nn.Sequential(
        encoder,
        nn.Flatten(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(le.classes_))
    ).to(DEVICE)
    
    optimizer = optim.Adam(classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 
    
    print(f"🚀 Starting Fine-Tuning (GPU Accelerated)...")
    
    for epoch in range(EPOCHS):
        classifier.train()
        correct = 0
        total = 0
        total_loss = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y, sr in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.cuda.amp.autocast():
                mels = gpu_prep(x, original_sr=44100)
                logits = classifier(mels)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({'acc': f"{correct/total*100:.1f}%", 'loss': f"{loss.item():.4f}"})
            
        train_acc = correct / total
        
        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y, sr in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                mels = gpu_prep(x, original_sr=44100)
                logits = classifier(mels)
                _, pred = torch.max(logits, 1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        print(f"🏆 Epoch {epoch+1} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}%")
        
        # Save Last Model
        save_path = os.path.join(WEIGHTS_DIR, 'finetune_best.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'acc': val_acc
        }, save_path)
        print(f"💾 Saved checkpoint to {save_path}")

if __name__ == "__main__":
    run_finetuning()
