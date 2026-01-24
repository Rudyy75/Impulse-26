
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from src.config import DATA_DIR, SPECTROGRAM_DIR
from src.spectrogram import GPUPipeline

def main():
    print("----------------------------------------------------------------")
    print(" ⚡ PHASE 1 REVISITED: GPU Preprocessing (Caching)")
    print("----------------------------------------------------------------")
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    pipeline = GPUPipeline(device=device)
    
    # 2. Find Files
    files = glob.glob(os.path.join(DATA_DIR, '**/*.mp3'), recursive=True)
    print(f"Found {len(files)} MP3s. Processing...")
    
    # 3. Process Loop
    # We process one by one but fully on GPU.
    # Batching (e.g. 32 files) would be faster but more complex logic for variable lengths.
    
    success_count = 0
    
    # Disable autograd for inference/preprocessing
    with torch.no_grad():
        for file_path in tqdm(files):
            try:
                # Output Name
                file_name = os.path.basename(file_path).replace('.mp3', '.pt')
                save_path = os.path.join(SPECTROGRAM_DIR, file_name)
                
                if os.path.exists(save_path):
                    continue
                
                # Check MP3 integrity
                try:
                    y, sr = sf.read(file_path, dtype='float32')
                except Exception:
                    continue
                    
                if len(y.shape) > 1: y = np.mean(y, axis=1)
                
                # To GPU
                waveform = torch.from_numpy(y).unsqueeze(0).to(device)
                
                # MelSpec
                # We save the NORMALIZED mel spec (-1, 1) to be ready for model
                # Note: GPUPipeline usually does (mel + 80)/80 * 2 - 1.
                # So we are saving "Ready to Train" tensors.
                mels = pipeline(waveform, original_sr=sr) # (1, 1, 128, T)
                
                # Save as FP16 to save space (4x smaller than Wav usually)
                torch.save(mels.half().cpu(), save_path)
                
                success_count += 1
                
            except Exception as e:
                # print(f"Error {file_path}: {e}")
                pass
                
    print(f"✅ Cached {success_count} new spectrograms to {SPECTROGRAM_DIR}")

if __name__ == "__main__":
    main()
