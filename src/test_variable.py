
import torch
import torch.nn.functional as F
import os
import sys
import soundfile as sf
import numpy as np
import glob
import random

# Add project root to path
sys.path.append(os.getcwd())

from src.inference import load_trained_model
from src.spectrogram import GPUPipeline
from src.config import DATA_DIR, N_SAMPLES

def test_variable_length():
    print("----------------------------------------------------------------")
    print(" 📏 PHASE 7: Variable-Length Robustness Test")
    print("----------------------------------------------------------------")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Model
    model = load_trained_model(device) # ResNet50 with GAP
    model.eval()
    
    pipeline = GPUPipeline(device=device)
    
    # 2. Get Test Files
    files = glob.glob(os.path.join(DATA_DIR, '**/*.mp3'), recursive=True)
    if len(files) < 5:
        print("❌ Not enough files.")
        return
        
    test_files = random.sample(files, 5)
    
    print("\nComparing Embeddings: Full Clip (30s) vs Short Crop (3s)")
    print(f"{'Filename':<30} | {'Simarity':<10} | {'Status'}")
    print("-" * 60)
    
    total_sim = 0
    
    # 3. Test Loop
    with torch.no_grad():
        for f in test_files:
            try:
                # Load Full Audio
                y, sr = sf.read(f, dtype='float32')
                if len(y.shape) > 1: y = np.mean(y, axis=1)
                
                # --- View 1: 30s (or max available) ---
                crop_30s = int(sr * 30.0)
                y_long = y[:crop_30s] if len(y) > crop_30s else y
                
                # --- View 2: Random 3s Crop ---
                crop_3s = int(sr * 3.0)
                if len(y) > crop_3s:
                    start = random.randint(0, len(y) - crop_3s)
                    y_short = y[start:start+crop_3s]
                else:
                    y_short = np.pad(y, (0, crop_3s - len(y)))
                
                # Encode Long
                w_long = torch.from_numpy(y_long).unsqueeze(0).to(device)
                mels_long = pipeline(w_long, original_sr=sr)
                h_long, _ = model(mels_long)
                h_long = F.normalize(h_long, dim=1)
                
                # Encode Short
                w_short = torch.from_numpy(y_short).unsqueeze(0).to(device)
                mels_short = pipeline(w_short, original_sr=sr)
                h_short, _ = model(mels_short)
                h_short = F.normalize(h_short, dim=1)
                
                # Compare
                sim = torch.mm(h_long, h_short.T).item()
                total_sim += sim
                
                # Check Robustness
                status = "✅" if sim > 0.8 else "⚠️" 
                print(f"{os.path.basename(f)[:28]:<30} | {sim:.4f}     | {status}")
                
            except Exception as e:
                print(f"Error {f}: {e}")
                
    avg_sim = total_sim / len(test_files)
    print("-" * 60)
    print(f"Average Similarity: {avg_sim:.4f}")
    
    if avg_sim > 0.8:
        print("🎉 SUCCESS: Model is Robust to Length (GAP works!)")
    else:
        print("🚧 WARNING: Features diverge significantly. Consider Attention Pooling.")

if __name__ == "__main__":
    test_variable_length()
