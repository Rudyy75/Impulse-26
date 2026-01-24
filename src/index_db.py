
import torch
import torch.nn.functional as F
import os
import glob
from tqdm import tqdm
import soundfile as sf
import numpy as np
import faiss

# Imports
from .config import DATA_DIR, EMBEDDING_DIM, N_SAMPLES
from .inference import load_trained_model
from .spectrogram import GPUPipeline

# Output Paths
DB_PATH = os.path.join(os.path.dirname(DATA_DIR), 'vector_db.pt')
FAISS_PATH = os.path.join(os.path.dirname(DATA_DIR), 'index.faiss')

def build_index():
    print("----------------------------------------------------------------")
    print(" 📚 PHASE 3: Building Vector Database (FAISS + GPU)")
    print("----------------------------------------------------------------")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = load_trained_model(device)
    model.eval()
    
    # 2. Init Pipeline
    # 3 seconds window for signature
    # (22050 * 3) / 512 ~ 130 frames
    pipeline = GPUPipeline(device=device)
    target_samples = int(22050 * 3.0) 
    
    # 3. Find Files (Raw MP3s)
    files = glob.glob(os.path.join(DATA_DIR, '**/*.mp3'), recursive=True)
    print(f"Found {len(files)} audio files in {DATA_DIR}")
    
    if len(files) == 0:
        print("❌ No files found!")
        return

    embeddings = []
    filenames = []
    
    print("Indexing...")
    
    # Batch processing could be faster, but single file loop is safer for variable length audio handling
    # We will process 1 by 1 for simplicity of implementation in this step.
    
    with torch.no_grad():
        for file_path in tqdm(files):
            try:
                # Load (CPU)
                y, sr = sf.read(file_path, dtype='float32')
                
                # Mono
                if len(y.shape) > 1: y = np.mean(y, axis=1)
                
                # Center Crop 3s (Signature)
                # --- CRITICAL: Signature Windowing ---
                # We must use a fixed window (e.g. middle 5s) to match query distribution
                if len(y) > N_SAMPLES:
                    start = (len(y) - N_SAMPLES) // 2
                    y = y[start : start + N_SAMPLES]
                elif len(y) < N_SAMPLES:
                    y = np.pad(y, (0, N_SAMPLES - len(y)))
                    
                # To Tensor -> GPU
                waveform = torch.from_numpy(y).unsqueeze(0).to(device) # (1, T)
                
                # Pipeline (Resample -> Mel)
                # Note: We must pass original_sr=sr
                mels = pipeline(waveform, original_sr=sr) # (1, 1, 128, T)
                
                # Forward
                # model(x) returns (h, z). 
                # z is 128/512-dim, normalized, and much more discriminative for retrieval.
                _, z = model(mels)
                
                # Normalize L2 (z is already normalized by default in SimCLR, but good to be explicit)
                z = F.normalize(z, dim=1)
                
                embeddings.append(z.cpu())
                filenames.append(file_path)
                
            except Exception as e:
                if len(embeddings) == 0:
                    print(f"DEBUG Error on first file: {e}")
                pass
                
    if not embeddings:
        print("❌ No valid embeddings generated.")
        return

    # Stack
    E = torch.cat(embeddings, dim=0).numpy() # (N, D)
    print(f"📊 Embeddings Shape: {E.shape}")
    
    # 4. Build FAISS Index
    d = E.shape[1]
    index = faiss.IndexFlatIP(d) # Inner Product (~Cosine since normalized)
    index.add(E)
    
    # 5. Save
    print(f"💾 Saving Index to {FAISS_PATH}...")
    faiss.write_index(index, FAISS_PATH)
    
    # Save Metadata (Filenames)
    torch.save({
        'filenames': filenames,
        'embeddings': E # Optional, keeping for legacy compatibility
    }, DB_PATH)
    
    print("✅ vector_db.pt and index.faiss saved.")

if __name__ == "__main__":
    build_index()
