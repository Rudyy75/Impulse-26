
import torch
import torch.nn.functional as F
import os
import faiss
import numpy as np
import soundfile as sf
import time

# Imports
from .config import DATA_DIR, EMBEDDING_DIM, N_SAMPLES
from .inference import load_trained_model
from .spectrogram import GPUPipeline

# Paths
DB_PATH = os.path.join(os.path.dirname(DATA_DIR), 'vector_db.pt')
FAISS_PATH = os.path.join(os.path.dirname(DATA_DIR), 'index.faiss')

def load_db():
    print("⏳ Loading Vector DB & Index...")
    if not os.path.exists(FAISS_PATH) or not os.path.exists(DB_PATH):
        print("❌ Database not found! Run src/index_db.py")
        return None, None
        
    # Load Index
    index = faiss.read_index(FAISS_PATH)
    
    # Load Metadata
    meta = torch.load(DB_PATH, weights_only=False)
    filenames = meta['filenames']
    
    print(f"✅ Loaded Index with {index.ntotal} vectors.")
    return index, filenames

def query_audio(audio_path, model, index, filenames, device, k=5):
    """
    Search for similar audio using FAISS.
    """
    try:
        # 1. Load Query
        y, sr = sf.read(audio_path, dtype='float32')
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        
        # Center Crop/Pad (Signature)
        if len(y) > N_SAMPLES:
            start = (len(y) - N_SAMPLES) // 2
            y = y[start:start+N_SAMPLES]
        elif len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
            
        # 2. Pipeline
        pipeline = GPUPipeline(device=device) 
        # Note: We create pipeline here. In production, pass it in.
        
        waveform = torch.from_numpy(y).unsqueeze(0).to(device)
        mels = pipeline(waveform, original_sr=sr)
        
        # 3. Model
        with torch.no_grad():
            _, z = model(mels)
            z = F.normalize(z, dim=1).cpu().numpy()
            
        # 3.5. OOD Check (Extension B)
        from .ood import OODDetector
        detector = OODDetector.load()
        if detector:
            score, is_ood = detector.score(z)
            if is_ood:
                print(f"⚠️ ANOMALY DETECTED! (Score: {score:.4f} > {detector.threshold:.4f})")
                print("   This audio does not appear to be music. Rejecting.")
                return []
            else:
                 print(f"✅ OOD Check Passed. (Score: {score:.4f})")
        
        # 4. Search
        t0 = time.time()
        D, I = index.search(h, k)
        dt = time.time() - t0
        
        # 5. Results
        results = []
        print(f"\n🔍 Search Results ({dt*1000:.2f}ms):")
        for i in range(k):
            idx = I[0][i]
            score = D[0][i]
            fname = filenames[idx]
            basename = os.path.basename(fname)
            print(f"Rank {i+1}: {basename} (Score: {score:.4f})")
            results.append((fname, score))
            
        return results
        
    except Exception as e:
        print(f"Error querying {audio_path}: {e}")
        return []

def run_interactive():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(device)
    model.eval()
    
    index, filenames = load_db()
    if not index: return
    
    while True:
        path = input("\n🎵 Enter path to audio file (or 'q' to quit): ").strip().strip('"')
        if path.lower() == 'q': break
        
        if os.path.exists(path):
            query_audio(path, model, index, filenames, device)
        else:
            print("❌ File not found.")

if __name__ == "__main__":
    run_interactive()
