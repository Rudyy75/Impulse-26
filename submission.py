import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import faiss
import torchaudio

# Relative imports from src
from src.model import EchoFindModel
from src.config import WEIGHTS_DIR, SAMPLE_RATE, N_SAMPLES, EMBEDDING_DIM
from src.spectrogram import GPUPipeline

class AudioEncoder:
    """
    Standardized Encoder class for Impulse'26 Submission.
    Handles weights loading and feature extraction.
    """
    def __init__(self, weights_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EchoFindModel(embedding_dim=EMBEDDING_DIM).to(self.device)
        
        # Determine weights path
        if weights_path is None:
            weights_path = os.path.join(WEIGHTS_DIR, 'best_simclr.pt')
        
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded weights from {weights_path}")
        else:
            print(f"⚠️ Warning: Weights not found at {weights_path}. Using random initialization.")
            
        self.model.eval()
        self.pipeline = GPUPipeline(device=self.device)

    def get_embedding(self, audio_path):
        """
        Extracts a 512-dim (Backbone h) or Projection (z) embedding.
        We return Projection 'z' as it was optimized for similarity.
        """
        try:
            # Load Audio
            y, sr = torchaudio.load(audio_path)
            y = y.to(self.device)
            if y.shape[0] > 1:
                y = torch.mean(y, dim=0, keepdim=True)
                
            # Preprocess (Resample -> Mel -> Norm)
            mels = self.pipeline(y, original_sr=sr)
            
            # Signature Crop (Center 5s)
            T = mels.shape[3]
            if T > 216:
                start = (T - 216) // 2
                mels = mels[:, :, :, start : start + 216]
            elif T < 216:
                mels = F.pad(mels, (0, 216 - T))
            
            # Inference
            with torch.no_grad():
                h, z = self.model(mels)
                # We return z (normalized projection) for retrieval tasks
                return z.cpu().numpy()
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

def get_embedding(audio_path):
    """
    Submission helper function.
    """
    # Using a global cache to avoid reloading model every time
    if not hasattr(get_embedding, "encoder"):
        get_embedding.encoder = AudioEncoder()
    return get_embedding.encoder.get_embedding(audio_path)

def predict_track(noisy_audio_path, database_path=None):
    """
    Identifies the track from noisy audio using a FAISS index.
    
    noisy_audio_path: Path to the query .mp3/wav
    database_path: (Optional) Path to the .faiss index file. 
                   Defaults to project-wide index if not provided.
    """
    # 1. Get Embedding
    q_vec = get_embedding(noisy_audio_path)
    if q_vec is None: return None
    
    # 2. Load FAISS Index
    if database_path is None:
        from src.config import DATA_DIR
        database_path = os.path.join(os.path.dirname(DATA_DIR), 'index.faiss')
        meta_path = os.path.join(os.path.dirname(DATA_DIR), 'vector_db.pt')
    else:
        meta_path = database_path.replace('.faiss', '.pt')
        
    if not os.path.exists(database_path):
        print("❌ FAISS Index not found!")
        return None
        
    index = faiss.read_index(database_path)
    
    # 3. Search
    D, I = index.search(q_vec, 1) # Top-1 match
    best_idx = I[0][0]
    
    # 4. Map to Filename
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, weights_only=False)
        filenames = meta.get('filenames', [])
        if best_idx < len(filenames):
            return filenames[best_idx]
            
    return f"Index_{best_idx}" # Fallback

if __name__ == "__main__":
    # Example usage verification
    print("Testing Submission API...")
    # Add a dummy path or skip if not in test mode
    pass
