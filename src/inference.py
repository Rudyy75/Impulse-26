
import torch
import torch.nn.functional as F
import os
import torchaudio
from src.model import EchoFindModel
from src.config import WEIGHTS_DIR, SAMPLE_RATE, N_SAMPLES, HOP_LENGTH, EMBEDDING_DIM
from src.spectrogram import GPUPipeline

# Re-calculate target frames for consistency
TARGET_FRAMES = 216 # Match dataset and model

def load_trained_model(device):
    """
    Loads the EchoFind model with trained weights.
    Returns the encoder (backbone) ready for inference.
    """
    model = EchoFindModel(embedding_dim=EMBEDDING_DIM).to(device)
    # Prioritize SimCLR weights (Best for Retrieval)
    simclr_path = os.path.join(WEIGHTS_DIR, 'best_simclr.pt')
    encoder_path = os.path.join(WEIGHTS_DIR, 'encoder.pth')
    
    if os.path.exists(simclr_path):
        print(f"✅ Loading SimCLR Weights from {simclr_path}")
        checkpoint = torch.load(simclr_path, map_location=device, weights_only=True)
        # Handle state_dict key
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Filter for backbone only if needed, OR load full model
        # EchoFindModel contains .backbone
        # If saved state_dict is full model, we can load directly.
        try:
             model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
             # Try loading into backbone if keys match 'backbone.'
             # Actually, strict=False handles most.
             pass
             
    elif os.path.exists(encoder_path):
        print(f"⚠️ SimCLR weights not found. Loading fallback: {encoder_path}")
        state_dict = torch.load(encoder_path, map_location=device, weights_only=True)
        model.backbone.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Weights not found! Run src/train.py first.")
    
    model.eval() # CRITICAL: Disables Dropout & BatchNorm updates
    return model

def preprocess_for_inference(audio_path, device):
    """
    Loads an audio file and converts it to a tensor suitable for the model.
    Uses DETERMINISTIC processing (Center Crop), not random.
    """
    try:
        # 1. Pipeline
        pipeline = GPUPipeline(device=device)
        
        # 2. Load Audio
        y, sr = torchaudio.load(audio_path)
        y = y.to(device)
        
        # 3. Mono
        if y.shape[0] > 1:
            y = torch.mean(y, dim=0, keepdim=True)
        
        # 4. Pipeline Handles (Resample -> Mel -> Norm)
        spec = pipeline(y, original_sr=sr) # (1, 1, 128, T)
        
        # 5. Length Control (Deterministic Center Crop for 5s signature)
        # T should be around 216.
        T = spec.shape[3]
        if T > 216:
            start = (T - 216) // 2
            spec = spec[:, :, :, start : start + 216]
        elif T < 216:
            pad = 216 - T
            spec = F.pad(spec, (0, pad))
            
        return spec.to(device)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None
