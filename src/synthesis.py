
import torch
import torchaudio
import torchaudio.transforms as T
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
from src.decoder import EchoFindDecoder
from src.spectrogram import GPUPipeline
from src.config import WEIGHTS_DIR, DATA_DIR

def load_decoder(device):
    decoder = EchoFindDecoder().to(device)
    path = os.path.join(WEIGHTS_DIR, 'decoder_final.pt')
    
    # Try different checkpoints
    if not os.path.exists(path):
         ep20 = os.path.join(WEIGHTS_DIR, 'decoder_ep20.pt')
         if os.path.exists(ep20):
             path = ep20
             
    if os.path.exists(path):
        decoder.load_state_dict(torch.load(path, map_location=device))
        decoder.eval()
        print(f"✅ Decoder loaded from {path}")
        return decoder
    else:
        print("❌ Decoder weights not found. You must run src/train_decoder.py first!")
        return None

def interpolate_and_synthesize(file_a, file_b, alpha=0.5):
    """
    Interpolates between two audio files in latent space.
    alpha: 0.0 (A) -> 1.0 (B).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎵 Synthesis | Device: {device}")
    
    # 1. Load Models
    encoder = load_trained_model(device) # Needs src/train.py finished
    decoder = load_decoder(device)       # Needs src/train_decoder.py finished
    
    if not encoder or not decoder: 
        print("⚠️ Models not ready. Please finish training Phase 2 and Phase 5.")
        return

    # 2. Pipeline Components
    pipeline = GPUPipeline(device=device)
    
    # Vocoder Pipeline (Inverse)
    n_stft = 2048 // 2 + 1 # 1025
    inv_mel = T.InverseMelScale(n_stft=n_stft, n_mels=128, sample_rate=22050).to(device)
    griffin_lim = T.GriffinLim(n_fft=2048, hop_length=512, n_iter=60).to(device)
    
    # 3. Helper to Encode
    def get_latent(path):
        try:
            y, sr = sf.read(path, dtype='float32')
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            y_tensor = torch.from_numpy(y).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Pipeline does: Resample -> Mel -> Norm
                mels = pipeline(y_tensor, original_sr=sr)
                h, _ = encoder(mels) # (1, 2048)
            return h
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    print(f"Encoding A: {os.path.basename(file_a)}")
    z_a = get_latent(file_a)
    
    print(f"Encoding B: {os.path.basename(file_b)}")
    z_b = get_latent(file_b)
    
    if z_a is None or z_b is None: return
    
    # 4. Interpolate
    print(f"mixing... (Alpha={alpha})")
    z_new = (1 - alpha) * z_a + alpha * z_b
    
    # 5. Decode
    print("Decoding & Vocoding...")
    with torch.no_grad():
        rec_mel = decoder(z_new) # (1, 1, 128, 216)
        
        # Invert Normalization (Approx)
        rec_db = (rec_mel + 1) / 2 * 80 - 80
        rec_amp = torch.pow(10, rec_db / 20)
        
        # Inverse Mel -> Linear
        lin_spec = inv_mel(rec_amp.squeeze(1)) # (1, 1025, 216)
        
        # Griffin Lim -> Audio
        waveform = griffin_lim(lin_spec)
        
    # 6. Save
    out_name = f"morph_{os.path.basename(file_a)[:5]}_{os.path.basename(file_b)[:5]}.wav"
    out_path = os.path.join(os.getcwd(), out_name)
    
    sf.write(out_path, waveform.squeeze().cpu().numpy(), 22050)
    print(f"\n✨ Generated Hybrid Audio: {out_path}")

if __name__ == "__main__":
    # Auto-Pick
    all_files = glob.glob(os.path.join(DATA_DIR, '**/*.mp3'), recursive=True)
    if len(all_files) >= 2:
        a = random.choice(all_files)
        b = random.choice(all_files)
        while b == a: b = random.choice(all_files)
        
        print(f"A: {a}")
        print(f"B: {b}")
        interpolate_and_synthesize(a, b, alpha=0.5)
    else:
        print("Not enough files.")
