
import torch
import torch.nn.functional as F
import torchaudio
import os
import random
import numpy as np
from tqdm import tqdm
from src.inference import load_trained_model
from src.spectrogram import GPUPipeline
from src.dataset import FMAContrastiveDataset
from src.config import DATA_DIR, SPECTROGRAM_DIR, SAMPLE_RATE, N_SAMPLES, HOP_LENGTH
from src.augmentations import AugmentationPipeline

# Simulation Helper Functions
def add_heavy_reverb(waveform, sample_rate):
    # Simulate a large hall: Delay + Decay
    # Simple simulation using RIR (Impulse Response) convolution is best, 
    # but for pure python we use torchaudio effects if available, or simple delay.
    
    # Simple "Slapback" echo simulation
    delay_ms = 100
    decay = 0.6
    
    delay_samples = int(sample_rate * delay_ms / 1000)
    
    # Create decay copy
    delayed = torch.roll(waveform, shifts=delay_samples, dims=1)
    delayed[:, :delay_samples] = 0 # Updates silence
    
    return waveform + (delayed * decay)

def add_street_noise(waveform, snr_db=5):
    # Simulate street noise using white noise (approximation)
    # Target SNR = 5dB (Very Noisy!)
    
    noise = torch.randn_like(waveform)
    
    # Calculate energy
    signal_energy = torch.mean(waveform ** 2)
    noise_energy = torch.mean(noise ** 2)
    
    # Scale noise to match target SNR
    # SNR = 10 * log10(Es/En)
    # 5 = 10 * log10(Es/En) -> 0.5 = log10(Es/En) -> 10^0.5 = Es/En
    
    target_ratio = 10 ** (snr_db / 10) # ~3.16
    
    # We want En = Es / target_ratio
    target_noise_energy = signal_energy / target_ratio
    
    scale = torch.sqrt(target_noise_energy / (noise_energy + 1e-9))
    
    return waveform + (noise * scale)

def run_acid_test():
    print("----------------------------------------------------------------")
    print(" 🧪 THE ACID TEST: Simulating Hackathon Evaluation")
    print("----------------------------------------------------------------")
    print("Conditions: 5s Clips | SNR ≈ 5dB | Heavy Reverb")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Brain on {device}...")
    
    model = load_trained_model(device)
    
    # 1. Load Clean Database Vector DB
    # We assume index_db.py has run
    DB_PATH = os.path.join(os.path.dirname(DATA_DIR), 'vector_db.pt')
    if not os.path.exists(DB_PATH):
        print("❌ DB not found. Run src/index_db.py first.")
        return
        
    db = torch.load(DB_PATH, weights_only=False)
    db_embeddings = db['embeddings']
    if isinstance(db_embeddings, np.ndarray):
        db_vecs = torch.from_numpy(db_embeddings).to(device)
    else:
        db_vecs = db_embeddings.to(device)
    db_files = db['filenames']
    
    # 2. Select Test Candidates (50 Random Tracks)
    dataset = FMAContrastiveDataset(SPECTROGRAM_DIR)
    
    if len(dataset) < 50:
        print("Warning: Dataset smaller than 50. Testing all.")
        indices = list(range(len(dataset)))
    else:
        indices = random.sample(range(len(dataset)), 50)
        
    correct_matches = 0
    total = len(indices)
    
    # Pre-calc dimensions
    TARGET_FRAMES = int(N_SAMPLES / HOP_LENGTH) + 1
    
    print(f"Running query on {total} mystery tracks...")
    
    with torch.no_grad():
        for idx in tqdm(indices):
            # A. Get Ground Truth Info
            file_path = dataset.files[idx]
            # Basename for matching (e.g. 000123.mp3)
            true_filename = os.path.basename(file_path).replace('.pt', '.mp3')
            
            # B. Load Original Spectrogram to synthesize audio?
            # PROBLEM: We only have spectrograms in .pt! 
            # We can't revert spectrogram -> audio perfectly (phase loss).
            # SOLUTION: We must generate the 'Noisy Query' at spectrogram level 
            # using our Augmentor (which approximates signal degradation)
            # OR we load the original MP3s if available.
            
            # Let's try to load the MP3 if DATA_DIR is populated, else fallback to spec augment
            mp3_path = os.path.join(DATA_DIR, true_filename[:3], true_filename)
            
            spec_input = None
            
            if os.path.exists(mp3_path):
                # 1. Load Audio
                waveform, sr = torchaudio.load(mp3_path)
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
                
                # 2. Take a CENTER 5s slice (to match Indexing)
                if waveform.shape[1] > N_SAMPLES:
                    start = (waveform.shape[1] - N_SAMPLES) // 2
                    waveform = waveform[:, start : start + N_SAMPLES]
                else:
                    pad = N_SAMPLES - waveform.shape[1]
                    waveform = F.pad(waveform, (0, pad))
                
                # 3. Apply ACID EFFECTS (Audio Domain)
                if random.random() < 0.5:
                    waveform = add_street_noise(waveform, snr_db=5)
                else:
                    waveform = add_heavy_reverb(waveform, SAMPLE_RATE)
                
                # Use Pipeline
                gpu_pipeline = GPUPipeline(device=device)
                spec_input = gpu_pipeline(waveform.to(device), original_sr=SAMPLE_RATE).squeeze(1) # (1, 128, T)
                
            else:
                # Fallback: Spectrogram Noise (Approximation)
                spec = torch.load(file_path, weights_only=True).float()
                # Crop
                if spec.shape[2] > TARGET_FRAMES:
                    start = random.randint(0, spec.shape[2] - TARGET_FRAMES)
                    spec = spec[:, :, start : start + TARGET_FRAMES]
                
                augmentor = AugmentationPipeline(noise_std=0.1) # Higher noise for acid test
                spec_input = augmentor(spec)
            
            # C. Encode Query
            q = spec_input.unsqueeze(0).to(device)
            # Ensure correct shape (batch, 1, 128, 216)
            if len(q.shape) == 3: q = q.unsqueeze(0)
                
            _, z = model(q) # Use the discriminative SimCLR head (z)
            q_vec = F.normalize(z, p=2, dim=1)
            
            # D. Search
            scores = torch.matmul(q_vec, db_vecs.T).squeeze(0)
            best_idx = torch.argmax(scores).item()
            predicted_path = db_files[best_idx]
            predicted_filename = os.path.basename(predicted_path)
            
            if predicted_filename == true_filename:
                correct_matches += 1
            
            # DEBUG PRINT (First 5)
            if idx in indices[:5]:
                sim_score = scores[best_idx].item()
                print(f"DEBUG | True: {true_filename} | Pred: {predicted_filename} | Score: {sim_score:.4f} | Match: {predicted_filename == true_filename}")
                
    accuracy = (correct_matches / total) * 100
    print("\n----------------------------------------------------------------")
    print(f"🔥 FINAL SCORE: {accuracy:.1f}%")
    print("----------------------------------------------------------------")
    
    if accuracy >= 80:
        print("🏆 PASSED: Ready for Submission!")
    elif accuracy >= 60:
        print("⚠️ MARGINAL: Might need more augmentation training.")
    else:
        print("❌ FAILED: The model is too fragile.")

if __name__ == "__main__":
    run_acid_test()
