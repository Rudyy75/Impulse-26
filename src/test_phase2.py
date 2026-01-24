
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from src.model import EchoFindModel
from src.dataset import FMAContrastiveDataset
from src.config import DATA_DIR, WEIGHTS_DIR, SPECTROGRAM_DIR
from src.spectrogram import audio_to_melspec, load_audio
from src.augmentations import AugmentationPipeline

def test_retrieval_accuracy():
    print("----------------------------------------------------------------")
    print(" 🧪 PHASE 2 VERIFICATION: The 'Shazam' Simulation")
    print("----------------------------------------------------------------")
    
    # 1. Load the Trained Brain
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # improved: Load only the backbone (Encoder) since that's what we save
    model = EchoFindModel().to(device)
    encoder_path = os.path.join(WEIGHTS_DIR, 'encoder.pth')
    
    if not os.path.exists(encoder_path):
        print(f"❌ Error: Weights not found at {encoder_path}")
        return

    # Partial load: We only saved model.backbone
    # weights_only=True is safer
    state_dict = torch.load(encoder_path, map_location=device, weights_only=True)
    model.backbone.load_state_dict(state_dict)
    model.eval() # Freeze BatchNorm/Dropout
    print("✅ Weights loaded successfully.")
    
    # 2. Setup a Mini-Database (20 random songs)
    # We use the dataset class but grab a small subset
    # IMPORTANT: Use SPECTROGRAM_DIR not DATA_DIR
    dataset = FMAContrastiveDataset(SPECTROGRAM_DIR)
    
    if len(dataset) == 0:
         print(f"❌ Error: No .pt files found in {SPECTROGRAM_DIR}")
         return
         
    subset_indices = list(range(min(20, len(dataset)))) # Robustness: Use available files
    
    print(f"Building Database of {len(subset_indices)} songs...")
    
    db_embeddings = []
    queries = []
    
    augmentor = AugmentationPipeline()
    
    # Pre-calculated target frames for 5s crop logic (copied from dataset.py)
    from src.config import N_SAMPLES, HOP_LENGTH
    TARGET_FRAMES = int(N_SAMPLES / HOP_LENGTH) + 1
    
    with torch.no_grad():
        for i in tqdm(subset_indices):
            # Load Pre-computed Spectrogram manually
            file_path = dataset.files[i]
            spec = torch.load(file_path, weights_only=False).float()
            
            # Since .pt files are full songs, we need to crop them for the test to work
            # logic similar to Dataset.__getitem__
            total_frames = spec.shape[2]
            if total_frames > TARGET_FRAMES:
                start = 0 # Deterministic start for "Clean DB"
                spec = spec[:, :, start : start + TARGET_FRAMES]
            elif total_frames < TARGET_FRAMES:
                pad = TARGET_FRAMES - total_frames
                spec = F.pad(spec, (0, pad))
                
            spec = spec.unsqueeze(0).to(device) # (1, 1, 128, 216)
            
            # A. Clean Embedding (The Database Entry)
            h_clean, _ = model(spec)
            # Normalize for Cosine Similarity!
            h_clean = F.normalize(h_clean, p=2, dim=1)
            db_embeddings.append(h_clean)
            
            # B. Noisy Query (The Simulated Recording)
            # We apply augmentations to simulate recording environment
            aug_spec = augmentor(spec.squeeze(0)).unsqueeze(0).to(device)
            h_noisy, _ = model(aug_spec)
            h_noisy = F.normalize(h_noisy, p=2, dim=1)
            queries.append(h_noisy)
            
    # Stack into tensors
    db_tensor = torch.cat(db_embeddings, dim=0) # (20, 512)
    query_tensor = torch.cat(queries, dim=0)    # (20, 512)
    
    # 3. The Retrieval Test (All-vs-All Dot Product)
    # Similarity Matrix: Row i = Query i vs All DB entries
    similarity_matrix = torch.matmul(query_tensor, db_tensor.T) # (20, 20)
    
    # 4. Calculate Accuracy
    # For query i, the max similarity SHOULD be index i
    top1_matches = 0
    top3_matches = 0
    
    print("\n🔍 Retrieval Results:")
    for i in range(len(subset_indices)):
        scores = similarity_matrix[i]
        top_indices = torch.topk(scores, k=3).indices.tolist()
        
        is_top1 = (top_indices[0] == i)
        is_top3 = (i in top_indices)
        
        if is_top1:
            top1_matches += 1
            status = "✅"
        elif is_top3:
            top3_matches += 1
            status = "⚠️"
        else:
            status = "❌"
            
        # Optional: Print failed cases
        if not is_top1:
             print(f"Song {i}: {status} found at rank {top_indices.index(i) if i in top_indices else '>3'}")

    acc1 = (top1_matches / len(subset_indices)) * 100
    acc3 = (top3_matches / len(subset_indices)) * 100
    
    print("----------------------------------------------------------------")
    print(f"Top-1 Accuracy: {acc1:.1f}%  (Exact match)")
    print(f"Top-3 Accuracy: {acc3:.1f}%  (In top 3 candidates)")
    print("----------------------------------------------------------------")
    
    if acc1 > 80:
        print("🎉 SUCCESS: The model has learned robust semantic representations!")
    else:
        print("🚧 WARNING: Accuracy is low. Check training duration or augmentation strength.")

    # 5. Visualize the Similarity Matrix
    try:
        plt.figure(figsize=(8, 8))
        plt.imshow(similarity_matrix.cpu().numpy(), cmap='viridis')
        plt.title("Cosine Similarity Matrix (Diagonal = Good)")
        plt.xlabel("Database Index")
        plt.ylabel("Query Index")
        plt.colorbar()
        plt.savefig("phase2_retrieval_matrix.png")
        print("Saved 'phase2_retrieval_matrix.png' - Look for a bright diagonal line.")
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    test_retrieval_accuracy()
