
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.dataset import FMAUnsupervisedDataset
from src.spectrogram import GPUPipeline
from src.inference import load_trained_model
from src.decoder import EchoFindDecoder
from src.config import DATA_DIR, WEIGHTS_DIR, BATCH_SIZE

def train_decoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎨 GenAI Training | Device: {device}")
    
    # 1. Load Frozen Encoder
    print("Loading Encoder...")
    try:
        # Load trained weights (SimCLR or Finetune)
        encoder_full = load_trained_model(device) # Returns full EchoFindModel
        encoder_full.eval()
        for param in encoder_full.parameters():
            param.requires_grad = False
        print("✅ Encoder Loaded & Frozen.")
    except Exception as e:
        print(f"❌ Error loading encoder: {e}")
        return

    # 2. Init Decoder
    decoder = EchoFindDecoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 3. Data
    dataset = FMAUnsupervisedDataset(DATA_DIR)
    # Using smaller batch size for decoder training if needed, but 64 (standard) is fine
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    gpu_prep = GPUPipeline(device=device)
    
    EPOCHS = 20
    
    print(f"🚀 Starting Decoder Training ({EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        decoder.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for waveform, sr in pbar:
            waveform = waveform.to(device)
            
            # Get Targets (Spectrograms)
            with torch.no_grad():
                mels = gpu_prep(waveform)  # (B, 1, 128, 216)
                
                # Get Latent (h)
                # model(x) -> (h, z)
                h, _ = encoder_full(mels) # (B, 2048)
            
            # Train Decoder
            optimizer.zero_grad()
            rec = decoder(h) # (B, 1, 128, 216)
            
            # Loss: Reconstruct the input mel-spectrogram
            loss = criterion(rec, mels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'mse': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"📉 Epoch {epoch+1} MSE: {avg_loss:.4f}")
        
        # Save Intermediate
        if (epoch+1) % 5 == 0:
            torch.save(decoder.state_dict(), os.path.join(WEIGHTS_DIR, f'decoder_ep{epoch+1}.pt'))
            
    # Final Save
    save_path = os.path.join(WEIGHTS_DIR, 'decoder_final.pt')
    torch.save(decoder.state_dict(), save_path)
    print(f"✅ Decoder Saved to {save_path}")

if __name__ == "__main__":
    train_decoder()
