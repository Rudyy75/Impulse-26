
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from .config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, TEMPERATURE, 
    WEIGHTS_DIR, EMBEDDING_DIM, SPECTROGRAM_DIR
)
from .dataset import FMAContrastiveDataset
from .model import EchoFindModel
from .losses import NT_Xent_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Phase 2 (SimCLR) | Speed: Optimized CPU Workers | Device: {device}")

    # 1. Data (Pre-augmented on CPU for max throughput)
    dataset = FMAContrastiveDataset(SPECTROGRAM_DIR)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True, 
        num_workers=8, # More workers = 30s/epoch
        persistent_workers=True,
        pin_memory=True
    )
    
    print(f"📂 Dataset: {len(dataset)} items. Batch: {BATCH_SIZE}")
    
    # 2. Model
    model = EchoFindModel(embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    criterion = NT_Xent_loss(temperature=TEMPERATURE).to(device)
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision
    
    # 3. Training Loop
    best_loss = float('inf')
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    print(f"🚀 Starting Pre-training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            
            with torch.cuda.amp.autocast():
                # Forward (Two Views)
                _, z1 = model(view1)
                _, z2 = model(view2)
                
                loss = criterion(z1, z2)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"📉 Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, os.path.join(WEIGHTS_DIR, 'best_simclr.pt'))
            
    print(f"✅ Training Complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
