import torch
from src.spectrogram import load_audio, audio_to_melspec
from src.augmentations import AugmentationPipeline
from src.dataset import FMAContrastiveDataset
from src.config import DATA_DIR, N_SAMPLES

def test_pipeline():
    print("Testing Phase 1 Pipeline...")
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"WARNING: Data directory {DATA_DIR} not found. Skipping dataset test.")
        return

    # 1. Test Spectrogram
    dataset = FMAContrastiveDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No files found in dataset! Check path.")
        return

    print(f"Dataset size: {len(dataset)}")
    
    # Get one item
    view1, view2 = dataset[0]
    
    print(f"View 1 shape: {view1.shape}") # Should be (1, 128, 216)
    print(f"View 2 shape: {view2.shape}") 
    
    assert view1.shape == (1, 128, 216), f"Shape mismatch! Got {view1.shape}"
    assert view2.shape == (1, 128, 216), f"Shape mismatch! Got {view2.shape}"
    assert not torch.equal(view1, view2), "Augmentations failed! Views are identical."
    
    print("SUCCESS: Phase 1 Pipeline Verified (Shapes & Distinctness)!")
    
    # 2. Visual Inspection (Requirement: Preserve semantic structure)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Remove channel dim for plotting: (1, 128, 216) -> (128, 216)
        axes[0].imshow(view1.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title("View 1 (Augmented)")
        axes[1].imshow(view2.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title("View 2 (Augmented)")
        
        plt.tight_layout()
        plt.savefig("phase1_verification.png")
        print("Generated 'phase1_verification.png' for visual inspection.")
    except ImportError:
        print("Matplotlib not found. Skipping visual generation.")

if __name__ == "__main__":
    import os
    test_pipeline()
