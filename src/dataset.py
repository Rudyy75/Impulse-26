
import os
import torch
from torch.utils.data import Dataset
import glob
import random
import torch.nn.functional as F
from .config import SPECTROGRAM_DIR
from .augmentations import AugmentationPipeline

class FMAContrastiveDataset(Dataset):
    """
    Consumes pre-computed .pt spectrograms.
    Returns TWO augmented views for SimCLR.
    """
    def __init__(self, data_dir=SPECTROGRAM_DIR):
        self.files = glob.glob(os.path.join(data_dir, '*.pt'))
        if len(self.files) == 0:
            print(f"WARNING: No .pt files found in {data_dir}. Did you run src/preprocess.py?")
            
        self.augment = AugmentationPipeline()
        self.target_frames = 216 
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load (cached at float16)
        spec = torch.load(self.files[idx], weights_only=False).float()
        if len(spec.shape) == 3: spec = spec.squeeze(0)
            
        # 1. Random Crop (The "content" of the views)
        total_frames = spec.shape[1]
        if total_frames > self.target_frames:
            start = random.randint(0, total_frames - self.target_frames)
            spec = spec[:, start : start + self.target_frames]
        else:
            spec = F.pad(spec, (0, self.target_frames - total_frames))
             
        # 2. Create Two Different Augmented Views (SimCLR style)
        view1 = self.augment(spec.clone())
        view2 = self.augment(spec.clone())
        
        # Add channel dim: (1, 128, 216)
        return view1.unsqueeze(0), view2.unsqueeze(0)
