import os
import torch
from torch.utils.data import Dataset
import glob
from .spectrogram import load_audio, audio_to_melspec
from .augmentations import AugmentationPipeline

class FMAContrastiveDataset(Dataset):
    """
    PyTorch Dataset for Self-Supervised Learning.
    Returns TWO augmented views of the same audio file.
    """
    def __init__(self, data_dir, ext='.mp3'):
        """
        Args:
            data_dir (str): Path to fma_small folder
            ext (str): File extension to search for (default .mp3)
        """
        # Recursively find all audio files
        self.files = glob.glob(os.path.join(data_dir, '**', f'*{ext}'), recursive=True)
        print(f"Found {len(self.files)} files in {data_dir}")
        
        self.augment = AugmentationPipeline()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        # 1. Load and Convert to Spectrogram
        # Note: We do this ONCE per item to save I/O, then augment duplicates
        waveform = load_audio(audio_path)
        spec = audio_to_melspec(waveform)
        
        # 2. Generate Two Views (Augmentations)
        # Randomness is inside the __call__ method of AugmentationPipeline
        view1 = self.augment(spec)
        view2 = self.augment(spec)
        
        return view1, view2
