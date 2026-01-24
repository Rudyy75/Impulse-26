
import torch
import random
import torch.nn.functional as F

class AugmentationPipeline:
    """
    CPU-based stochastic augmentation for Latent Space learning.
    Applied inside the DataLoader (parallel workers).
    """
    def __init__(self, time_mask_param=30, freq_mask_param=20):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def __call__(self, spec):
        # spec: (128, T)
        
        # 1. Random Noise
        if random.random() < 0.5:
            strength = random.uniform(0.01, 0.05)
            spec = spec + torch.randn_like(spec) * strength
            
        # 2. Time Mask
        if random.random() < 0.5:
            n_frames = spec.shape[1]
            mask_len = random.randint(0, self.time_mask_param)
            mask_start = random.randint(0, max(0, n_frames - mask_len))
            spec[:, mask_start:mask_start + mask_len] = 0
            
        # 3. Freq Mask
        if random.random() < 0.5:
            n_mels = spec.shape[0]
            mask_len = random.randint(0, self.freq_mask_param)
            mask_start = random.randint(0, max(0, n_mels - mask_len))
            spec[mask_start:mask_start + mask_len, :] = 0
            
        return spec
