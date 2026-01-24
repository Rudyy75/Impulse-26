import torch
import torch.nn as nn
import random

class AugmentationPipeline:
    """
    Stochastic augmentation module for Self-Supervised Learning.
    Applies random transformations to spectrograms to create diverse 'views'.
    """
    def __init__(self, time_mask_param=30, freq_mask_param=20, noise_std=0.03):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_std = noise_std
        
        # We can use Torchaudio's built-in masking, but implementing manually gives more control
        # logic is simple enough.

    def add_noise(self, spec):
        """Add random Gaussian noise."""
        noise = torch.randn_like(spec) * self.noise_std
        return spec + noise

    def time_mask(self, spec):
        """Mask random strips of time (Vertical columns)."""
        # spec shape: (channels, n_mels, time_steps)
        _, _, time_steps = spec.shape
        mask_len = random.randint(0, self.time_mask_param)
        start = random.randint(0, time_steps - mask_len)
        
        # Clone to avoid modifying original in-place if passed by reference
        augmented_spec = spec.clone()
        augmented_spec[:, :, start : start + mask_len] = 0
        return augmented_spec

    def freq_mask(self, spec):
        """Mask random strips of frequencies (Horizontal rows)."""
        # spec shape: (channels, n_mels, time_steps)
        _, n_mels, _ = spec.shape
        mask_len = random.randint(0, self.freq_mask_param)
        start = random.randint(0, n_mels - mask_len)
        
        augmented_spec = spec.clone()
        augmented_spec[:, start : start + mask_len, :] = 0
        return augmented_spec
    
    def time_shift(self, spec, max_shift=20):
        """Roll the spectrogram along time axis."""
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(spec, shifts=shift, dims=2)

    def __call__(self, spec):
        """
        Apply a random sequence of augmentations.
        Inputs: spec (Tensor) - The original Log-Mel Spectrogram
        Returns: augmented_spec (Tensor)
        """
        out = spec.clone()

        # Randomly apply augmentations
        # 80% chance for noise (Critical for Shazam robustness)
        if random.random() < 0.8:
            out = self.add_noise(out)
            
        # 50% chance for Time Masking
        if random.random() < 0.5:
            out = self.time_mask(out)
            
        # 50% chance for Frequency Masking
        if random.random() < 0.5:
            out = self.freq_mask(out)
            
        # 30% chance for Time Shift
        if random.random() < 0.3:
            out = self.time_shift(out)

        return out
