# Implementing the math of SimCLR

import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_Xent_loss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR Loss).
    
    The goal:
    - Maximize cosine similarity between z_i and z_j (positive pair).
    - Minimize cosine similarity between z_i and ALL other vectors (negatives).
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        
        # 1. Concatenate all vectors: [z1_0, z1_1..., z2_0, z2_1...]
        z = torch.cat([z1, z2], dim=0) # Shape: (2N, D)
        
        # 2. Compute Similarity Matrix (Cosine Similarity)
        # sim[i, j] = cos(z[i], z[j]) / temperature
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature
        
        # 3. Create Labels (The "Correct" Matches)
        # For index i (0 to N-1), the positive match is i + N
        # For index i (N to 2N-1), the positive match is i - N
        labels = torch.cat([
            torch.arange(batch_size, device=z.device) + batch_size,
            torch.arange(batch_size, device=z.device)
        ])
        
        # 4. Mask out self-similarity (diagonal)
        # We don't want the model to learn that "A is similar to A". That's trivial.
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        
        # We set diagonal to -infinity so softmax ignores it (exp(-inf) = 0)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # 5. Calculate Cross Entropy Loss
        # PyTorch's CrossEntropyLoss includes Softmax automatically.
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss