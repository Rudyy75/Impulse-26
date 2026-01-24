#Defining the neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .config import N_MELS

class EchoFindModel(nn.Module):
    def __init__(self, embedding_dim = 128):
        super().__init__()

        # 1. LOAD BACKBONE (ResNet50 Upgrade)
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        # 2. MODIFY INPUT LAYER (1 Channel)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Removing the final fc layer
        self.backbone.fc = nn.Identity()

        # Projection Head
        # ResNet50 (2048) -> 256 -> Embedding
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # x shape: (batch, 1, 128, 216)
        
        # We modified conv1 to take 1 channel, so we pass x directly.
        # No need to repeat channels.
        
        # Get features from ResNet (Backbone)
        h = self.backbone(x)
        
        # Project to embedding space (SimCLR Head)
        z = self.projection_head(h)
        
        # Normalize to unit sphere (L2 Normalization)
        # Critical for Cosine Similarity loss
        z = F.normalize(z, p=2, dim=1)
        
        return h, z