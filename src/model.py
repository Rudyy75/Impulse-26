import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .config import N_MELS

class EchoFindModel(nn.Module):
    def __init__(self, embedding_dim = 128):
        super().__init__()

        # 1. LOAD BACKBONE (ResNet18)
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # 2. MODIFY INPUT LAYER (1 Channel)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Removing the final fc layer
        self.backbone.fc = nn.Identity()

        # Projection Head
        # ResNet18 outputs 512 features (instead of 2048 in ResNet50)
        # 512 -> 256 -> Embedding
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # x shape: (batch, 1, 128, 216)
        
        # Get features from ResNet (Backbone)
        h = self.backbone(x)
        
        # Project to embedding space (SimCLR Head)
        z = self.projection_head(h)
        
        # Normalize to unit sphere (L2 Normalization)
        # Critical for Cosine Similarity loss
        z = F.normalize(z, p=2, dim=1)
        
        return h, z