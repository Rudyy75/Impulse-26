# Legacy V1 Architecture (ResNet18)
# This version achieved the 36% Search and 45% F1 Benchmarks.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EchoFindModelV1(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # 1. LOAD BACKBONE (ResNet18 - Faster and less prone to collapse on 8k data)
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # 2. MODIFY INPUT LAYER (1 Channel)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Removing the final fc layer
        self.backbone.fc = nn.Identity()

        # Projection Head (Matches Legacy V1 math)
        # ResNet18 (512) -> 256 -> 128
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        z = F.normalize(z, p=2, dim=1)
        return h, z
