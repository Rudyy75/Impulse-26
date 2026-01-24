
import torch
import torch.nn as nn
import torch.nn.functional as F

class EchoFindDecoder(nn.Module):
    """
    Decodes 2048-dim latent vector back into Mel Spectrogram (1, 128, 216).
    Architecture: FC -> 4x4 -> ConvTranspose (Upsampling) -> 128x128 -> Resize to 128x216.
    """
    def __init__(self, input_dim=2048):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Project linear vector to spatial map 4x4
        # 512 channels * 4 * 4 = 8192 parameters
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)
        
        # Upsampling Blocks
        # H_out = (H_in - 1)*stride - 2*padding + kernel_size
        # (4-1)*2 - 2 + 4 = 8. (Doubles every layer)
        
        self.blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        
        # Final adjustment to output channels (1)
        # 128x128 -> 128x128 (1 channel)
        self.final = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, z):
        # z: (B, 2048)
        B = z.size(0)
        
        x = self.fc(z)
        x = x.view(B, 512, 4, 4)
        
        x = self.blocks(x) # -> (B, 16, 128, 128)
        
        x = self.final(x) # -> (B, 1, 128, 128)
        
        # Upsample to Target (128, 216)
        # We need width=216. Height=128 matches (lucky).
        # We preserve Frequency (H=128) but stretch Time (W=128 -> 216)
        x = F.interpolate(x, size=(128, 216), mode='bilinear', align_corners=False)
        
        return x
