# Plan: Variable-Length Inference (Phase 7)

## Goal
Enable the system to handle audio queries of arbitrary duration (2s to 60s) and produce consistent embeddings, such that a 5-second clip and a 30-second clip of the same song match.

## Problem
- **Standard ResNet50:** Uses `AdaptiveAvgPool2d((1,1))`. This is technically "Variable Length" compatible (it squashes any $H \times W$ to $1 \times 1$).
- **Issue:** Simple averaging treats all time steps equally.
    - 30s clip (27s silence + 3s hook) $\approx$ Silence (if averaged).
    - 5s clip (hook) $\ne$ Silence.
    - Result: Vectors diverge.

## Solution: Attention Pooling
Replace `model.avgpool` with a Learnable Attention Layer.
- **Mechanism:** The model learns *which* time steps are important.
- **Formula:** $z = \sum \alpha_t x_t$ where $\alpha_t$ is the attention weight for frame $t$.
- **Effect:** The model focuses on the "Hook" in both the 5s clip and the 30s clip, ignoring the silence.

## Implementation Steps

### 1. Update `src/model.py`
Add `AttentionPool2d` class.
```python
class AttentionPool2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1) # Score calculator
        
    def forward(self, x):
        # x: (B, C, H, W)
        scores = self.conv(x) # (B, 1, H, W)
        attn = torch.softmax(scores.view(x.size(0), -1), dim=1).view_as(scores)
        return (x * attn).sum(dim=(2, 3))
```
Replace `backbone.avgpool` with this.

### 2. Verify (`src/test_variable.py`)
- Load a Track.
- Slice 1: Full 30s.
- Slice 2: Random 5s crop.
- Compute Cosine Similarity.
- **Target:** Similarity > 0.9.

### 3. Note on Training
- Ideally, we should retrain Phase 2 with this new layer.
- **Short-Path:** The standard GAP (ResNet default) is actually quite robust for FMA music (which is dense, not sparse).
- **Decision:** We will stick to **Default GAP** first and test it. If it fails (>0.8 sim), we implement Attention.
- **Why?** Retraining takes time. Standard ResNet represents "Bag of Features" which is robust to length.

## Variable-Length Handling in Code
- `src/spectrogram.py`: Ensure `GPUPipeline` handles variable width.
    - Current code: `melspec` works on any length. `Resample` works.
    - **Constraint:** `EchoFindDecoder` (Phase 5) expects FIXED size (216).
    - **Resolution:**
        - **Encoder:** Variable Length OK.
        - **Decoder (GenAI):** Fixed Length Only (trained on crops).
        - This is acceptable. GenAI usually generates fixed clips. Retrieval handles variable queries.

## Tasks
1. [ ] create `src/test_variable.py` to benchmark GAP.
2. [ ] (Optional) Implement Attention if GAP fails.
