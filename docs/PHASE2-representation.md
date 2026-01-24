# Phase 2: Representation Learning - Masterclass Documentation

> **Goal:** Build a "Brain" that understands music structure without supervision.
> **Outcome:** A trained `encoder.pth` that maps audio to a semantic vector space.
> **Scope:** Deep Learning Theory $\to$ ResNet Architecture $\to$ Contrastive Loss Calculus.

---

# 1. 🧠 Theory: The "Self-Supervised" Revolution

### 1.1 The Learning Problem
How do biological brains learn concepts?
- **Supervised Learning (The Old Way):** The dataset approach.
    - Input: A picture of a dog.
    - Label: "Dog".
    - Feedback: "You said Cat. Wrong. Adjust weights."
    - **Problem:** Labels are expensive. The world isn't labeled.
- **Unsupervised Learning (Clustering):**
    - Input: 1000 pictures.
    - Output: "These 50 look similar."
    - **Problem:** Doesn't learn *semantic* features, just pixel statistics.
- **Self-Supervised Learning (SSL - The Modern Way):**
    - Input: A picture of a dog.
    - Logic: "If I rotate this picture, it's *still* the same object. If I crop the tail, it's *still* the same object."
    - **Method:** We create our own labels using **Data Augmentation**.

### 1.2 The Fundamental Axiom of SSL (SimCLR)
We rely on the **Invariance Principle**:
> "The Semantic Identity of a signal is invariant to stochastic transformations."

Mathematically:
Let $x$ be an audio snippet.
Let $T(\cdot)$ be a family of transformations (Noise, Reverb, Shift).
We want to learn a function $f(\cdot)$ such that:
$$ f(T_1(x)) \approx f(T_2(x)) $$
AND, for any other audio $y$:
$$ f(x) \neq f(y) $$

### 1.3 The Latent Space (Geometry)
We map every 5-second audio clip to a single point in a 128-dimensional hypersphere.
- **Why a Hypersphere?**
    - We normalize output vectors: $\|z\|_2 = 1$.
    - This creates a **Riemannian Manifold**.
    - The "Distance" between two songs is simply the **Angle** between them (Cosine Similarity).
- **The Ideal Manifold:**
    - **Alignment:** Positive pairs ($x_i, x_j$) should approach the same point.
    - **Uniformity:** All unrelated points should be spread distinctly across the surface (Maximum Entropy).

---

# 2. 🏛️ Architecture: Deep Dive (`src/model.py`)

We employ a **ResNet-18**, a Convolutional Neural Network (CNN) originally designed for images.

### 2.1 The Vanishing Gradient Problem
Deep networks (e.g., 20 layers) are hard to train. The gradient signal ($dL/dw$) gets multiplied by small numbers ($<1$) at each layer during backpropagation.
By Layer 1, the signal is $0.9^{20} \approx 0.12$. The first layers stop learning.

### 2.2 The Residual Solution (ResNet)
He et al. (2015) introduced "Skip Connections".
$$ y = F(x) + x $$
The gradient can flow directly through the $+ x$ path, bypassing the complex $F(x)$ block. This acts like a "gradient superhighway", allowing us to train very deep networks.

### 2.3 Adaptation for Audio
ResNet expects RGB Images: `[Batch, 3, Height, Width]`.
Our Spectrograms are Grayscale: `[Batch, 1, 128, 216]`.

**Surgery Performed:**
1.  **Input Layer Replacement:**
    - Original: `nn.Conv2d(3, 64, kernel=7, stride=2, padding=3)`
    - New: `nn.Conv2d(1, 64, kernel=7, stride=2, padding=3)`
    - **Impact:** We learn filters for "Frequency x Time" features immediately.

2.  **Removal of Global Classification Layer:**
    - Original: `Linear(512, 1000)` (for 1000 ImageNet classes).
    - New: `Identity()` (Pass through raw features).
    - **Impact:** We extract the 512-dimensional "Thought Vector" ($h$) instead of a class probability.

### 2.4 The Projection Head (The "SimCLR Trick")
Hinton et al. discovered that training directly on the representation layer ($h$) hurts performance.
**Why?** The loss function forces invariance. Some variability is *good* (e.g., exact pitch or rotation). If we force $h$ to be invariant, we lose that info.
**Solution:**
- We keep $h$ (512-dim).
- We project it to $z$ (128-dim) using a Multi-Layer Perceptron (MLP).
- We train the loss on $z$.
- After training, we **delete** the MLP and use $h$ for search.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EchoFindModel(nn.Module):
    def __init__(self, embedding_dim = 128):
        # Parent Class Initialization (Vital for PyTorch)
        super().__init__()
        
        # 1. LOAD BACKBONE
        # We use pre-trained ImageNet weights. 
        # "Transfer Learning" - The model already knows what "Lines" and "Curves" are.
        # This speeds up convergence by 10x compared to random initialization.
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # 2. MODIFY INPUT LAYER (1 Channel)
        # We replace the first convolution.
        # kernel_size=7: Receptive field of 7x7 pixels. Good for detecting broad strokes.
        # stride=2: Downsamples image by 2x immediately (Efficiency).
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 3. MODIFY OUTPUT LAYER (Identity)
        # The 'fc' (Fully Connected) layer usually outputs probabilities.
        # We delete it so the forward pass returns the features *before* classification.
        self.backbone.fc = nn.Identity()

        # 4. PROJECTION HEAD (The MLP)
        # Architecture: Linear -> ReLU -> Linear.
        # Hidden Dimension: 256.
        # Output Dimension: 128 (Paper recommends 128 for loss calculation).
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        """
        The Forward Pass Logic.
        Input: Spectrogram [B, 1, 128, 216]
        """
        # A. Feature Extraction
        # h shape: [B, 512]
        # This represents the abstract concept of the audio.
        # It is robust to noise but retains rich detail.
        h = self.backbone(x)

        # B. Projection
        # z shape: [B, 128]
        # This is a compressed, highly-invariant version for the Loss function.
        z = self.projection_head(h)
        
        # C. Normalization (CRITICAL)
        # We force vectors to length 1.
        # Without this, vectors could grow infinitely large to minimize loss.
        # With this, only the Direction matters.
        z = F.normalize(z, p=2, dim=1)
        
        return h, z
```

---

# 3. 📉 The Math of Loss (NT-Xent) (`src/losses.py`)

**Normalized Temperature-scaled Cross Entropy Loss.**
This is the heart of Contrastive Learning.

### 3.1 The Energy Landscape
Imagine every song is a magnet.
- **Positive Pair ( $z_i, z_j$ ):** Same origin. We want **Attraction**.
- **Negative Pair ( $z_i, z_k$ ):** Different origin. We want **Repulsion**.

### 3.2 The Formula Breakdown
For a given positive pair $(i, j)$, the loss is:

$$ \ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)} $$

Let's dissect this monster:

1.  **Cosine Similarity $\text{sim}(u, v) = u^T v / \|u\|\|v\|$.**
    - Since vectors are normalized, this is just the Dot Product.
    - Range: $[-1, 1]$. 1 is identical, 0 is orthogonal, -1 is opposite.

2.  **Temperature Scaling ($\tau$):**
    - Let $\tau = 0.1$.
    - Why? The Softmax operator is "soft". It assigns probabilities to everything.
    - Dividing by small $\tau$ makes the distribution "Harder".
    - Example:
        - Sim 0.9 vs 0.8. Softmax: $e^{0.9} \approx 2.4$, $e^{0.8} \approx 2.2$. (Close).
        - With $\tau=0.1$: $e^9 \approx 8103$, $e^8 \approx 2980$. (Huge difference!).
    - **Effect:** The model pays EXTREME attention to the hardest negatives. It ignores easy negatives.

3.  **The Numerator (Positives):**
    - We maximize $e^{\text{sim}}$. This pushes sim towards 1.0.

4.  **The Denominator (Negatives):**
    - We minimize the sum of $e^{\text{sim}}$ for all other 126 examples in the batch.
    - This pushes their sim towards 0.0 or -1.0.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_Xent_loss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1: Batch of View 1 [64, 128]
        z2: Batch of View 2 [64, 128]
        """
        batch_size = z1.shape[0]
        
        # 1. Concatenate into one giant batch [128, 128]
        # Indices 0-63 are View1.
        # Indices 64-127 are View2.
        # Correct Pairs are: (0,64), (1,65), ... (i, i+batch_size).
        z = torch.cat([z1, z2], dim=0) 
        
        # 2. Compute Similarity Matrix (All-vs-All)
        # Matrix Multiplication is efficient.
        # Shape: [128, 128].
        # entry [i, j] is similarity between vector i and vector j.
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        
        # 3. Apply Temperature
        sim_matrix = sim_matrix / self.temperature 
        
        # 4. Mask Self-Similarity
        # The main diagonal (0,0), (1,1) is always 1.0.
        # We don't want the model to learn "A is A". That's trivial.
        # We set diagonal to -infinity (which becomes 0 in Softmax).
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        
        # 5. Define Targets
        # For row 0, the target is column 64.
        # For row 64, the target is column 0.
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size), # 64...127
            torch.arange(0, batch_size)             # 0...63
        ], dim=0).to(z.device)
        
        # 6. Cross Entropy
        # PyTorch CrossEntropy computes Softmax + NLLLoss efficiently.
        return F.cross_entropy(sim_matrix, labels)
```

---

# 4. 🚂 The Training Loop (`src/train.py`)

The engine room where integration happens.

```python
def train():
    # 1. INITIALIZATION
    # Detect GPU (CUDA). Tensors on CUDA run 50-100x faster than CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. DATA LOADING
    # drop_last=True:
    # If dataset size is 8000 and batch is 64, 8000 % 64 != 0.
    # The last batch might have, say, 12 items.
    # BatchNorm layers often crash or produce NaN if Batch Size=1.
    # SimCLR relies on large batches for negatives. A small batch is "easier" (less negatives).
    # We drop the last incomplete batch for stability.
    dataset = FMAContrastiveDataset(SPECTROGRAM_DIR)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    # 3. MODEL SETUP
    model = EchoFindModel().to(device)
    model.train() # Enable Dropout and BatchNorm tracking.
    
    # 4. OPTIMIZER
    # Why Adam?
    # SGD (Stochastic Gradient Descent) is simple but gets stuck in saddle points.
    # Adam (Adaptive Moments) maintains per-parameter learning rates.
    # LR=3e-4 is the "Karpathy Constant" - widely cited as best default for Adam.
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # 5. LOSS FUNCTION
    criterion = NT_Xent_loss(temperature=0.1)
    
    # 6. EPOCH LOOP
    for epoch in range(50):
        total_loss = 0
        
        # Use tqdm for a progress bar (sanity check for developer).
        pbar = tqdm(dataloader)
        
        for view1, view2 in pbar:
            # Transfer Memory: RAM -> VRAM (Expensive Op)
            view1, view2 = view1.to(device), view2.to(device)
            
            # Forward Pass
            # We ignore 'h' output during training, we only need 'z' for loss.
            _, z1 = model(view1)
            _, z2 = model(view2)
            
            # Loss Calculation
            loss = criterion(z1, z2)
            
            # Backward Pass
            optimizer.zero_grad() # Reset accumulators
            loss.backward()       # Compute Gradients
            optimizer.step()      # Update Weights
            
            total_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        # Checkpointing
        # We save ONLY the backbone.
        # The projection head is useful for training but not for search.
        # The Backbone contains the general audio understanding.
        torch.save(model.backbone.state_dict(), 'weights/encoder.pth')
```

---

# 5. 📊 Training Dynamics & Convergence

What happens during training?

- **Epoch 0-5 (Chaos Phase):**
    - Loss starts high (~4.0 - 5.0).
    - Model predicts randomly.
    - Gradients are large. Weights swing wildly.
    - Filters look like random noise.

- **Epoch 5-20 (Pattern Phase):**
    - Loss drops rapidly (~2.0).
    - Filters start looking like "Gabor Filters" (Edge detectors).
    - Model learns to distinguish "Silence" vs "Noise" vs "Music".
    - It learns simple frequency bands (Bass vs Treble).

- **Epoch 20-40 (Semantic Phase):**
    - Loss slowing down (~0.5 - 1.0).
    - Model learns chords, rhythm patterns, and timbre (Piano vs Guitar).
    - This is where the magic happens.

- **Epoch 40-50 (Fine Tuning):**
    - Loss plateaus (~0.1 - 0.2).
    - Model refines boundaries between similar genres (Rock vs Metal).

### Why Loss never reaches 0?
In Supervised Learning, loss can go to 0 (perfect memorization).
In Contrastive Learning, loss theoretically can go very low, but due to augmentation overlaps ("View 1" might actually look like "Negative 55"), there's an irreducible error floor. This prevents overfitting!

---

# 6. Summary

Phase 2 is where the "Intelligence" is born. We used:
1.  **ResNet-18 Backbone:** To capture deep hierarchical features.
2.  **Projection Head:** To absorb task-specific invariance.
3.  **NT-Xent Loss:** To push positives together and negatives apart.
4.  **Adam Optimizer:** To navigate the 20-million-parameter landscape.

The result is a `.pth` file that contains a compressed understanding of musical structure.
