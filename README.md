<div align="center">
  <h1>🎵 EchoFind</h1>
  <p><strong>A retrieval and synthesis engine built using SimCLR (Contrastive Learning)</strong></p>
  <p>Built for the <em>Impulse'26 Hackathon</em></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C.svg)](https://pytorch.org/)
  [![SimCLR](https://img.shields.io/badge/Architecture-SimCLR-brightgreen.svg)]()
</div>

<hr/>

## ✨ Overview

**EchoFind** is an advanced retrieval and synthesis engine powered by **SimCLR (Contrastive Learning)**. Designed to understand complex audio representations, it bridges the gap between raw spectrograms and highly discriminative latent spaces.

> Built modularly, optimized for scale, and engineered for high-fidelity audio search.

---

## 🏗️ Architecture

EchoFind leverages a dual-stage architecture for maximum retrieval accuracy:

- **Backbone**: `ResNet-18` (Provides an optimal balance of feature extraction and speed). 
- **Projector**: A robust 128-dimensional discriminative head (`z`), specifically optimized for retrieval tasks.
- **Learning Objective**: SimCLR Contrastive Loss.

*Note: We initially experimented with a ResNet-50 backbone. While highly scalable, ResNet-18 proved significantly more efficient for our accuracy targets under current training cycle constraints. Detailed experimental logs can be found in [`BENCHMARK_LOGS.txt`](./BENCHMARK_LOGS.txt).*

---

## 🚀 Quick Start & Evaluation

Want to run the acid test and verify our search robustness instantly? 

```bash
# Run the evaluation suite
python -m src.test_acid
```

For final evaluations, `submission.py` acts as the primary entry point, housing the core `AudioEncoder` and `predict_track` methods.

---

## 📊 Benchmarks & Performance

Our models have been rigorously tested and benchmarked:

| Model Version | Architecture | Peak Search Acc | F1 Score | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **V1 (Current)** | ResNet-18 | **36%** | **45%** | Highly stable, optimal retrieval |
| **V2 (Experimental)** | ResNet-50 | ~20% | ~31% | Requires further training cycles |

<details>
<summary><b>🔍 View Proof of Results</b></summary>
Detailed proof, including t-SNE visualizations and confusion matrices, are available in <code>RESULTS_PROOF.md</code> and the <code>docs/results/</code> directory. Full terminal outputs of our high-water benchmarks are recorded in <code>BENCHMARK_LOGS.txt</code>.
</details>

---

## 📂 Repository Structure

```text
├── submission.py       # Primary entry point for evaluation
├── src/                # Modular source code spanning all phases
├── weights/            # Final model weights (best_simclr.pt)
├── notebooks/          # Experimental documentation & insights
└── docs/results/       # Visual proofs, t-SNE plots, and matrices
```

---

## 🗺️ Phase Mapping

EchoFind was developed in structured phases, from core caching to generative morphing:

### Core Pipelines
- **`src/preprocess.py`** ➡️ *Phase 1: High-speed Spectrogram Cache*
- **`src/train.py`** ➡️ *Phase 2: SimCLR Representation Learning*
- **`src/index_db.py`** ➡️ *Phase 3: FAISS Vector DB Generation*
- **`src/eval_phase4.py`** ➡️ *Phase 4: Full Suite Evaluation (F1 + t-SNE)*

### Advanced Extensions
- **`src/synthesis.py`** ➡️ *Ext A: GenAI Latent Audio Morphing*
- **`src/ood.py`** ➡️ *Ext B: Out-of-Distribution Density Estimation*
- **`src/test_variable.py`** ➡️ *Ext C: Variable Length Robustness*

---
<div align="center">
  <p>Project submission for the Impulse'26 Hackathon</p>
</div>
