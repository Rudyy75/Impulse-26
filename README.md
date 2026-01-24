# 🎵 EchoFind: Submission Guide

EchoFind is a retrieval and synthesis engine built using SimCLR (Contrastive Learning) for the Impulse'26 Hackathon.

## 📂 Submission Structure
- **`submission.py`**: The entry point for evaluation. Contains `AudioEncoder` and `predict_track`.
- **`src/`**: Full modular source code for all 7 Phases.
- **`weights/`**: Final model weights (`best_simclr.pt`).
- **`notebooks/`**: Experimental documentation (see `notebooks/README.md`).
- **`docs/results/`**: Visualization proof (t-SNE, Matrices).

## 🧪 Evaluation & Benchmarks
We have achieved two distinct performance tiers during development:

- **Legacy V1 (ResNet18):** Peak Search Accuracy: **36%** | F1 Score: **45%**.
- **Current V2 (ResNet50):** Highly modular and scalable, but currently requires more training cycles to reach peak discriminative power (Last recorded F1: ~31%).

> [!IMPORTANT]
> For proof of the high-water benchmarks (36%/45%), please refer to **`src/BENCHMARK_LOGS.txt`**, which contains the full terminal outputs from those runs.

## 🚀 Quick Evaluation
Run the Acid test to verify search robustness (V2):
```powershell
python -m src.test_acid
```

## 🧠 Model Architecture
- **V2 Backbone:** ResNet50 (1-Channel input).
- **Projector:** 128-dim discriminative head (`z`) optimized for retrieval.
- **Optimizer:** SimCLR Contrastive Objective.

## 🛠️ Phase Mapping
| File | Phase |
| :--- | :--- |
| `src/preprocess.py` | Phase 1: High-speed Spectrogram Cache |
| `src/train.py` | Phase 2: SimCLR Representation Learning |
| `src/index_db.py` | Phase 3: FAISS Vector DB Generation |
| `src/eval_phase4.py` | Phase 4: Full Suite Evaluation (F1 + t-SNE) |
| `src/synthesis.py` | Ext A: GenAI Latent Audio Morphing |
| `src/ood.py` | Ext B: Out-of-Distribution Density Estimation |
| `src/test_variable.py`| Ext C: Variable Length Robustness |
