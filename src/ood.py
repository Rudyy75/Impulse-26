
import numpy as np
import torch
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from src.config import DATA_DIR, WEIGHTS_DIR

OOD_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'ood_model.pkl')

class OODDetector:
    def __init__(self, k=50):
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
        self.threshold = None
        self.fitted = False

    def fit(self, embeddings):
        """
        Fit KNN on Music Embeddings and determine OOD Threshold.
        Threshold = 95th Percentile of internal distances (Distance to k-th neighbor).
        """
        print(f"🧐 Fitting OOD Detector on {len(embeddings)} vectors...")
        
        # Convert to numpy if tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
            
        self.knn.fit(embeddings)
        
        # Determine internal density statistics
        distances, _ = self.knn.kneighbors(embeddings)
        
        # We use the mean distance to the k neighbors as the anomaly score
        mean_dists = distances.mean(axis=1)
        
        # Threshold: 95th percentile (Reject bottom 5% outliers of Training Set)
        # Or maybe 99th percentile? Let's be safe with 95.
        self.threshold = np.percentile(mean_dists, 95)
        self.fitted = True
        
        print(f"✅ OOD Fitted. Threshold (Mean Cosine Dist): {self.threshold:.4f}")
        print(f"   (0.0 = Clone, 1.0 = Orthogonal, 2.0 = Opposite)")

    def score(self, query_vec):
        """
        Returns (score, is_ood)
        score: Mean distance to k nearest neighbors. Higher = More Anomalous.
        """
        if not self.fitted:
             raise ValueError("Model not fitted!")
             
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.cpu().numpy()
            
        if len(query_vec.shape) == 1:
            query_vec = query_vec.reshape(1, -1)
            
        distances, _ = self.knn.kneighbors(query_vec)
        mean_dist = distances.mean(axis=1)[0]
        
        is_ood = mean_dist > self.threshold
        return mean_dist, is_ood

    def save(self):
        with open(OOD_MODEL_PATH, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Saved OOD Model to {OOD_MODEL_PATH}")

    @staticmethod
    def load():
        if os.path.exists(OOD_MODEL_PATH):
            with open(OOD_MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        return None

def train_ood():
    print("----------------------------------------------------------------")
    print(" 🚧 PHASE 6: Training OOD Detector (Gatekeeper)")
    print("----------------------------------------------------------------")
    
    # 1. Load Vector DB
    db_path = os.path.join(os.path.dirname(DATA_DIR), 'vector_db.pt')
    if not os.path.exists(db_path):
        print("❌ Vector DB not found. Run src/index_db.py first.")
        return
        
    db = torch.load(db_path, weights_only=False)
    embeddings = db['embeddings'] # (N, 2048) or (N, 512) depending on what we indexed
    
    # Note: OOD must run on the SAME features as Query.
    # index_db.py saves normalized vectors? Yes.
    
    # 2. Fit
    detector = OODDetector(k=50) # 50 neighbors for robustness
    detector.fit(embeddings)
    
    # 3. Save
    detector.save()

if __name__ == "__main__":
    train_ood()
