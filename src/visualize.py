import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_DIR, METADATA_DIR
from src.linear_probe import load_metadata # Reuse

def run_visualization():
    print("----------------------------------------------------------------")
    print(" 🎨 PHASE 4: Latent Space Visualization (t-SNE)")
    print("----------------------------------------------------------------")
    
    # 1. Load Embeddings
    db_path = os.path.join(os.path.dirname(DATA_DIR), 'vector_db.pt')
    if not os.path.exists(db_path):
        print("❌ DB not found.")
        return
        
    db = torch.load(db_path, weights_only=False)
    embeddings = db['embeddings']
    if torch.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    else:
        embeddings = np.array(embeddings)
    filenames = db['filenames']
    
    # 2. Extract IDs
    track_ids = []
    valid_indices = []
    for i, fname in enumerate(filenames):
        try:
            basename = os.path.basename(fname)
            tid = int(os.path.splitext(basename)[0])
            track_ids.append(tid)
            valid_indices.append(i)
        except ValueError:
            continue
            
    embeddings = embeddings[valid_indices]
    track_ids = np.array(track_ids)
    
    # 3. Load Metadata
    meta_path = os.path.join(METADATA_DIR, 'tracks.csv')
    genres_series = load_metadata(meta_path)
    
    # 4. Join
    X = []
    y = []
    
    for i, tid in enumerate(track_ids):
        if tid in genres_series.index:
            genre = genres_series.loc[tid]
            if pd.isna(genre): continue
            X.append(embeddings[i])
            y.append(genre)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Plotting {len(X)} points...")
    
    # 5. Encode Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    genre_names = le.classes_
    
    # 6. Run t-SNE
    print("Running t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 7. Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_embedded[:,0], 
        y=X_embedded[:,1],
        hue=y,
        palette="tab10",
        alpha=0.7,
        s=40
    )
    plt.title("Self-Supervised Embeddings (t-SNE) colored by Genre", fontsize=16)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Save
    out_path = os.path.join('docs', 'tsne_genre.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✅ Plot saved to {out_path}")

if __name__ == "__main__":
    run_visualization()
