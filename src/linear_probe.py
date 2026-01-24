import torch
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_DIR, METADATA_DIR

def load_metadata(tracks_csv_path):
    """
    Load FMA metadata.
    FMA CSVs have 3 header rows. 
    Row 0: Header (track, album, etc)
    Row 1: Subheader (genre_top, title, etc)
    Row 2: Type (string, int, etc)
    """
    print(f"Loading metadata from {tracks_csv_path}...")
    # Read header row 0 and 1
    tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])
    
    # Filter for 'small' subset
    small = tracks[tracks[('set', 'subset')] == 'small']
    
    # Get Top Genre
    genres = small[('track', 'genre_top')]
    
    print(f"Loaded {len(genres)} tracks with genre labels.")
    return genres

def run_probe():
    print("----------------------------------------------------------------")
    print(" 🎻 PHASE 4: Linear Probe (Genre Classification)")
    print("----------------------------------------------------------------")
    print("Checking if the model learned 'Music' (Genre) or just 'Noise'...")
    
    # 1. Load Embeddings
    db_path = os.path.join(os.path.dirname(DATA_DIR), 'vector_db.pt')
    if not os.path.exists(db_path):
        print("❌ DB not found. Run index_db.py first.")
        return
        
    db = torch.load(db_path, weights_only=False)
    embeddings = db['embeddings']
    if torch.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    else:
        embeddings = np.array(embeddings)
    filenames = db['filenames']
    
    print(f"Loaded {len(embeddings)} embeddings.")
    
    # 2. Extract Track IDs from filenames
    # Filename format: "000123.mp3" -> 123
    track_ids = []
    
    valid_indices = []
    
    for i, fname in enumerate(filenames):
        try:
            # Extract ID from basename (e.g. C:\...\000123.mp3 -> 000123.mp3 -> 123)
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
    if not os.path.exists(meta_path):
        print(f"❌ Metadata not found at {meta_path}")
        return

    genres_series = load_metadata(meta_path)
    
    # 4. Join Labels
    # We only keep tracks that exist in both DB and Metadata
    X = []
    y = []
    
    found_count = 0
    missing_count = 0
    
    for i, tid in enumerate(track_ids):
        if tid in genres_series.index:
            genre = genres_series.loc[tid]
            if pd.isna(genre):
                continue
            X.append(embeddings[i])
            y.append(genre)
            found_count += 1
        else:
            missing_count += 1
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Matched {len(X)} samples. Missing metadata for {missing_count}.")
    
    # 5. Encode Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes ({len(le.classes_)}): {le.classes_}")
    
    # 6. Split (80% Train, 20% Test) - Standard Linear Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, train_size=0.8, random_state=42, stratify=y_enc
    )
    
    print(f"Training on {len(X_train)} samples (10%). Testing on {len(X_test)} samples (90%).")
    
    # 7. Train Linear Classifier
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, y_train)
    
    # 8. Evaluate
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n----------------------------------------------------------------")
    print(f"📊 Accuracy: {acc*100:.2f}%")
    print(f"🎯 F1 Score: {f1*100:.2f}%")
    print("----------------------------------------------------------------")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    if f1 > 0.40:
        print("🏆 SUCCESS: The model learned semantic genre features!")
    else:
        print("⚠️ WEAK: The model struggled to separate genres linearly.")

if __name__ == "__main__":
    run_probe()
