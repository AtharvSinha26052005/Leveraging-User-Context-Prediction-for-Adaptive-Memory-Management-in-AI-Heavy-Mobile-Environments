"""Test script for App2Vec training and verification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.prediction.app2vec import App2Vec

# Load multiuser data
df = pd.read_csv("data/multiuser/multiuser_events.csv")
print(f"Loaded {len(df)} events, {df['current_app'].nunique()} apps")

# Train App2Vec
app2vec = App2Vec(
    embedding_dim=64, pca_dim=8,
    walk_length=30, num_walks=200,
    epochs=50, seed=42
)
embeddings = app2vec.train(df)

# Save
os.makedirs("models_trained", exist_ok=True)
app2vec.save("models_trained")

# Show most similar apps
for target in ["Instagram", "Gmail", "BGMI", "WhatsApp", "ChatGPT"]:
    sims = app2vec.most_similar(target, top_k=5)
    print(f"\n  Top-5 Most Similar to {target}:")
    for app, sim in sims:
        print(f"    {app:<15} {sim:.3f}")

# Transition stats
stats = app2vec.get_transition_stats()
print("\n  Transition Graph Stats (Top 10):")
print(stats.head(10).to_string(index=False))

print("\n  App2Vec training COMPLETE!")
