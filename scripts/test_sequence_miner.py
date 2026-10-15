"""Test PrefixSpan mining on multiuser data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.prediction.sequence_miner import SequenceMiner

# Load a sample (20K events sufficient for pattern mining)
df = pd.read_csv("data/multiuser/multiuser_events.csv")
sample = df.sample(n=min(20000, len(df)), random_state=42)
print(f"Using {len(sample):,} events for pattern mining")

miner = SequenceMiner(
    min_support=0.03,
    max_pattern_length=4,
    top_k_patterns=10,
    session_gap_minutes=30,
)

patterns = miner.mine_patterns(sample)

# Save patterns
os.makedirs("models_trained", exist_ok=True)
miner.save("models_trained/frequent_patterns.json")

# Show pattern summary
print("\n  Pattern Summary:")
print(miner.get_pattern_summary().to_string(index=False))

# Test feature creation on small sample
feat_sample = sample.head(100)
X_patterns = miner.create_features(feat_sample)
print(f"\n  Pattern features shape: {X_patterns.shape}")
print(f"  Feature names: {miner.get_feature_names()}")
print(f"  Non-zero pattern matches: {(X_patterns[:, :-1] > 0).sum()}")
print(f"  Avg confidence score: {X_patterns[:, -1].mean():.3f}")
print("\n  PrefixSpan mining COMPLETE!")
