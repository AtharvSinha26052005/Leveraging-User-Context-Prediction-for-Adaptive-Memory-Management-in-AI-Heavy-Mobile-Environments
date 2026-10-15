"""
run_phase2.py - Phase 2 Pipeline: Merged Data + App2Vec + PrefixSpan + XGBoost

Steps:
  1. Load merged dataset (181K events)
  2. Train App2Vec embeddings from transition graph
  3. Mine sequential patterns via PrefixSpan
  4. Train XGBoost with enriched features (19 base + 16 App2Vec + 11 PrefixSpan = 46)
  5. Benchmark 4 policies
  6. Print results

Usage: python scripts/run_phase2.py
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from xgboost import XGBClassifier

from src.prediction.feature_engine import FeatureEngine
from src.prediction.app2vec import App2Vec
from src.prediction.sequence_miner import SequenceMiner
from src.prediction.markov_predictor import MarkovPredictor


def main():
    print("=" * 72)
    print("  PHASE 2 PIPELINE - App2Vec + PrefixSpan + Merged Dataset")
    print("  Adaptive Memory Management - Enhanced Prediction Engine")
    print("=" * 72)
    total_start = time.time()

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Load merged data
    # ─────────────────────────────────────────────────────────────
    print("\n--- STEP 1/5: Loading Merged Dataset ---")

    train_path = "data/merged/train_events.csv"
    test_path = "data/merged/test_events.csv"

    if not os.path.isfile(train_path):
        print("  Merged data not found. Running merge script first...")
        os.system("python scripts/merge_data.py")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"  Train: {len(train_df):,} events | Test: {len(test_df):,} events")
    print(f"  Train apps: {train_df['current_app'].nunique()} | "
          f"Test apps: {test_df['current_app'].nunique()}")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: App2Vec embeddings
    # ─────────────────────────────────────────────────────────────
    print("\n--- STEP 2/5: Training App2Vec Embeddings ---")
    t0 = time.time()

    app2vec = App2Vec(
        embedding_dim=64, pca_dim=8,
        walk_length=30, num_walks=200,
        epochs=50, seed=42,
    )
    app2vec.train(train_df)
    os.makedirs("models_trained", exist_ok=True)
    app2vec.save("models_trained")
    print(f"  App2Vec done in {time.time()-t0:.1f}s")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: PrefixSpan pattern mining
    # ─────────────────────────────────────────────────────────────
    print("\n--- STEP 3/5: Mining Sequential Patterns (PrefixSpan) ---")
    t0 = time.time()

    # Use 20K sample for pattern mining (speed)
    sample_for_mining = train_df.sample(n=min(20000, len(train_df)), random_state=42)
    miner = SequenceMiner(
        min_support=0.03,
        max_pattern_length=4,
        top_k_patterns=10,
        session_gap_minutes=30,
    )
    miner.mine_patterns(sample_for_mining)
    miner.save("models_trained/frequent_patterns.json")
    print(f"  PrefixSpan done in {time.time()-t0:.1f}s")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Feature engineering + XGBoost training
    # ─────────────────────────────────────────────────────────────
    print("\n--- STEP 4/5: Training Phase 2 XGBoost (Extended Features) ---")
    t0 = time.time()

    # Filter to rows where next_app is known
    train_clean = train_df[train_df["next_app"].notna() &
                           (train_df["next_app"] != "")].copy()
    test_clean = test_df[test_df["next_app"].notna() &
                         (test_df["next_app"] != "")].copy()

    # Build base features — instantiate engine, fit, get train targets (populates target_encoder)
    engine = FeatureEngine()
    engine.fit(train_clean)
    y_train = engine.get_target(train_clean)   # This populates label_encoders["target_encoder"]

    # Now filter test to only labels seen during training
    known_apps = set(engine.label_encoders["target_encoder"].classes_)
    test_clean = test_clean[test_clean["next_app"].isin(known_apps)].copy()
    test_clean = test_clean[test_clean["current_app"].isin(known_apps)].copy()

    X_train_base = engine.transform(train_clean)
    X_test_base = engine.transform(test_clean)
    y_test = engine.get_target(test_clean)

    # Build App2Vec features (8 per app x 2 apps = 16) — vectorized lookup
    def get_app2vec_features(df: pd.DataFrame) -> np.ndarray:
        dim = app2vec.pca_dim
        # Build app->index map and embedding matrix for fast numpy indexing
        app_list = app2vec.app_names
        emb_matrix = np.array([app2vec.get_embedding(a) for a in app_list], dtype=np.float32)
        app_to_idx = {a: i for i, a in enumerate(app_list)}
        default_idx = 0  # fallback for unknown apps

        cur_indices  = df["current_app"].astype(str).map(
            lambda x: app_to_idx.get(x, default_idx)).values
        prev_indices = df["prev_app_1"].fillna("").astype(str).map(
            lambda x: app_to_idx.get(x, default_idx)).values

        X_cur  = emb_matrix[cur_indices]   # (n, 8)
        X_prev = emb_matrix[prev_indices]  # (n, 8)
        return np.hstack([X_cur, X_prev])  # (n, 16)

    print("  Computing App2Vec features for train (vectorized)...")
    X_train_emb = get_app2vec_features(train_clean)
    print("  Computing App2Vec features for test (vectorized)...")
    X_test_emb = get_app2vec_features(test_clean)

    # Build PrefixSpan features (11)
    print("  Computing PrefixSpan features for train...")
    X_train_pat = miner.create_features(train_clean)
    print("  Computing PrefixSpan features for test...")
    X_test_pat = miner.create_features(test_clean)

    # Concatenate all features
    X_train = np.hstack([X_train_base, X_train_emb, X_train_pat])
    X_test = np.hstack([X_test_base, X_test_emb, X_test_pat])

    total_features = X_train.shape[1]
    print(f"\n  Feature dimensions:")
    print(f"    Base (FeatureEngine): {X_train_base.shape[1]}")
    print(f"    App2Vec embeddings:   {X_train_emb.shape[1]}")
    print(f"    PrefixSpan patterns:  {X_train_pat.shape[1]}")
    print(f"    TOTAL:                {total_features}")

    # Train XGBoost
    print(f"\n  Training XGBoost on {len(train_clean):,} events...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    y_prob = model.predict_proba(X_test)
    n_cls = y_prob.shape[1]
    top3_acc = top_k_accuracy_score(
        y_test, y_prob, k=min(3, n_cls),
        labels=list(range(n_cls))
    )

    print(f"\n  Phase 2 XGBoost Results:")
    print(f"    Test Top-1 accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"    Test Top-3 accuracy:  {top3_acc:.4f} ({top3_acc*100:.1f}%)")
    print(f"    Classes (apps):       {n_cls}")
    print(f"    Features used:        {total_features}")
    print(f"    Training time:        {time.time()-t0:.1f}s")

    # Save Phase 2 model
    model.save_model("models_trained/xgboost_phase2.json")
    engine.save("models_trained/feature_engine_phase2.pkl")

    # Save metrics
    metrics = {
        "phase": 2,
        "train_events": len(train_clean),
        "test_events": len(test_clean),
        "total_features": total_features,
        "base_features": int(X_train_base.shape[1]),
        "app2vec_features": int(X_train_emb.shape[1]),
        "prefixspan_features": int(X_train_pat.shape[1]),
        "test_top1_accuracy": round(test_acc, 4),
        "test_top3_accuracy": round(top3_acc, 4),
        "n_classes": int(n_cls),
    }
    with open("results/phase2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Benchmark 4 policies on multiuser data
    # ─────────────────────────────────────────────────────────────
    print("\n--- STEP 5/5: Running Benchmark Simulations ---")
    print("  (Using original benchmark runner on multiuser data)")

    multi_df = pd.read_csv("data/multiuser/multiuser_events.csv")
    from src.simulation.benchmark_runner import BenchmarkRunner
    from src.prediction.xgboost_predictor import XGBoostPredictor

    # Train predictor on original synthetic data for policy benchmarking
    synth_df = pd.read_csv("data/synthetic_events.csv")
    split = int(len(synth_df) * 0.7)
    predictor = XGBoostPredictor(n_estimators=200, max_depth=6)
    predictor.fit(synth_df.iloc[:split], val_df=synth_df.iloc[split:])

    runner = BenchmarkRunner(verbose=False)
    demo_events = {}
    for persona in multi_df["persona"].unique():
        demo_events[persona] = multi_df[multi_df["persona"] == persona].head(2000).copy()

    os.makedirs("results", exist_ok=True)
    results_df = runner.run_all_policies(
        events_by_persona=demo_events,
        predictor=predictor,
        ram_configs=[4096, 6144, 8192],
        n_runs=2,
    )
    results_df.to_csv("results/benchmark_results_phase2.csv", index=False)
    runner.print_comparison_table(results_df)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 72}")
    print(f"  Phase 2 pipeline complete in {total_elapsed:.1f}s")
    print(f"  Phase 2 metrics: results/phase2_metrics.json")
    print(f"  Benchmark:       results/benchmark_results_phase2.csv")
    print(f"{'=' * 72}")
    return metrics


if __name__ == "__main__":
    main()
