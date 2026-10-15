"""
run_all.py — One-click demo: Generate → Train → Benchmark → Print Results
Run this in front of the panel to show the full pipeline working.

Usage: python scripts/run_all.py
"""
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def main():
    print("=" * 80)
    print("  ADAPTIVE MEMORY MANAGEMENT — FULL PIPELINE DEMO")
    print("  Leveraging User Context Prediction for AI-Heavy Mobile Environments")
    print("=" * 80)
    total_start = time.time()

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Generate Synthetic Data
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 1/4: Generating Synthetic App-Switch Events")
    print("─" * 70)

    from src.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    events_per_persona = 10000  # 10K per persona = 50K total (matches README spec)
    df = gen.generate_all(events_per_persona=events_per_persona)

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_events.csv", index=False)
    print(f"\n  ✓ Generated {len(df)} events across 5 personas")
    print(f"  ✓ Saved to data/synthetic_events.csv")
    print(f"  ✓ Unique apps: {df['current_app'].nunique()}")
    print(f"  ✓ Date range: {df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Train ML Models
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 2/4: Training ML Models (Markov + XGBoost)")
    print("─" * 70)

    from src.prediction.xgboost_predictor import XGBoostPredictor
    from sklearn.model_selection import train_test_split

    # Train/test split (temporal — train on first 70%, test on last 30%)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\n  Train set: {len(train_df)} events")
    print(f"  Test set:  {len(test_df)} events")

    # Train XGBoost + Markov
    predictor = XGBoostPredictor(n_estimators=200, max_depth=6)
    metrics = predictor.fit(train_df, val_df=test_df)

    print(f"\n  ✓ XGBoost Training Results:")
    print(f"    • Training accuracy:  {metrics['train_accuracy']:.1%}")
    print(f"    • 5-fold CV accuracy: {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}")
    print(f"    • Top-3 accuracy:     {metrics['top3_accuracy']:.1%}")
    if 'val_accuracy' in metrics:
        print(f"    • Validation top-1:   {metrics['val_accuracy']:.1%}")
        print(f"    • Validation top-3:   {metrics['val_top3_accuracy']:.1%}")
    print(f"    • Classes (apps):     {metrics['n_classes']}")
    print(f"    • Features:           {metrics['n_features']}")

    # Feature importance
    fi = predictor.get_feature_importance()
    print(f"\n  ✓ Top 10 Feature Importances:")
    for name, imp in fi[:10]:
        bar = "█" * int(imp * 200)
        print(f"    {name:<25} {imp:.4f}  {bar}")

    # Save models
    os.makedirs("models_trained", exist_ok=True)
    predictor.save(
        "models_trained/xgboost_model.json",
        "models_trained/feature_engine.pkl",
        "models_trained/markov_matrix.pkl",
    )
    print(f"\n  ✓ Models saved to models_trained/")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Run Benchmarks
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 3/4: Running Benchmark Simulations (4 Policies)")
    print("─" * 70)

    from src.simulation.benchmark_runner import BenchmarkRunner

    runner = BenchmarkRunner(verbose=True)

    # Quick demo: run all 4 policies on a subset (2000 events per persona)
    demo_events = {}
    for persona in df["persona"].unique():
        persona_df = df[df["persona"] == persona].head(2000).copy()
        demo_events[persona] = persona_df

    results_df = runner.run_all_policies(
        events_by_persona=demo_events,
        predictor=predictor,
        ram_configs=[4096, 6144, 8192],  # 3 RAM configs: 4GB, 6GB, 8GB
        n_runs=2,  # 2 runs per config for variance
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/benchmark_results.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Print Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 4/4: Results Summary")
    print("─" * 70)

    runner.print_comparison_table(results_df)

    # Per-persona breakdown
    print("\n  Per-Persona Breakdown (6 GB RAM):")
    print(f"  {'Persona':<15} {'NoOpt':>10} {'LRU':>10} {'StaticPre':>10} {'Proposed':>10}")
    print("  " + "-" * 60)
    ram_df = results_df[results_df["ram_mb"] == 6144]
    for persona in sorted(ram_df["persona"].unique()):
        p_df = ram_df[ram_df["persona"] == persona]
        vals = {}
        for _, row in p_df.iterrows():
            vals[row["policy"]] = f"{row['cache_hit_rate']:.1%}"
        print(f"  {persona:<15} "
              f"{vals.get('noopt', 'N/A'):>10} "
              f"{vals.get('lru', 'N/A'):>10} "
              f"{vals.get('static_preload', 'N/A'):>10} "
              f"{vals.get('proposed', 'N/A'):>10}")

    total_elapsed = time.time() - total_start
    print(f"\n  Total execution time: {total_elapsed:.1f}s")
    print(f"  Results saved to: results/benchmark_results.csv")
    print(f"\n  Next: Run 'streamlit run dashboard/app.py' to see the interactive dashboard")
    print("=" * 80)


if __name__ == "__main__":
    main()
