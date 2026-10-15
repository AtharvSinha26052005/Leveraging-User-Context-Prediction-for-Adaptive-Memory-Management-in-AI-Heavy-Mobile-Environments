import os

files = [
    ("data/merged/train_events.csv",           "Task 4 - Merged Data"),
    ("models_trained/app_embeddings.npy",       "Task 2 - App2Vec"),
    ("models_trained/frequent_patterns.json",   "Task 3 - PrefixSpan"),
    ("models_trained/xgboost_phase2.json",      "Task 4 - Phase2 Model"),
    ("results/phase2_metrics.json",             "Task 4 - Phase2 Metrics"),
    ("results/benchmark_results_phase2.csv",    "Task 4 - Benchmark"),
    ("docs/LITERATURE_SURVEY_20_PAPERS.md",     "Task 5 - 20 Papers"),
    ("aosp_integration/cpp_prototype/inference.cpp", "Task 7 - AOSP C++"),
    ("aosp_integration/docs/AOSP_ARCHITECTURE.md",  "Task 7 - AOSP Docs"),
]
print("=== PROJECT STATE ===")
for f, label in files:
    exists = os.path.isfile(f)
    size = os.path.getsize(f) if exists else 0
    tag = "OK" if exists else "XX"
    print(f"  [{tag}] {label:<35} {f'{size:,} bytes' if exists else 'MISSING'}")
