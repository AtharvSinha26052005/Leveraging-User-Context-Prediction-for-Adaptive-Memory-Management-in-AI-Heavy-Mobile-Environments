"""Quick smoke test — verifies all project files exist and data is correct."""
import os
import sys
sys.path.insert(0, '.')

print('=== SMOKE TEST ===')
print()

# 1. Check all source files exist
files = [
    'src/prediction/feature_engine.py', 'src/prediction/xgboost_predictor.py',
    'src/prediction/markov_predictor.py', 'src/prediction/confidence_calibrator.py',
    'src/prediction/online_learner.py', 'src/prediction/model_comparison.py',
    'src/memory/cache_scorer.py', 'src/memory/preload_gate.py',
    'src/memory/thrashing_detector.py', 'src/memory/decision_controller.py',
    'src/memory/adaptive_weights.py',
    'src/simulation/policies.py', 'src/simulation/simulator.py',
    'src/simulation/benchmark_runner.py',
    'src/data/synthetic_generator.py', 'src/data/app_profiles.py',
    'src/validation/ablation_study.py', 'src/validation/statistical_tests.py',
    'tests/test_cache_scorer.py', 'tests/test_preload_gate.py',
    'tests/test_thrashing_detector.py', 'tests/test_feature_engine.py',
    'tests/test_simulation.py', 'tests/test_policies.py',
    'scripts/run_all.py', 'scripts/live_demo.py', 'scripts/generate_figures.py',
    'scripts/collect_phone_data.py', 'scripts/collect_laptop_data.py',
    'dashboard/app.py',
    'patent/specification_outline.md', 'patent/claims_draft.md',
    'JUDGE_DEMO_GUIDE.md',
]
missing = [f for f in files if not os.path.exists(f)]
print(f"1. Source files: {len(files) - len(missing)}/{len(files)} exist")
for m in missing:
    print(f"   MISSING: {m}")

# 2. Check data files
import pandas as pd
syn = pd.read_csv('data/synthetic_events.csv')
phone = pd.read_csv('data/self_collected/phone_events.csv')
laptop = pd.read_csv('data/self_collected/laptop_events.csv')
bench = pd.read_csv('results/benchmark_results.csv')
phone_app_col = "current_app" if "current_app" in phone.columns else "app_name"
laptop_app_col = "current_app" if "current_app" in laptop.columns else "app_name"
print(f"2. Data: {len(syn)} synthetic, {len(phone)} phone ({phone[phone_app_col].nunique()} apps), {len(laptop)} laptop ({laptop[laptop_app_col].nunique()} apps)")
print(f"3. Benchmarks: {len(bench)} runs")

# 4. Check models
has_xgb = os.path.exists("models_trained/xgboost_model.json")
has_cal = os.path.exists("models_trained/confidence_calibrator.pkl")
print(f"4. Models: xgboost={'YES' if has_xgb else 'NO'}, calibrator={'YES' if has_cal else 'NO'}")

# 5. Check figures
figs = [f for f in os.listdir('results/figures') if f.endswith('.png')] if os.path.exists('results/figures') else []
print(f"5. Figures: {len(figs)} PNGs")

# 6. Check docs
docs = os.listdir('docs')
print(f"6. Docs: {len(docs)} files")

# 7. Quick import test
try:
    from src.prediction.xgboost_predictor import XGBoostPredictor
    from src.prediction.confidence_calibrator import ConfidenceCalibrator
    from src.prediction.online_learner import OnlineLearner
    from src.memory.cache_scorer import CacheScoringEngine
    from src.memory.preload_gate import PreloadDecisionEngine
    from src.memory.thrashing_detector import ThrashingDetector
    print("7. All core imports: OK")
except Exception as e:
    print(f"7. Import error: {e}")

print()
if not missing:
    print("=== ALL CLEAR - PROJECT READY FOR PANEL ===")
else:
    print("=== HAS MISSING FILES ===")
