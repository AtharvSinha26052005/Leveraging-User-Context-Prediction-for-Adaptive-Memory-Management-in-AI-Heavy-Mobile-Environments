# Test Checklist — Proof It Works

> Not just claims. Evidence.

---

## 1. Unit Tests

### Prediction Layer
| Test | File | Command | Expected | Status |
|------|------|---------|----------|--------|
| Markov fit + predict | `src/prediction/markov_predictor.py` | `python -c "from src.prediction.markov_predictor import MarkovPredictor; m=MarkovPredictor(); m.fit(['A','B','A'],['B','A','B']); print(m.predict('A'))"` | PredictionResult with B as top prediction | ✅ Pass |
| Feature engine produces 19 features | `src/prediction/feature_engine.py` | `python -c "..."` | X_train.shape[1] == 19 | ✅ Pass |
| XGBoost train + predict cycle | `src/prediction/xgboost_predictor.py` | `python scripts/run_all.py` | Top-1 accuracy > 50% | ✅ Pass |
| XGBoost fallback to Markov at low confidence | `src/prediction/xgboost_predictor.py` | Inspect logs | Fallback fires ~15-25% of the time | ✅ Verified |

### Memory Layer
| Test | File | Expected | Status |
|------|------|----------|--------|
| Cache scorer produces bounded scores [0, 1] | `src/memory/cache_scorer.py` | All scores in range | ✅ Pass |
| Preload gate blocks when RAM < 500MB | `src/memory/preload_gate.py` | `should_preload() == False` | ✅ Pass |
| Thrashing detector enters Safe Mode at score > 0.70 | `src/memory/thrashing_detector.py` | Mode switches to SAFE | ✅ Pass |
| Decision controller evicts lowest-scored apps | `src/memory/decision_controller.py` | Lowest score evicted first | ✅ Pass |

### Simulation Layer
| Test | Expected | Status |
|------|----------|--------|
| NoOpt policy gives worst hit rate | Hit rate < LRU < Proposed | ✅ Pass |
| LRU matches Android default behavior | Evicts least-recently-used | ✅ Pass |
| Proposed beats all baselines on hit rate | Proposed > StaticPreload > LRU > NoOpt | ✅ Pass |
| 3 RAM configs produce different results | 4GB < 6GB < 8GB hit rates | ✅ Pass |

---

## 2. Integration Tests

### Full Pipeline
| Test | Command | Expected | Status |
|------|---------|----------|--------|
| One-command pipeline | `python scripts/run_all.py` | Generates data, trains, benchmarks, prints results | ✅ Pass |
| Dashboard launches | `streamlit run dashboard/app.py` | Browser opens with 5 tabs | ✅ Pass |
| Benchmark CSV is valid | `python -c "import pandas as pd; df=pd.read_csv('results/benchmark_results.csv'); print(df.shape)"` | 60+ rows, 12+ columns | ✅ Pass |

### Data Collection
| Test | Expected | Status |
|------|----------|--------|
| Phone collector detects foreground app | Returns `com.instagram.android` → `Instagram` | ✅ Pass (tested live) |
| Phone collector reads battery | Returns level + charging state | ✅ Pass (3%, charging) |
| Phone collector reads RAM | Returns total + free in MB | ✅ Pass (7619MB total, 2092MB free) |
| Laptop collector detects foreground window | Returns process name + window title | ⬜ Needs interactive shell |
| CSV output has correct schema | All 13 columns present | ✅ Pass |

---

## 3. Statistical Validation

| Test | Expected | Status |
|------|----------|--------|
| Proposed vs LRU: p < 0.05 on hit rate | Statistically significant improvement | ⬜ Run after benchmark |
| Proposed vs NoOpt: p < 0.001 | Highly significant | ⬜ Run after benchmark |
| Cohen's d > 0.8 (large effect) | Proposed is a meaningful improvement, not noise | ⬜ Run after benchmark |
| Ablation: removing prediction hurts most | Largest delta in hit rate | ⬜ Run after benchmark |

---

## 4. Real Data Validation

| Test | Expected | Status |
|------|----------|--------|
| 6 Kaggle datasets load successfully | All 6 files parse as CSVs | ⬜ Run `real_data_analysis.py` |
| Screen time distributions match real data | Within 1 std dev of Kaggle means | ⬜ Compare output |
| App category splits match real data | Social 20-40%, productivity 15-30% | ⬜ Compare output |

---

## 5. Demo Readiness

| Check | Status |
|-------|--------|
| `python scripts/run_all.py` completes in < 5 min | ✅ |
| Dashboard loads without errors | ✅ |
| Phone data collector runs for 30 min | 🔄 Running now |
| Laptop data collector runs for 30 min | 🔄 Running now |
| All docs exist in `docs/` | ✅ |
| `results/benchmark_results.csv` has valid data | ✅ |
