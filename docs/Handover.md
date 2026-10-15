# Handover Document — Session Context

> So the next session (human or AI) isn't starting from zero.

---

## Project: Context-Aware Predictive Memory Management for Android
**Last Updated:** 2026-09-14 20:30 IST  
**Status:** All implementation phases complete. Ready for panel review.

---

## What This Project Is

An ML-powered Android memory management system that **predicts which app the user will open next** and proactively manages the cache (keep/evict/preload) to reduce cold-start latency. It replaces Android's basic LRU eviction with a 6-factor scoring formula driven by XGBoost predictions.

## What's Done ✅

### Core Pipeline (Phase 0-2)
- [x] **Data Models** (`src/models/`): `AppProfile`, `ContextSnapshot`, `PredictionResult`, `MemoryState`, `ActionPlan`, `BenchmarkConfig`, `BenchmarkResult`
- [x] **Synthetic Data Generator** (`src/data/synthetic_generator.py`): 5 personas × 10000 events each = 50K events
- [x] **App Profiles** (`src/data/app_profiles.py`): 20 apps with realistic memory footprints, importance scores, categories
- [x] **Feature Engine** (`src/prediction/feature_engine.py`): 19 features including temporal, categorical, memory-pressure
- [x] **Markov Predictor** (`src/prediction/markov_predictor.py`): First-order transition matrix, used as fallback
- [x] **XGBoost Predictor** (`src/prediction/xgboost_predictor.py`): 200 trees, max_depth=6, with Markov fallback at confidence < 0.30
- [x] **Cache Scorer** (`src/memory/cache_scorer.py`): S = 0.25R + 0.20F + 0.25P + 0.15I + 0.15C - 0.30M
- [x] **Preload Gate** (`src/memory/preload_gate.py`): 5-condition safety check
- [x] **Thrashing Detector** (`src/memory/thrashing_detector.py`): Sliding window + Safe Mode
- [x] **Decision Controller** (`src/memory/decision_controller.py`): Orchestrates scorer + preload + thrashing
- [x] **Simulation Engine** (`src/simulation/simulator.py`): Discrete-event simulator
- [x] **Benchmark Runner** (`src/simulation/benchmark_runner.py`): 4 policies × 5 personas × 3 RAM configs
- [x] **Policies** (`src/simulation/policies.py`): NoOpt, LRU, StaticPreload, Proposed
- [x] **CLI Runner** (`scripts/run_all.py`): One-command full pipeline

### Data Collection (Phase 3)
- [x] **Phone Collector** (`scripts/collect_phone_data.py`): ADB-based, tested on Realme 13 5G (Android 16)
- [x] **Laptop Collector** (`scripts/collect_laptop_data.py`): Windows API via PowerShell
- [x] **ADB installed** at `C:\Users\sinha\Desktop\platform-tools\adb.exe`
- [x] **6 Kaggle datasets** downloaded to `data/external/`

### Validation (Phase 4)
- [x] **Ablation Study** (`src/validation/ablation_study.py`): 9 variants
- [x] **Statistical Tests** (`src/validation/statistical_tests.py`): Paired t-test, Cohen's d
- [x] **Real Data Analysis** (`src/data/real_data_analysis.py`): 6-dataset comparison
- [x] **Model Comparison** (`src/prediction/model_comparison.py`): 5 models trained — XGBoost chosen

### Dashboard
- [x] **Streamlit Dashboard** (`dashboard/app.py`): Enhanced 10-tab version (v2.0)

### Advanced Features (Phase 5)
- [x] **Adaptive Weight Learning** (`src/memory/adaptive_weights.py`): Per-persona optimal weights
- [x] **Live Demo Script** (`scripts/live_demo.py`): Full pipeline demo with rich output
- [x] **Confidence Calibration** (`src/prediction/confidence_calibrator.py`): Platt scaling, ECE 0.12→0.007
- [x] **Online Learning** (`src/prediction/online_learner.py`): KL-divergence drift detection, model versioning, rollback

### Testing
- [x] **Unit Tests** (`tests/`): 6 files, 50 tests, all passing
- [x] **Publication Figures** (`scripts/generate_figures.py`): 6 camera-ready PNGs

### Patent
- [x] **Specification Outline** (`patent/specification_outline.md`): Claims, abstract, module descriptions
- [x] **Claims Draft** (`patent/claims_draft.md`): 3 independent + 7 dependent claims

### Documentation
- [x] `docs/Architecture.md` — System map
- [x] `docs/Flow.md` — Execution trace
- [x] `docs/Decisions.md` — 12 AI decisions logged with rationale
- [x] `docs/Handover.md` — This file
- [x] `docs/Bug.md` — Bug trail
- [x] `docs/Feature.md` — 19 features logged + post-review roadmap
- [x] `docs/TestChecklist.md` — Proof it works

## What's In Progress 🔄

1. **Full 120-benchmark run** — `python scripts/run_all.py` (50K events, 3 RAM configs, 2 runs each)

## What's NOT Done ❌ (Post-Review Roadmap)

1. **App Embeddings (App2Vec)** — Needs 5000+ real events
2. **Temporal Pattern Mining (PrefixSpan)** — Needs multi-day data
3. **IEEE LaTeX Paper** — Writing effort
4. **IPO Patent Filing** — Legal review required
5. **1-week phone data collection** — Calendar time
6. **Production AOSP Integration** — Would require Android framework modification

## Key Files to Know

| What | Path |
|------|------|
| Entry point | `scripts/run_all.py` |
| Synthetic data | `data/synthetic_events.csv` (50K events) |
| Trained model | `models_trained/xgboost_model.json` |
| Calibrator | `models_trained/confidence_calibrator.pkl` |
| Benchmark results | `results/benchmark_results.csv` (120 runs) |
| Publication figures | `results/figures/` (6 PNGs) |
| Dashboard | `dashboard/app.py` |
| Phone collector | `scripts/collect_phone_data.py` |
| Self-collected data | `data/self_collected/phone_events.csv` (312 events) |
| Unit tests | `tests/` (6 files, 50 tests) |
| Patent | `patent/` (spec + claims) |

## Environment

- **Python:** 3.12
- **Key packages:** xgboost, scikit-learn, pandas, numpy, scipy, streamlit, plotly
- **OS:** Windows 11
- **Phone:** Realme 13 5G (RMX3951), Android 16, Realme UI 7.0, 8GB RAM
- **ADB:** `C:\Users\sinha\Desktop\platform-tools\adb.exe`

## Known Gotchas

1. **Android 16 uses `topResumedActivity`** not `mResumedActivity` — already handled in phone collector
2. **ADB daemon dies frequently** — script handles reconnection gracefully
3. **Laptop collector needs interactive shell** — PowerShell `GetForegroundWindow()` returns 0 in headless mode
4. **MarkovPredictor.fit()** takes `(current_apps_list, next_apps_list)`, NOT a DataFrame
