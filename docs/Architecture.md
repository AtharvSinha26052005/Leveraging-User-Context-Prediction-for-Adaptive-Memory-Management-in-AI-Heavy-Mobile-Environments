# Architecture — System Map

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Synthetic    │  │ Real Data     │  │ Self-Collected│ │
│  │ Generator    │  │ (6 Kaggle)    │  │ (ADB/Laptop) │ │
│  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘ │
│         └──────────────────┼─────────────────┘         │
│                            ▼                            │
│               ┌────────────────────┐                    │
│               │    Data Loader     │                    │
│               │  (Unified Format)  │                    │
│               └────────┬───────────┘                    │
└────────────────────────┼────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 PREDICTION LAYER                        │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Feature      │  │ XGBoost       │  │ Markov Chain │ │
│  │ Engine       │──│ Predictor     │──│ (Fallback)   │ │
│  │ (19 features)│  │ (Primary)     │  │              │ │
│  └──────────────┘  └───────┬───────┘  └──────────────┘ │
│                            │                            │
│  ┌──────────────────────┐  │  ┌────────────────────┐   │
│  │ Model Comparison     │  │  │ Confidence         │   │
│  │ (5 models evaluated) │  │  │ Calibration        │   │
│  └──────────────────────┘  │  └────────────────────┘   │
└────────────────────────────┼────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│                  DECISION LAYER                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Memory Decision Controller             │   │
│  │  ┌─────────────┐ ┌──────────┐ ┌──────────────┐  │   │
│  │  │ Cache Scorer │ │ Preload  │ │ Thrashing    │  │   │
│  │  │ (6-Factor)  │ │ Gate     │ │ Detector     │  │   │
│  │  │ S=wR+wF+wP  │ │ (5-Gate) │ │ (Safe Mode)  │  │   │
│  │  │  +wI+wC-wM  │ │          │ │              │  │   │
│  │  └─────────────┘ └──────────┘ └──────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│                SIMULATION LAYER                         │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ NoOpt     │ │ LRU      │ │ Static   │ │ Proposed │ │
│  │ (FIFO)    │ │ (Android)│ │ Preload  │ │ (Ours)   │ │
│  └───────────┘ └──────────┘ └──────────┘ └──────────┘ │
│                       ▼                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Benchmark Runner (4 × 5 × 3 × 30 = 1800 runs)  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│               VALIDATION LAYER                          │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Ablation    │  │ Statistical   │  │ Real Data    │ │
│  │ Study       │  │ Tests         │  │ Validation   │ │
│  │ (9 variants)│  │ (p-values)    │  │ (6 datasets) │ │
│  └─────────────┘  └───────────────┘  └──────────────┘ │
└─────────────────────────────┬───────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                         │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ CLI Runner   │  │ Streamlit     │  │ Data         │ │
│  │ (run_all.py) │  │ Dashboard     │  │ Collectors   │ │
│  └──────────────┘  └───────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
run_all.py
  ├── src/data/synthetic_generator.py
  │     └── src/data/app_profiles.py
  │     └── src/models/context.py
  ├── src/prediction/xgboost_predictor.py
  │     ├── src/prediction/feature_engine.py
  │     └── src/prediction/markov_predictor.py
  │           └── src/models/prediction.py
  ├── src/simulation/benchmark_runner.py
  │     └── src/simulation/simulator.py
  │           └── src/simulation/policies.py
  │                 ├── src/memory/decision_controller.py
  │                 │     ├── src/memory/cache_scorer.py
  │                 │     ├── src/memory/preload_gate.py
  │                 │     └── src/memory/thrashing_detector.py
  │                 └── src/models/memory.py
  └── src/models/benchmark.py
```

## Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.12 | ML ecosystem, rapid prototyping |
| ML Framework | XGBoost 2.0 | Best accuracy-speed tradeoff for tabular data |
| Feature Engineering | scikit-learn | StandardScaler, LabelEncoder, cross-validation |
| Data Processing | pandas, numpy | Industry standard for tabular data |
| Statistical Testing | scipy.stats | Paired t-test, Wilcoxon, effect sizes |
| Dashboard | Streamlit + Plotly | Interactive charts, zero JS needed |
| Data Collection | ADB (phone), ctypes (laptop) | Direct OS-level app monitoring |

## File Structure

```
capstone/
├── src/
│   ├── models/          # Data classes (4 files)
│   │   ├── context.py       # AppProfile, ContextSnapshot
│   │   ├── prediction.py    # PredictionResult
│   │   ├── memory.py        # MemoryState, ActionPlan, CacheEntry
│   │   └── benchmark.py     # BenchmarkConfig, BenchmarkResult
│   ├── data/            # Data generation & loading (4 files)
│   │   ├── app_profiles.py      # 20 app definitions
│   │   ├── synthetic_generator.py # 5-persona event generator
│   │   └── real_data_analysis.py  # 6 Kaggle dataset analyzer
│   ├── prediction/      # ML pipeline (6 files)
│   │   ├── feature_engine.py        # 19-feature extraction
│   │   ├── markov_predictor.py      # Transition matrix baseline
│   │   ├── xgboost_predictor.py     # Primary model (200 trees)
│   │   ├── model_comparison.py      # 5-model evaluation
│   │   ├── confidence_calibrator.py # Platt scaling (ECE: 0.12→0.007)
│   │   └── online_learner.py        # Drift detection + sliding window retrain
│   ├── memory/          # Decision layer (5 files)
│   │   ├── cache_scorer.py       # 6-factor scoring formula
│   │   ├── preload_gate.py       # 5-condition safety check
│   │   ├── thrashing_detector.py # Sliding window + Safe Mode
│   │   ├── decision_controller.py # Orchestrator
│   │   └── adaptive_weights.py   # Per-persona weight optimization
│   ├── simulation/      # Benchmarking (3 files)
│   │   ├── policies.py          # 4 eviction policies
│   │   ├── simulator.py         # Discrete-event engine
│   │   └── benchmark_runner.py  # Multi-config runner
│   └── validation/      # Proof (2 files)
│       ├── ablation_study.py    # 9-variant component analysis
│       └── statistical_tests.py # p-values, Cohen's d
├── scripts/             # Entry points
│   ├── run_all.py           # One-click full pipeline (50K events)
│   ├── live_demo.py         # Full pipeline demo with rich output
│   ├── generate_figures.py  # Publication figure generator (6 PNGs)
│   ├── collect_phone_data.py # ADB data collector
│   └── collect_laptop_data.py # Windows app monitor
├── tests/               # Unit tests (6 files, 50 tests)
│   ├── test_cache_scorer.py, test_preload_gate.py
│   ├── test_thrashing_detector.py, test_feature_engine.py
│   ├── test_simulation.py, test_policies.py
├── dashboard/
│   └── app.py               # Streamlit (10 tabs, v2.0)
├── patent/              # Patent drafts
│   ├── specification_outline.md
│   └── claims_draft.md
├── data/
│   ├── external/            # 6 Kaggle datasets
│   ├── self_collected/      # Phone + laptop data
│   └── synthetic_events.csv # Generated events (50K)
├── models_trained/          # Saved ML models + versions/
├── results/                 # Benchmark outputs + figures/
└── docs/                    # Documentation suite (7 docs)
```
