# Execution Flow — How Code Runs Step by Step

## Main Pipeline: `scripts/run_all.py`

```
run_all.py main()
│
├── STEP 1: Generate Data
│   └── SyntheticDataGenerator(seed=42)
│       └── generate_all(events_per_persona=5000)
│           ├── For each persona in [student, professional, casual, power_user, developer]:
│           │   └── _generate_persona_events()
│           │       ├── Get APP_PROFILES from app_profiles.py (20 apps)
│           │       ├── Select apps based on persona.preferred_categories
│           │       ├── Build transition probabilities (40% transition + 20% temporal + 40% persona)
│           │       └── For each event:
│           │           ├── Pick next_app from transition matrix
│           │           ├── Generate timestamp (advancing 10-300s)
│           │           ├── Simulate battery decay
│           │           └── Append {timestamp, current_app, next_app, hour, battery, persona, ...}
│           └── Return: DataFrame with 25K rows
│
├── STEP 2: Train ML Models
│   └── XGBoostPredictor(n_estimators=200, max_depth=6)
│       └── fit(train_df, val_df=test_df)
│           ├── feature_engine.fit(train_df)
│           │   ├── LabelEncoder for: current_app, previous_app, category
│           │   ├── StandardScaler for: hour, day_of_week, battery, ram_usage
│           │   └── Derive: is_peak_hour, is_weekend, ram_utilization, session_length, etc.
│           ├── X_train = feature_engine.transform(train_df) → 19-dim vectors
│           ├── y_train = LabelEncoder(next_app)
│           ├── xgb.XGBClassifier.fit(X_train, y_train)
│           ├── 5-fold cross_val_score → cv_accuracy_mean ± cv_accuracy_std
│           ├── markov_predictor.fit(current_apps, next_apps) → transition matrix
│           └── Return: metrics dict
│
├── STEP 3: Benchmark Simulation
│   └── BenchmarkRunner.run_all_policies()
│       └── For each (persona × ram_config × policy):
│           └── SimulationEngine.run(events, policy, predictor, ram_mb)
│               ├── Initialize MemoryState(total_ram_mb)
│               ├── For each event in events_df:
│               │   ├── Check: is current_app in cache?
│               │   │   ├── YES → cache_hits++, use warm_launch_ms
│               │   │   └── NO  → cache_misses++, use cold_launch_ms
│               │   ├── If policy == "proposed":
│               │   │   └── predictor.predict(event) → PredictionResult
│               │   │       ├── feature_engine.transform(event)
│               │   │       ├── xgb_model.predict_proba() → probabilities
│               │   │       ├── If max_prob < fallback_threshold (0.3):
│               │   │       │   └── Use markov_predictor instead
│               │   │       └── Return PredictionResult(app, confidence, top_k)
│               │   └── policy.on_app_switch(event, memory_state, prediction, now)
│               │       └── [For ProposedPolicy]:
│               │           └── decision_controller.decide()
│               │               ├── cache_scorer.rank_all() → [(app, score), ...]
│               │               │   └── For each cached app:
│               │               │       ├── R = exp(-λ × seconds_since_access)
│               │               │       ├── F = access_count / max_access_count
│               │               │       ├── P = prediction.get_app_probability(app)
│               │               │       ├── I = app.importance (category-based)
│               │               │       ├── C = context_relevance(category, hour)
│               │               │       ├── M = pressure_penalty(ram_utilization)
│               │               │       └── S = 0.25R + 0.20F + 0.25P + 0.15I + 0.15C - 0.30M
│               │               ├── While free_ram < target:
│               │               │   └── Evict lowest-scored app
│               │               ├── preload_engine.should_preload()
│               │               │   ├── Gate 1: confidence >= 0.60?
│               │               │   ├── Gate 2: free_ram >= 500MB?
│               │               │   ├── Gate 3: battery >= 20%?
│               │               │   ├── Gate 4: thrashing_score < 0.70?
│               │               │   └── Gate 5: app NOT already cached?
│               │               ├── thrashing_detector.record_eviction/miss/reload()
│               │               │   └── score = 0.40×reload_rate + 0.35×miss_rate + 0.25×eviction_rate
│               │               │       └── If score > 0.70 → enter Safe Mode
│               │               └── Return MemoryActionPlan
│               └── Return BenchmarkResult (hit_rate, latency, evictions, etc.)
│
└── STEP 4: Print Results
    └── print_comparison_table() → formatted terminal output
```

## Key Branching Points

### When does Markov fire instead of XGBoost?
```
xgboost_predictor.predict(event)
  → feature_engine.transform(event)
  → xgb_model.predict_proba(features)
  → max_probability = max(probabilities)
  → IF max_probability < 0.30:   ← fallback_threshold
      → markov_predictor.predict(current_app)  ← MARKOV FIRES
    ELSE:
      → Use XGBoost prediction
```

### When does Safe Mode activate?
```
thrashing_detector._update_mode(now)
  → score = get_score(now)  [sliding 120-second window]
  → IF mode == NORMAL AND score >= 0.70:
      → mode = SAFE  [increases free RAM target by 1.5×, disables preloading]
  → IF mode == SAFE AND score < 0.30 for 60 seconds:
      → mode = NORMAL  [resume normal operations]
```

---

## 6. Adaptive Weight Optimizer (`src/memory/adaptive_weights.py`)

```
optimize_all(events_df, predictor, ram_mb)
  → FOR each persona IN events_df.persona.unique():
      → optimize_persona(persona, events_df, predictor, ram_mb, n_trials=40)
          → persona_df = filter events for this persona (max 500)
          → Phase 1: Coarse Grid Search
              → Generate combos from [0.05, 0.15, 0.25, 0.35]^5 × pressure[0.10..0.40]
              → Randomly sample n_trials combos (normalize positive weights to sum to 1.0)
              → FOR each weight config:
                  → Run SimulationEngine with ProposedPolicy (custom weights)
                  → Track best hit rate
          → Phase 2: Fine-Tuning
              → FOR 20 iterations:
                  → Perturb best weights by N(0, 0.03)
                  → Clip, normalize, evaluate
                  → Track improvements
          → RETURN best_weights, best_hit_rate
  → Save per-persona optimal weights to results/adaptive_weights.csv
```

---

## 7. Live Demo Script (`scripts/live_demo.py`)

```
run_live_demo(source="auto")
  → Step 1: Load data (phone_events.csv if exists, else synthetic_events.csv)
  → Step 2: Load or build predictor
      → TRY load XGBoost from models_trained/
      → FALLBACK to DummyPredictor (Markov transition frequencies)
  → Step 3: Show sample predictions (10 events with confidence)
  → Step 4: Run 4-policy benchmark on first 500 events
      → noopt, lru, static_preload, proposed
  → Step 5: Print comparison table (hit rate, latency, evictions)
  → Step 6: Print system summary
```

