# Decisions Log — Why We Made Each Choice

> Every AI/ML and engineering decision documented with rationale.

---

## D-001: XGBoost as Primary Predictor (not LSTM, not Transformer)
**Date:** 2026-08-18  
**Decision:** Use XGBoost (gradient-boosted trees) for app-next prediction.  
**Why NOT a deep learning model?**
- Our feature space is 19-dimensional tabular data — trees dominate here (see [Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815))
- XGBoost inference: ~0.05ms. LSTM inference: ~5ms. On a phone, this is the difference between imperceptible and noticeable lag.
- XGBoost natively handles missing features (phone in airplane mode = no network feature). NNs need imputation.
- Model size: ~200KB JSON. LSTM: 5-50MB. Matters for on-device deployment.

**Trade-off:** Slightly lower sequence modeling (no memory of long-term patterns), compensated by Markov chain fallback.

---

## D-002: Markov Chain as Fallback (not uniform random)
**Date:** 2026-08-18  
**Decision:** When XGBoost confidence < 0.30, fall back to a first-order Markov chain.  
**Why?**
- XGBoost sometimes outputs near-uniform probabilities for unseen contexts (cold start).
- Markov chain captures the dominant "App A → App B" habit even with zero features.
- Fallback rate is ~15-25% in practice — not a crutch, a safety net.

**Alternative considered:** Second-order Markov (bigram). Rejected because it requires O(n²) memory for n apps.

---

## D-003: 6-Factor Scoring Formula (not a single metric)
**Date:** 2026-08-18  
**Decision:** Cache priority score = 0.25R + 0.20F + 0.25P + 0.15I + 0.15C - 0.30M  
**Why 6 factors?**
- Android's LRU uses only recency (R). We showed in ablation that adding each factor improves hit rate.
- Prediction (P) and Recency (R) get the highest weights because ablation showed they have the largest individual impact.
- Memory pressure penalty (M) is subtractive — it actively pushes evictions when RAM is low, preventing OOM kills.

**How weights were chosen:** Grid search over 5 personas × 3 RAM configs. Optimized for cache hit rate while keeping eviction count within 2× of LRU baseline.

---

## D-004: 5-Gate Preload System (not unconditional preloading)
**Date:** 2026-08-18  
**Decision:** Preloading requires passing ALL 5 gates:
1. Prediction confidence ≥ 0.60
2. Free RAM ≥ 500MB  
3. Battery ≥ 20%
4. Thrashing score < 0.70
5. App not already cached

**Why so conservative?**
- Aggressive preloading CAUSES thrashing (we proved this in ablation — removing Gate 4 increased evictions by 40%).
- Battery gate prevents the ML system from draining battery faster than stock Android — a hard requirement for any real product.
- The 0.60 confidence threshold was calibrated: below it, the prediction is wrong >50% of the time, making preloading a net negative.

---

## D-005: Sliding Window Thrashing Detection (not fixed threshold)
**Date:** 2026-08-18  
**Decision:** Thrashing score = 0.40 × reload_rate + 0.35 × miss_rate + 0.25 × eviction_rate, computed over a 120-second sliding window.  
**Why?**
- Fixed thresholds (e.g., "if evictions > 10, stop") fail because normal gaming sessions have high evictions but low reloads.
- The sliding window forgets old behavior in 2 minutes — important because phone usage is bursty.
- Safe Mode (triggered at score > 0.70) increases free RAM target by 1.5× and disables preloading until score drops below 0.30 for 60 seconds.

---

## D-006: Synthetic Data with Real-World Calibration (not pure simulation)
**Date:** 2026-08-18  
**Decision:** Generate synthetic app-switch events calibrated against 6 real-world Kaggle datasets.  
**Why not train directly on Kaggle data?**
- No Kaggle dataset contains raw app-switch sequences (timestamp, current_app, next_app). They contain aggregated usage stats.
- We use real data to validate: "Does our synthetic generator produce realistic session lengths, app category distributions, and battery curves?"
- Comparison table in `results/real_vs_synthetic.csv` proves distributions match.

---

## D-007: 5-Persona Simulation (not single-user)
**Date:** 2026-08-18  
**Decision:** Simulate 5 distinct user personas: student, professional, casual, power_user, developer.  
**Why?**
- A single "average user" hides important performance differences. Our system must work for a gamer (rapid switching, high RAM) AND a casual user (slow switching, low RAM).
- Each persona has different app preferences, session lengths, and temporal patterns.
- The benchmark matrix (5 personas × 3 RAM × 4 policies) produces 60 data points for statistical significance.

---

## D-008: Self-Collected Data via ADB + Windows API
**Date:** 2026-09-13  
**Decision:** Build real data collectors for phone (ADB dumpsys) and laptop (PowerShell + Win32 API).  
**Why?**
- Panel reviewers value real data over synthetic. "We collected 30 minutes of YOUR behavior" is more convincing.
- ADB dumpsys is non-invasive — no root required, no app installation needed.
- Laptop collector uses zero external dependencies (pure Windows API via PowerShell).

---

## D-009: Streamlit Dashboard (not Flask/React)
**Date:** 2026-08-18  
**Decision:** Use Streamlit for the interactive dashboard.  
**Why?**
- Zero frontend code needed. Python-only. Plotly charts are interactive by default.
- Hot reload — change code, see results instantly.
- Tabs for different analysis views (overview, model comparison, ablation, real data).

**Trade-off:** Not suitable for production deployment. This is a research/demo tool.

---

## D-010: Adaptive Weight Learning via Grid Search (not fixed weights)
**Date:** 2026-09-14  
**Decision:** Discover per-persona optimal weights using grid search + perturbation refinement.  
**Why?**
- Default hand-tuned weights (D-003) work "OK" for all personas but are suboptimal for each.
- Grid search with intelligent sampling finds better weights in ~40 trials per persona.
- Power_user improved by **+5.8%** (74.4% vs 68.6% default) — prediction and importance matter more for heavy switchers.
- Developer improved by **+0.6%** (89.6% vs 89.0%) — already near ceiling due to predictable patterns.

**Patent Claim:** "An adaptive cache scoring system that automatically discovers per-user-type weight configurations using iterative optimization over a multi-factor scoring formula."

**Alternative considered:** Bayesian optimization (Optuna). Rejected because grid search + perturbation achieves similar results with zero additional dependencies.

---

## D-011: Platt Scaling for Confidence Calibration (not isotonic regression)
**Date:** 2026-09-14  
**Decision:** Calibrate XGBoost prediction probabilities using Platt scaling (logistic regression on raw probabilities).  
**Why?**
- Raw XGBoost "78% confident" doesn't mean 78% accuracy at that level — it's miscalibrated.
- ECE (Expected Calibration Error) was 0.1217 before calibration — meaning probabilities were off by ~12% on average.
- After Platt scaling: ECE = 0.0071 (94.1% improvement).
- This matters because the preload gate uses a 60% confidence threshold (FEAT-003). Calibrated probabilities make this threshold meaningful.

**Why NOT isotonic regression?** Isotonic is non-parametric and can overfit on small validation sets. Platt's single-sigmoid fit is more robust with our ~7500 test events.

**Reference:** Platt, J. (1999). "Probabilistic outputs for SVMs."

---

## D-012: Sliding Window Retraining for Online Learning (not exponential weighting)
**Date:** 2026-09-14  
**Decision:** Use a sliding window buffer (last N events) for retraining, not exponential weighting of old samples.  
**Why?**
- XGBoost is a tree-based model — it can't do gradient updates like neural nets. Full retrain is required.
- Sliding window naturally "forgets" stale patterns (apps uninstalled, routine changes).
- Window size (5000) is tunable per-persona: power_users need shorter windows, casual_users longer.
- Drift detection via KL-divergence + PSI (Population Stability Index) provides dual-metric triggering for robustness.

**Rollback logic:** If retrained model accuracy < previous model - 2%, auto-rollback to previous version. Prevents deploying degraded models during temporary distribution shifts.

**Alternative considered:** River/online learning libraries. Rejected because XGBoost's batch-retrain approach achieves better accuracy than incremental learners on tabular data.
