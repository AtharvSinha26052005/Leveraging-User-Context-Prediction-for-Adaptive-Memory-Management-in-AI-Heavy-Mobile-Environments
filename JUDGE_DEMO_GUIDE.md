# 🎯 Judge Demo Guide — Step by Step

> **Project:** Context-Aware Predictive Memory Management for Android  
> **What this does:** Predicts which app you'll open next, and proactively keeps it in RAM so it launches instantly.

---

## ⚡ Quick Demo Script (5 minutes total)

### Demo 1: Unit Tests — "It's tested" (30 seconds)

```bash
cd C:\Users\sinha\Desktop\capstone
python -m pytest tests/ -v
```

**What judges see:** 50 tests, 6 files, all green. Shows rigor.

**Talking point:** _"We have 50 unit tests covering all 6 core modules — cache scorer, preload gate, thrashing detector, feature engine, simulation engine, and policies."_

---

### Demo 2: Full Pipeline — "One command, everything runs" (30 seconds to start, explain while it runs)

```bash
python scripts/live_demo.py
```

**What judges see:**
- Data generated (50K events, 5 personas)
- XGBoost trained (75% top-1 accuracy, 93% top-3)
- 4 policies compared side by side
- Proposed policy wins: **+6.8% hit rate**, 111ms faster, 34 fewer evictions

**Talking point:** _"One command runs the entire pipeline — data generation, model training, and simulation of 4 different memory policies. Our proposed policy achieves 93.3% cache hit rate for students versus 87% for Android's LRU."_

---

### Demo 3: Confidence Calibration — "ML depth" (30 seconds)

```bash
python -c "from src.prediction.confidence_calibrator import run_calibration; run_calibration()"
```

**What judges see:**
- ECE drops from 0.1217 → 0.0071 (94.1% improvement)
- Brier score improves

**Talking point:** _"Raw XGBoost probabilities are miscalibrated — a '70% confident' prediction was actually right only 58% of the time. We apply Platt scaling (logistic regression on raw outputs) to calibrate. This makes our preload gate's 60% threshold statistically meaningful."_

---

### Demo 4: Online Learning / Drift Detection — "Production thinking" (1 minute)

```bash
python -c "from src.prediction.online_learner import run_online_learning_demo; run_online_learning_demo()"
```

**What judges see:**
- Streaming events processed
- Drift detection via KL-divergence and PSI
- 8 drifts detected, 2 successful model retrains
- Model versioning with rollback

**Talking point:** _"In production, user behavior changes. Our online learner monitors app distribution drift using KL-divergence and Population Stability Index. When drift exceeds thresholds, it triggers sliding-window retraining on the last 3000 events. If the new model is worse, it auto-rollbacks."_

---

### Demo 5: Dashboard — "Visualization" (1 minute)

```bash
streamlit run dashboard/app.py
```

**What judges see:** 10-tab interactive dashboard with charts.

**Talking point:** _"The Streamlit dashboard provides real-time visualization of all benchmark results, model performance, and ablation studies."_

---

### Demo 6: Self-Collected Data — "Real validation" (30 seconds)

```bash
python -c "import pandas as pd; phone=pd.read_csv('data/self_collected/phone_events.csv'); laptop=pd.read_csv('data/self_collected/laptop_events.csv'); print(f'Phone: {len(phone)} events from Realme 13 5G (Android 16)'); print(f'Laptop: {len(laptop)} events from Windows'); print('Phone apps:', phone.columns[1], phone.iloc[:,1].nunique()); print('Laptop apps:', laptop.columns[1], laptop.iloc[:,1].nunique())"
```

**Talking point:** _"We collected real data from two devices — 312 events from my Realme 13 5G phone using ADB and [X] events from my Windows laptop using ctypes Windows API."_

---

## 🔑 Key Numbers to Memorize

| What | Number |
|------|--------|
| **Total events** | 50,000 synthetic + 312 phone + laptop |
| **Benchmark runs** | 120 (4 policies × 5 personas × 3 RAM × 2 runs) |
| **Unit tests** | 50 (all passing) |
| **ML features** | 19 (12 raw + 7 derived) |
| **App profiles** | 20 realistic apps |
| **User personas** | 5 (student, professional, casual, power_user, developer) |
| **Best improvement** | developer: +9.3% hit rate over LRU |
| **Average improvement** | +5% overall hit rate |
| **Latency reduction** | ~50ms average faster launches |
| **Eviction reduction** | 22% fewer evictions |
| **Calibration** | ECE improved 94.1% |
| **Drift detection** | 8/15 checks triggered drift alerts |
| **Patent claims** | 10 (3 independent + 7 dependent) |

---

## 🎓 Anticipated Panel Questions & Answers

### Q: "Why XGBoost and not a deep learning model?"
**A:** Our data is 19-dimensional tabular data — Grinsztajn et al. (2022) showed tree-based models dominate neural nets on tabular data. XGBoost inference is 0.05ms vs LSTM's 5ms. On a phone, that's imperceptible vs noticeable. Model size: 200KB vs 5-50MB. Also handles missing features natively.

### Q: "How is this different from Android's current LRU?"
**A:** Android's LMKD is reactive — it evicts after OOM. Ours is proactive — we predict the next app and preload it before the user switches. We also use 6 factors (not just recency), detect thrashing, and gate preloads through 5 safety checks.

### Q: "What's the 6-factor scoring formula?"
**A:** S = w_R·Recency + w_F·Frequency + w_P·Prediction + w_I·Importance + w_T·Temporal − w_M·MemoryPressure. Prediction (from XGBoost) and Temporal (time-of-day) are our novel additions. Weights are tuned per-persona via grid search.

### Q: "How do you handle concept drift?"
**A:** We monitor the app distribution using KL-divergence between the training distribution and a sliding window of recent events. When KL > 0.10 or PSI > 0.15 or accuracy drops > 5%, we retrain on the last 3000 events. If the new model is worse, auto-rollback.

### Q: "Why not real Android integration?"
**A:** Modifying LMKD requires AOSP source access and a custom ROM build. Our simulation proves the algorithm works; deployment is an engineering step, not a research question. The patent covers the algorithm regardless of implementation platform.

### Q: "What about the confidence threshold — why 60%?"
**A:** We calibrate probabilities using Platt scaling first. The 60% threshold means: "preload only when we're genuinely 60% sure." Without calibration, 60% might mean anything from 40-80% actual accuracy. ECE before: 0.12, after: 0.007.

### Q: "What's the patent novelty?"
**A:** Three novel claims: (1) ML prediction probability as a weighted factor in cache eviction (no prior art does this), (2) 5-condition preload gate with real-time thrashing detection, (3) Online learning with drift detection and auto-rollback for memory management.

---

## 📁 Project File Count

```
Source files:     23+ Python files
Test files:       6 files (50 tests)  
Documentation:    7 markdown files
Scripts:          6 entry points
Patent:           2 draft documents
Data:             50K+ events
Figures:          6 publication PNGs
Dashboard:        10-tab Streamlit
```

---

## 🏃 Emergency Commands

```bash
# If something breaks, regenerate everything:
python scripts/run_all.py

# Just figures:
python scripts/generate_figures.py

# Just tests:
python -m pytest tests/ -v

# Just calibration:
python -c "from src.prediction.confidence_calibrator import run_calibration; run_calibration()"
```
