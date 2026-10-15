# Master Index & Presentation Slide Outline

**Project Title:** Leveraging User Context Prediction for Adaptive Memory Management in AI-Heavy Mobile Environments  
**Target Audience:** Capstone Project Mentors, Evaluation Committee, and Panel Reviewers  
**Documentation Package:** Complete Technical Evidence, Benchmark Proofs, Weight Evolution, and Presentation Reference

---

## 📌 Executive Overview

Modern mobile platforms (Android) face severe resource contention: on-device artificial intelligence workloads (Large Language Models, real-time computer vision, diffusion models) consume 300 MB to 4.5 GB of RAM per application. Simultaneously, users frequently switch across 15–29 daily applications. Android’s default memory management architecture relies on:
1. **Least Recently Used (LRU)** cache replacement, which naively evicts the application with the longest idle time without considering user intent.
2. **Low Memory Killer Daemon (LMKD)**, which reacts only when memory pressure reaches critical kernel thresholds (via Pressure Stall Information - PSI), triggering abrupt process kills.

This leads to catastrophic **cold-start reload latencies (1,200 ms – 6,000 ms)**, battery waste, and **memory thrashing** (rapid eviction-reload cycles).

### The Proposed Innovation: Adaptive Memory Management (AMM)
Our system transforms mobile memory management from **reactive process killing** to **proactive, context-aware retention and preloading**:
- **46-Dimensional Context Vector:** Fuses 19 spatiotemporal/device signals with **16-dim App2Vec graph embeddings** and **11-dim PrefixSpan sequential pattern mining**.
- **Calibrated Prediction Engine:** An on-device gradient-boosted tree (XGBoost) achieving **14.26% Top-1, 37.42% Top-3, and 50.70% Top-5 accuracy** across 29 classes (an **11x improvement** over baseline LRU).
- **6-Factor Dynamic Cache Scorer:** Computes an eviction priority score $S(a) = w_R R + w_F F + w_P P + w_I I + w_C C - w_M M$, where weights adapt dynamically to user personas (e.g., $w_P$ scales to 0.305 for Power Users).
- **5-Gate Preload Controller:** Preloads predicted apps strictly when prediction confidence, available RAM headroom, battery level, network state, and thrashing score pass concurrent safety checks.
- **Native AOSP & Patent Ready:** Prototype ported to native C++ (<0.05 ms latency, <100 KB footprint) with complete Indian Patent Form 2 specification (10 claims).

---

## 📂 Navigation: The 7 In-Depth Documentation Files

| File | Core Subject | Key Evidence & Deliverables |
|---|---|---|
| **[01_TASK1_DATA_COLLECTION_AND_MULTIUSER_PIPELINE.md](file:///c:/Users/sinha/Desktop/capstone/docs/mentor_presentation/01_TASK1_DATA_COLLECTION_AND_MULTIUSER_PIPELINE.md)** | Multi-User Dataset & Hardware Collection | 181,546 events; 15 personas; 6 Kaggle dataset calibrations; Android ADB & Windows ctypes hardware validation. |
| **[02_TASK2_AND_TASK3_APP2VEC_AND_PREFIXSPAN.md](file:///c:/Users/sinha/Desktop/capstone/docs/mentor_presentation/02_TASK2_AND_TASK3_APP2VEC_AND_PREFIXSPAN.md)** | Graph Embeddings & Sequence Mining | Random-walk App2Vec (64-dim $\to$ 8-dim PCA; YouTube-Netflix similarity 0.853); PrefixSpan sequence miner (WhatsApp $\to$ Insta 50.7%). |
| **[03_TASK4_PREDICTION_ENGINE_AND_WEIGHT_EVOLUTION.md](file:///c:/Users/sinha/Desktop/capstone/docs/mentor_presentation/03_TASK4_PREDICTION_ENGINE_AND_WEIGHT_EVOLUTION.md)** | ML Prediction & Feature Weight Shifts | Phase 1 (19 feats) vs Phase 2 (46 feats); 50K stratified training; App2Vec dominant gain; per-class breakdown across 29 apps. |
| **[04_TASK5_EMPIRICAL_BENCHMARKS_AND_STATISTICAL_PROOFS.md](file:///c:/Users/sinha/Desktop/capstone/docs/mentor_presentation/04_TASK5_EMPIRICAL_BENCHMARKS_AND_STATISTICAL_PROOFS.md)** | Empirical Validation & Mathematical Proofs | 120-run benchmark; +10.8% hit rate; -10.8% latency; -20.7% evictions; Paired t-test ($p = 0.000172$, Cohen’s $d = 1.939$); 9-variant ablation. |
| **[05_TASK6_ADAPTIVE_CACHE_WEIGHTS_AND_MEMORY_CONTROLLER.md](file:///c:/Users/sinha/Desktop/capstone/docs/mentor_presentation/05_TASK6_ADAPTIVE_CACHE_WEIGHTS_AND_MEMORY_CONTROLLER.md)** | Dynamic Scoring & Thrashing Mitigation | Mathematical derivation of 6-factor score; Developer vs Power User weight adaptation; 5-condition Preload Gate; Sliding-window Thrashing Detector. |
| **[06_TASK7_AOSP_CPP_INTEGRATION_AND_PATENT_SPECIFICATION.md](file:///c:/Users/sinha/Desktop/capstone/docs/mentor_presentation/06_TASK7_AOSP_CPP_INTEGRATION_AND_PATENT_SPECIFICATION.md)** | C++ OS Engine & Patent Form 2 | Native `inference.cpp` benchmark; `ActivityManagerService.java` & `lmkd.cpp` hook architecture; Complete Indian Patent Specification & 10 Claims. |
| **[07_MENTOR_QA_DEFENSE_AND_SLIDE_TALKING_POINTS.md](file:///c:/Users/sinha/Desktop/capstone/docs/mentor_presentation/07_MENTOR_QA_DEFENSE_AND_SLIDE_TALKING_POINTS.md)** | Presentation Script & Mentor Defense | 15 slide-by-slide scripts; Graphic placement mapping; 15 tough mentor Q&A responses with technical justifications. |

---

## 📊 Summary of Headline Metrics (Slide 1 / Executive Snapshot)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CAPSTONE QUANTITATIVE SCORECARD                            │
├────────────────────────────┬─────────────────────────────┬──────────────────────┤
│ Metric                     │ Baseline (Android LRU / P1) │ Proposed AMM System  │
├────────────────────────────┼─────────────────────────────┼──────────────────────┤
│ Dataset Training Scale     │ 7,000 synthetic events      │ 50,000 events (181K) │
│ Feature Dimensionality     │ 19 base features            │ 46 enriched features │
│ Next-App Top-1 Accuracy    │ 9.90%                       │ 14.26% (+44.0% rel)  │
│ Next-App Top-3 Accuracy    │ 28.90%                      │ 37.42% (+29.5% rel)  │
│ Next-App Top-5 Accuracy    │ N/A (unsupported)           │ 50.70% (new)         │
│ Prediction vs Random (29)  │ 2.87x                       │ 4.14x (Top-1)        │
│ Cache Hit Rate (Overall)   │ 68.3%                       │ 75.7% (+10.8% rel)   │
│ Cache Hit Rate (PowerUser) │ 63.8% (4GB RAM)             │ 72.0% (+12.9% rel)   │
│ Average Launch Latency     │ 708.1 ms                    │ 631.7 ms (-10.8%)    │
│ Process Evictions (Kills)  │ 624.6 / benchmark session   │ 495.6 (-20.7%)       │
│ Statistical Significance   │ Baseline                    │ p = 0.00017, d=1.939 │
│ On-Device C++ Latency      │ N/A                         │ 0.048 ms (<50 µs)    │
│ Memory Overhead of Model   │ N/A                         │ 217 KB (<0.01% RAM)  │
│ Intellectual Property      │ None                        │ Complete IPO Form 2  │
└────────────────────────────┴─────────────────────────────┴──────────────────────┘
```

---

## 🖥️ Recommended 14-Slide Presentation Structure for Mentor Review

1. **Slide 1: Title & Team Credentials** (Project title, team members, supervisor, problem scope).
2. **Slide 2: The Core Problem** (RAM bottlenecks in AI mobile environments, LRU shortcomings, LMKD latency penalty).
3. **Slide 3: Architectural Innovation** (Closed-loop architecture: Context Engine $\to$ Prediction Engine $\to$ Decision Controller $\to$ OS Eviction/Preload).
4. **Slide 4: Task 1 — Multi-User Data Engine** (181K events, 15 user personas, ADB smartphone telemetry, Kaggle validation).
5. **Slide 5: Task 2 — App2Vec Graph Embeddings** (App transition network, random walks, 16-dim semantic space, cosine similarity clusters).
6. **Slide 6: Task 3 — PrefixSpan Sequential Pattern Mining** (Mining deep temporal dependencies, WhatsApp $\to$ Instagram 50.7% support, pattern vectors).
7. **Slide 7: Task 4 — Machine Learning Pipeline & Weight Evolution** (19 $\to$ 46 features, 50K stratified training, XGBoost vs Markov/RF/MLP).
8. **Slide 8: Task 4 — Per-Class Prediction & Feature Dominance** (App2Vec ranking #1 in gain, Top-3 accuracy 37.4%, predictable vs context apps).
9. **Slide 9: Task 6 — Dynamic Cache Scoring & Adaptive Weights** ($S$-formula, mathematical derivation, persona weight shifts: Developer vs Power User).
10. **Slide 10: Task 6 — 5-Gate Preload & Thrashing Mitigation** (Confidence gate, RAM headroom, battery gate, sliding-window Safe Mode).
11. **Slide 11: Task 5 — Benchmark Results & Statistical Proofs** (+10.8% Hit rate, -10.8% latency, paired t-test $p < 0.001$, Cohen’s $d = 1.939$).
12. **Slide 12: Task 5 — Ablation Study** (Component breakdown: impact of removing prediction, frequency, recency, and thrashing gates).
13. **Slide 13: Task 7 — AOSP Native Integration (C++)** (C++ prototype, <0.05 ms latency, `ActivityManagerService` & `lmkd` hook architecture).
14. **Slide 14: Task 7 — Intellectual Property (Patent Form 2)** (Indian Patent Office submission, 10 legal claims, novelty mapping vs 20 papers).
