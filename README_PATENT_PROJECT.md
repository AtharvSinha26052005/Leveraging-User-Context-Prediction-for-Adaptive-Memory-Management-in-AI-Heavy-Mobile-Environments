# Leveraging User Context Prediction for Adaptive Memory Management in AI-Heavy Mobile Environments

## Technical Specification & Capstone Report

> **Patent Application Title:** System and Method for Context-Aware Predictive Memory Management and Cache Eviction in Mobile Computing Devices Using Application Usage Prediction

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Domain Background & Motivation](#14-domain-background--motivation)
3. [Literature Review](#15-literature-review)
4. [Proposed Methodology](#16-proposed-methodology)
5. [Module Descriptions](#17-module-descriptions)
6. [Comparison with Existing Approaches](#18-comparison-with-existing-approaches)
7. [Novelty Statement](#19-novelty-statement)
8. [High-Level Design (HLD)](#2-high-level-design-hld)
9. [Low-Level Design (LLD)](#3-low-level-design-lld)
10. [ML Pipeline — Detailed](#4-ml-pipeline--detailed)
11. [Core Algorithms — Pseudocode](#5-core-algorithms--pseudocode)
12. [Database Schema](#6-database-schema)
13. [API Contracts](#7-api-contracts)
14. [Build Steps — Week by Week](#8-build-steps--week-by-week)
15. [Directory Structure](#9-directory-structure)
16. [Verification Plan](#10-verification-plan)

---

## 1. System Overview

### 1.1 Problem Statement

Mobile devices with limited physical RAM (4–8 GB) run concurrent AI workloads (camera AI, translation, on-device LLMs, voice assistants) alongside user applications. Current mobile operating systems manage memory reactively using static policies (LRU, Low Memory Killer) that respond only after memory pressure is critical, causing:

- Application cold-start latency (1–3 seconds per reload)
- Memory thrashing (repeated evict → reload → evict cycles)
- AI inference delays from evicted model caches
- Battery drain from redundant I/O operations

### 1.2 Proposed Solution

A **predictive memory management system** that:

1. **Collects** real-time device context (app usage, time, battery, RAM, AI tasks)
2. **Predicts** the next application the user will open, with a confidence score
3. **Integrates** prediction confidence into a multi-factor cache eviction scoring formula
4. **Preloads** predicted applications only when resource conditions are favorable
5. **Detects** memory thrashing and automatically degrades to a protective mode

### 1.3 Key Innovation (Patent Claim)

The inventive step is the **integration of ML prediction confidence as a first-class numerical factor in cache eviction scoring**, combined with **multi-condition resource-gated preloading** and **thrashing-triggered automatic degradation** — operating as a unified system on a mobile computing device to produce measurable technical effects (reduced latency, reduced thrashing, improved memory utilization).

---

## 1.4 Domain Background & Motivation

### 1.4.1 Mobile Memory Constraints in the AI Era

The average smartphone user installs 40–80 applications and actively uses 9–10 apps per day, switching between them roughly 65–70 times daily (Statista, 2024). As of 2025, the median Android device ships with 6 GB RAM, yet on-device AI models for tasks like computational photography (300–600 MB), real-time translation (200–400 MB), and emerging on-device LLMs (1–4 GB) now compete directly with user applications for this limited resource. The global mobile AI market is projected to reach $46.7 billion by 2027 (MarketsandMarkets), indicating that this competition for memory will only intensify.

### 1.4.2 Limitations of Current Memory Management

Android's memory management relies on two primary mechanisms:

- **Low Memory Killer Daemon (LMKD):** Assigns each process an `oom_adj_score` based on its priority class (foreground, visible, service, background). When free memory drops below defined thresholds (minfree levels), LMKD terminates processes starting from the highest `oom_adj_score`. This mechanism is purely reactive — it kills processes only after memory pressure is already critical, offering no predictive intelligence.

- **LRU-Based Process Cache:** Background applications are maintained in an LRU (Least Recently Used) list. When memory is needed, the least recently accessed process is evicted first. This approach ignores temporal usage patterns (e.g., a user always opens Maps at 8:30 AM), contextual signals (e.g., low battery means less gaming), and the variable cost of reloading different applications.

Recent Android 17 changes (Google, 2026) have introduced stricter per-app memory limits, but these still operate reactively and do not incorporate prediction.

### 1.4.3 The App Cold-Start Problem

When a cached application is evicted and the user returns to it, the system must perform a cold start — reloading the process, restoring UI state, reinitializing data, and re-establishing network connections. This process typically takes 1–3 seconds for standard applications and 3–6 seconds for complex AI-enabled applications, compared to a warm launch (50–200 ms from cache). Across 65 daily app switches, unnecessary cold starts can cumulatively add 2–4 minutes of wait time per day and consume 8–15% additional battery due to redundant disk I/O and CPU activity.

### 1.4.4 Opportunity for Prediction-Based Approaches

User behavior on mobile devices exhibits strong temporal and contextual regularity. Studies (Shin et al., 2012; Xu et al., 2024) have demonstrated that app usage follows predictable patterns based on time-of-day, day-of-week, preceding app sequence, and device state. This regularity presents an opportunity: if the system can predict which app the user is likely to open next, it can make proactive memory decisions — retaining predicted apps in cache, pre-warming them when resources permit, and preferentially evicting apps with low predicted reuse probability.

---

## 1.5 Literature Review

The following review covers 18 recent and foundational works across four research areas that underpin this project. Papers are organized thematically and cited in IEEE format.

### 1.5.1 App Usage Prediction

| # | Reference | Year | Key Contribution | Limitation Addressed by Our Work |
|---|---|---|---|---|
| [1] | C. Shin, J.-H. Hong, and A. K. Dey, "Prediction of Mobile Application Usage Using Contextual Information," *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.*, vol. 6, no. 4, 2012 | 2012 | Introduced contextual features (time, location, battery) for app prediction using Naive Bayes and decision trees; achieved ~72% accuracy | Does not integrate predictions into memory management; models are now outdated for modern app ecosystems |
| [2] | T. Xu, Y. Zhu, and R. Zhong, "Predicting Mobile App Usage with Heterogeneous Graph Attention Networks," *IEEE Trans. Mobile Computing*, vol. 23, no. 5, pp. 4512–4525, 2024 | 2024 | Used heterogeneous graph neural networks capturing app-app and app-context relationships; achieved 78% top-1 accuracy on LiveLab dataset | Requires GPU inference; too heavyweight for real-time on-device memory management loop |
| [3] | Z. Huang, J. Ma, and Y. Li, "AppUsage2Vec: Modeling Smartphone App Usage for Prediction," *Proc. IEEE ICDE*, pp. 1322–1333, 2019 | 2019 | Learned dense app embeddings from usage sequences using attention mechanism; outperformed Markov models by 15% | Does not address resource-constrained inference; no memory management integration |
| [4] | S. Zhao, J. Xu, and K. Ramamritham, "ContextSense: Context-Aware Mobile App Prediction via Deep Sequential Learning," *ACM UbiComp*, 2023 | 2023 | Combined LSTM with contextual features (time, location, battery) for sequential app prediction; 74% top-1 accuracy | Deep learning overhead makes it unsuitable for per-event memory management decisions |
| [5] | H. Pan, C. Xu, and Y. Zhang, "DeepApp: Predicting Personalized Smartphone App Usage via Context-aware Multi-task Learning," *ACM Trans. Intell. Syst. Technol.*, vol. 14, no. 2, 2023 | 2023 | Multi-task framework jointly predicting app usage and duration; personalized per-user models | Training complexity is prohibitive on-device; no connection to OS-level memory policies |

### 1.5.2 Mobile Memory Management & OS Optimization

| # | Reference | Year | Key Contribution | Limitation Addressed by Our Work |
|---|---|---|---|---|
| [6] | Google Android Open Source Project, "Low Memory Killer Daemon (LMKD)," *AOSP Documentation*, 2024 | 2024 | Official documentation of Android's PSI-based memory pressure detection and oom_adj_score-based process killing | Purely reactive; no prediction capability; kills processes after pressure is critical |
| [7] | S. Lee, J. Kim, and H. Bahn, "SWAM: Revisiting Swap and OOM Killer for Improving Application Responsiveness on Mobile Devices," *Samsung Research*, MobiSys 2023 Workshop, 2023 | 2023 | Proposed swap-clean technique and enhanced OOM killer considering relaunch cost of applications | Does not use ML prediction; relaunch cost is static, not context-dependent |
| [8] | Y. Chen, L. Yang, and W. Zhang, "Memory-Efficient Mobile Application Management via Predictive Caching," *IEEE Access*, vol. 12, pp. 34521–34534, 2024 | 2024 | Introduced frequency-based predictive caching for mobile apps; reduced cold starts by 22% | Uses only frequency heuristics, not context-aware ML; no thrashing detection |
| [9] | Android Developers, "App Hibernation and Standby Buckets," *Android Developer Documentation*, 2024 | 2024 | Classifies apps into usage buckets (active, working set, frequent, rare, restricted) to manage resources | Coarse-grained buckets updated hourly; cannot respond to per-event context changes |
| [10] | J. Park, Y. Lee, and S. Ha, "Learning-Based Memory Management for Mobile Devices," *Proc. IEEE/ACM MICRO*, pp. 289–301, 2023 | 2023 | Applied reinforcement learning to learn optimal memory eviction policies on real Android traces | RL training requires extensive online interaction; reward engineering is fragile |

### 1.5.3 Machine Learning for System Resource Management

| # | Reference | Year | Key Contribution | Limitation Addressed by Our Work |
|---|---|---|---|---|
| [11] | T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proc. ACM SIGKDD*, pp. 785–794, 2016 | 2016 | Foundational work on gradient boosted decision trees; fast, interpretable, handles mixed feature types | General-purpose ML framework; we apply it specifically to app prediction with calibrated confidence output |
| [12] | E. Ipek, O. Mutlu, and J. F. Martínez, "Self-Optimizing Memory Controllers: A Reinforcement Learning Approach," *Proc. IEEE/ACM ISCA*, pp. 39–50, 2008 | 2008 | Pioneering work on ML-driven memory controllers using RL for DRAM scheduling | Targets hardware-level DRAM scheduling, not application-level cache management on mobile |
| [13] | Z. Shi, X. Huang, and A. Jain, "Applying Deep Learning to Data Prefetching," *Proc. IEEE/ACM MICRO*, 2023 | 2023 | Used neural networks for data prefetching in processor caches; 15–20% IPC improvement | Operates at hardware cache level (L1/L2); our work targets application-level process cache |
| [14] | M. Hashemi, K. Swersky, and J. A. Smith, "Learning Memory Access Patterns," *Proc. ICML*, pp. 1919–1928, 2018 | 2018 | LSTM-based memory access prediction for hardware prefetching | Hardware-level prediction; does not consider user-level context or mobile constraints |

### 1.5.4 On-Device AI & Edge Computing

| # | Reference | Year | Key Contribution | Limitation Addressed by Our Work |
|---|---|---|---|---|
| [15] | J. Lin, W. M. Chen, and S. Han, "MCUNet: Tiny Deep Learning on IoT Devices," *Proc. NeurIPS*, 2020 | 2020 | Demonstrated efficient neural architecture search for memory-constrained devices (256 KB RAM) | Focuses on model architecture, not on OS-level memory management for multiple concurrent workloads |
| [16] | S. Laskaridis, A. Kouris, and N. D. Lane, "Adaptive Inference through Early-Exit Networks for Edge Devices," *Proc. ACM MobiSys*, 2024 | 2024 | Dynamic inference depth based on input complexity and available resources | Targets per-inference optimization; orthogonal to our cross-application memory management |
| [17] | Q. He, Y. Chen, and X. Wang, "On-Device LLM Inference: Memory Optimization Strategies for Mobile Deployment," *Proc. IEEE Edge Computing*, pp. 156–167, 2025 | 2025 | Analyzed memory bottlenecks of on-device LLMs; proposed KV-cache compression and model offloading | Focuses on single-model memory optimization; does not address multi-app contention |
| [18] | A. Dhakal, S. Kulkarni, and I. Stoica, "mLLM: Multi-dimensional Optimization for On-Device LLM Serving," *Proc. USENIX OSDI*, 2025 | 2025 | Elastic chunk-wise pipelines for on-device LLM serving with thermal-aware scheduling | Complements our work — our system can manage memory allocation across multiple such AI workloads |

### 1.5.5 Research Gap Identification

Based on the above review, the following research gap is identified:

**No existing system integrates real-time ML-based app usage prediction confidence as a weighted numerical factor into the OS-level cache eviction scoring formula, combined with multi-condition preload gating and automatic thrashing degradation, specifically targeting AI-heavy mobile environments where multiple AI model caches compete with user applications for limited RAM.**

Individual components (app prediction, memory management, on-device ML) have been studied in isolation, but their synthesis into a unified, adaptive memory management system remains unexplored.

---

## 1.6 Proposed Methodology

### 1.6.1 Methodology Overview

The proposed system follows a **predict-decide-act** pipeline that operates at each app-switch event on the mobile device. The methodology consists of five stages:

```
Stage 1: Context Collection
    ↓
Stage 2: Feature Engineering & App Prediction (ML)
    ↓
Stage 3: Cache Priority Scoring (with prediction integration)
    ↓
Stage 4: Memory Action Decision (eviction / preload / retain)
    ↓
Stage 5: Execution & Feedback Loop
```

### 1.6.2 Mathematical Formulation

#### Cache Priority Score

For each cached application *a* at time *t*, the cache priority score *S(a, t)* is computed as:

```
S(a, t) = w₁ · R(a, t) + w₂ · F(a, t) + w₃ · P(a, t) + w₄ · I(a) + w₅ · C(a, t) − w₆ · M(t)
```

Where:
- **R(a, t)** — Recency score: `1 / (1 + Δt_a / τ)`, where `Δt_a` is seconds since last access and `τ = 300s` is the decay constant
- **F(a, t)** — Frequency score: `count(a) / max(count(*))`, normalized access frequency
- **P(a, t)** — Prediction reuse probability: probability assigned to app *a* by the ML model's `predict_proba()` output (**key innovation**)
- **I(a)** — Application importance: static weight by category (communication=0.9, navigation=0.85, ..., gaming=0.3)
- **C(a, t)** — Context relevance: weighted combination of temporal match, network affinity, and AI-task correlation
- **M(t)** — Memory pressure penalty: `max(0, utilization(t) − 0.7) × 2.0`

Default weights: `w₁=0.25, w₂=0.20, w₃=0.25, w₄=0.15, w₅=0.15, w₆=0.30`

The weight `w₃=0.25` assigned to the prediction component is the highest alongside recency, reflecting the core innovation of this work — that ML prediction confidence should be a primary driver of eviction decisions.

#### Preload Decision Gate

A preload is approved if and only if ALL of the following conditions hold:

```
PRELOAD(a, t) = TRUE  iff:
    (1) confidence(a) > θ_conf       [θ_conf = 0.6]
    (2) RAM_free(t) > mem(a) + h      [h = 500 MB headroom]
    (3) battery(t) > β_min            [β_min = 20%]
    (4) thrashing_score(t) < γ        [γ = 0.7]
    (5) a ∉ cache(t)                  [not already cached]
```

#### Thrashing Score

```
T(t) = 0.4 · (reload_count / 3) + 0.4 · (miss_rate / 0.5) + 0.2 · (eviction_rate / 0.1)
T(t) = clamp(T(t), 0, 1)
```

When `T(t) > 0.7`, the system enters Safe Mode (preloading disabled, eviction thresholds raised). Safe Mode exits when `T(t) < 0.3` for 60 continuous seconds.

### 1.6.3 ML Model Selection Rationale

| Criterion | Markov Chain (Baseline) | XGBoost (Primary) | LSTM (Considered, Rejected) |
|---|---|---|---|
| Inference latency | <1 ms | ~2 ms | ~15 ms |
| Context-awareness | None (only current app) | Full (time, battery, RAM, history, AI state) | Full |
| Interpretability | High (transition matrix) | High (feature importance, SHAP) | Low (black box) |
| Training data requirement | Low | Moderate (5K–50K events) | High (100K+ events) |
| Mobile deployment feasibility | Trivial | Lightweight (tree-based, no GPU) | Requires TFLite conversion, GPU |
| Expected top-1 accuracy | 55–65% | 70–80% | 72–82% |

XGBoost is selected because it offers context-awareness and interpretability comparable to deep learning, with inference latency suitable for the per-event memory management loop, and no GPU dependency.

### 1.6.4 Evaluation Methodology

The system is evaluated via discrete-event simulation:

1. **Synthetic Data Generation:** 50,000 app-switch events across 5 user personas, each with distinct behavioral patterns
2. **Baseline Comparison:** 4 policies (NoOpt, LRU, StaticPreload, Proposed) × 5 personas × 3 RAM configs (4/6/8 GB) × 30 runs = 1,800 simulations
3. **Ablation Study:** 6 feature-removal experiments × 5 personas × 3 RAM configs × 30 runs = 2,700 simulations
4. **Statistical Validation:** Paired t-tests with p < 0.05, mean ± standard deviation, confidence intervals

**Key Performance Indicators (KPIs):**

| KPI | Definition | Target |
|---|---|---|
| Cache Hit Rate | % of app launches served from cache | > 75% (vs ~55% for LRU) |
| Average Launch Latency | Mean app start time across all launches | < 400 ms (vs ~900 ms for LRU) |
| Thrashing Events | Count of evict→reload→evict cycles per session | < 3 (vs ~8 for LRU) |
| Preload Success Rate | % of preloaded apps actually used within 5 min | > 60% |
| Prediction Accuracy (Top-1) | ML model correct next-app prediction rate | > 70% |

---

## 1.7 Module Descriptions

### Module 1: Context Collection Layer

| Property | Description |
|---|---|
| **Purpose** | Captures real-time device state at each app-switch event and packages it into a normalized ContextSnapshot object |
| **Input** | Raw device signals: foreground app name, system clock, battery API, memory info API, network status, AI task registry |
| **Output** | `ContextSnapshot` data object containing 12 fields (current_app, prev_apps[1..3], hour, day_of_week, battery, is_charging, ram_free_mb, ram_total_mb, network_type, ai_task_active, screen_state, timestamp) |
| **Key Classes** | `ContextCollector`, `SystemMonitor`, `AppHistoryBuffer` |
| **Technology** | Python standard library; Android `ActivityManager`, `BatteryManager` APIs (in production); simulated via JSON config (in prototype) |
| **Trigger** | Invoked on every app foreground transition event |

### Module 2: Feature Engineering & Prediction Layer

| Property | Description |
|---|---|
| **Purpose** | Transforms raw context into ML features, runs the prediction model, and outputs the predicted next app with calibrated confidence |
| **Input** | `ContextSnapshot` from Module 1 |
| **Output** | `PredictionResult` containing predicted_app, confidence (0.0–1.0), top_k list of (app, probability) tuples, model_used |
| **Key Classes** | `FeatureEngine`, `MarkovPredictor`, `XGBoostPredictor` |
| **Technology** | scikit-learn (LabelEncoder, StandardScaler), XGBoost (`XGBClassifier` with `multi:softprob` objective) |
| **Sub-components** | (a) Feature encoder: categorical encoding via LabelEncoder, derived features (ram_utilization, is_peak_hour, is_weekend, session_depth, time_since_last_switch), (b) Markov model: first-order transition matrix as fallback, (c) XGBoost model: primary predictor with `predict_proba()` for confidence scores |

### Module 3: Memory Decision Layer

| Property | Description |
|---|---|
| **Purpose** | Computes cache priority scores for all cached apps, determines eviction/preload/retain actions, and monitors for thrashing |
| **Input** | `PredictionResult` from Module 2 + current `MemoryState` + `ContextSnapshot` |
| **Output** | `MemoryActionPlan` specifying which apps to evict, preload, retain, or compress, and whether the system is in NORMAL or SAFE mode |
| **Key Classes** | `CacheScoringEngine`, `PreloadDecisionEngine`, `ThrashingDetector`, `MemoryDecisionController` |
| **Sub-components** | (a) Cache Scorer: applies the 6-factor weighted scoring formula with prediction integration, (b) Preload Gate: checks 5 conditions before approving preload, (c) Thrashing Detector: monitors eviction/reload patterns in a sliding 120-second window |

### Module 4: Memory Action Executor

| Property | Description |
|---|---|
| **Purpose** | Applies the decided memory actions to the memory state (simulated or real) |
| **Input** | `MemoryActionPlan` from Module 3 + current `MemoryState` |
| **Output** | Updated `MemoryState` reflecting evictions, preloads, and retentions |
| **Key Classes** | `MemoryActionExecutor`, `MemoryState`, `AppCacheEntry` |
| **Operations** | Evict: remove app from cache dict, add freed memory to ram_free. Preload: create AppCacheEntry, subtract memory from ram_free. Retain: no-op (app stays in cache). |

### Module 5: Simulation Engine

| Property | Description |
|---|---|
| **Purpose** | Replays user sessions as discrete events, applying a given memory policy at each step, and collects comprehensive metrics |
| **Input** | `UserSession` (ordered list of app-switch events), `Policy` (one of NoOpt, LRU, StaticPreload, Proposed), RAM configuration |
| **Output** | `BenchmarkResult` containing all KPIs: cache hit rate, avg launch latency, thrashing count, eviction count, preload success rate, memory utilization timeline |
| **Key Classes** | `SimulationEngine`, `MetricsCollector`, `BenchmarkRunner` |
| **Technology** | Python, pandas, scipy (for statistical tests), joblib (for parallel simulation runs) |

### Module 6: Visualization Dashboard

| Property | Description |
|---|---|
| **Purpose** | Provides real-time visual monitoring of all system layers and benchmark comparison views |
| **Input** | Outputs from all modules: ContextSnapshot, PredictionResult, MemoryState, MemoryActionPlan, BenchmarkResult |
| **Output** | Interactive web dashboard with memory map, prediction timeline, KPI gauges, benchmark comparison charts |
| **Key Classes** | Streamlit pages, Plotly charts |
| **Technology** | Streamlit, Plotly, Matplotlib, Seaborn |

---

## 1.8 Comparison with Existing Approaches

| Feature | NoOpt (FIFO) | LRU | Android LMKD | Static Preload | **Proposed System** |
|---|---|---|---|---|---|
| Eviction strategy | First-in-first-out | Least recently used | oom_adj_score + PSI pressure | LRU + preload top-3 apps | ML-prediction-weighted multi-factor scoring |
| Prediction-aware | No | No | No | Partially (static top-N) | Yes (real-time per-event ML prediction) |
| Context-aware | No | No | No | No | Yes (time, battery, RAM, AI state, history) |
| Preload capability | No | No | No | Yes (fixed set) | Yes (confidence-gated, resource-gated) |
| Thrashing detection | No | No | PSI-based (reactive) | No | Yes (proactive, with auto-degradation) |
| Adaptivity | None | Recency only | Priority class only | None | Full (learns per-user patterns, adapts to context) |
| Expected cache hit rate | ~40% | ~55% | ~58% | ~62% | **>75%** |
| Expected avg latency | ~1200 ms | ~900 ms | ~850 ms | ~750 ms | **<400 ms** |

---

## 1.9 Novelty Statement

The novel contributions of this work are:

1. **Prediction-Integrated Cache Scoring:** No prior work uses ML model prediction probability as a weighted numerical factor in the cache eviction score alongside traditional metrics (recency, frequency, importance). This creates a forward-looking eviction policy that considers future access probability, not just past access history.

2. **Multi-Condition Preload Gating:** Unlike static preloading (which blindly preloads top-N apps), the proposed system gates preloading on five simultaneous conditions — prediction confidence, available RAM, battery level, thrashing status, and cache residency — preventing resource waste from speculative preloading.

3. **Automatic Thrashing Degradation:** The system monitors its own eviction-reload patterns in real-time and automatically enters a protective Safe Mode when thrashing is detected, preventing the feedback loop where preloading causes more evictions which cause more cold starts.

4. **AI-Workload Awareness:** The system treats AI model caches (on-device LLM weights, computer vision models) as first-class memory citizens with higher importance scores and eviction resistance, recognizing their disproportionately large reload cost.

---

## 2. High-Level Design (HLD)

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MOBILE DEVICE                                │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │   CONTEXT     │     │   PREDICTION     │     │   MEMORY       │  │
│  │   LAYER       │────▶│   LAYER          │────▶│   DECISION     │  │
│  │              │     │                  │     │   LAYER         │  │
│  │ • App State   │     │ • Feature Engine │     │ • Cache Scorer  │  │
│  │ • System State│     │ • Markov Model   │     │ • Preload Gate  │  │
│  │ • Temporal    │     │ • XGBoost Model  │     │ • Thrash Guard  │  │
│  │ • Resource    │     │ • Confidence     │     │ • Action Exec   │  │
│  │   Monitor     │     │   Calibration    │     │                │  │
│  └──────────────┘     └──────────────────┘     └───────┬────────┘  │
│                                                         │          │
│                                                         ▼          │
│                                               ┌────────────────┐   │
│                                               │ MEMORY ACTION  │   │
│                                               │ EXECUTOR       │   │
│                                               │                │   │
│                                               │ • Evict        │   │
│                                               │ • Retain       │   │
│                                               │ • Preload      │   │
│                                               │ • Compress     │   │
│                                               └────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    SIMULATION ENGINE                          │   │
│  │  • User session replay  • Memory state tracking              │   │
│  │  • Latency computation  • Baseline comparison                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    MONITORING / DASHBOARD                     │   │
│  │  • Real-time context  • Memory map  • KPI report             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Descriptions

| Component | Responsibility | Input | Output |
|---|---|---|---|
| **Context Layer** | Collects and normalizes all device signals | Raw device APIs / simulated events | `ContextSnapshot` object |
| **Prediction Layer** | Predicts next app with confidence score | `ContextSnapshot` | `PredictionResult` (app, confidence, top-K) |
| **Memory Decision Layer** | Computes cache scores, preload decisions, thrashing status | `PredictionResult` + `MemoryState` | `MemoryActionPlan` |
| **Memory Action Executor** | Applies decisions to simulated or real memory | `MemoryActionPlan` | Updated `MemoryState` |
| **Simulation Engine** | Replays user sessions and tracks all metrics | `UserSession` + `Policy` | `BenchmarkResult` |
| **Dashboard** | Visualizes all layers in real-time | All layer outputs | Interactive UI |

### 2.3 Data Flow (HLD)

```
STEP 1: Context Collection
─────────────────────────
    t = current timestamp
    Collect:
        current_app         ← foreground app name
        prev_apps[1..3]     ← last 3 apps from history buffer
        hour_of_day         ← extract from timestamp
        day_of_week         ← extract from timestamp
        battery_level       ← system API / simulated
        is_charging         ← system API / simulated
        ram_free_mb         ← system API / simulated
        ram_total_mb        ← device config
        network_type        ← system API / simulated
        ai_task_active      ← boolean (is any AI model loaded?)
        screen_state        ← on/off

    Output → ContextSnapshot

STEP 2: Prediction
──────────────────
    Input ← ContextSnapshot
    Feature vector ← FeatureEngine.transform(ContextSnapshot)
    
    Markov prediction    ← MarkovModel.predict(current_app)
    XGBoost prediction   ← XGBoostModel.predict(feature_vector)
    
    Final prediction     ← XGBoost (primary), Markov (fallback)
    Confidence           ← model.predict_proba() top-1 probability

    Output → PredictionResult {
        predicted_app: str,
        confidence: float (0.0 - 1.0),
        top_k: List[(app, probability)]
    }

STEP 3: Memory Decision
───────────────────────
    Input ← PredictionResult + current MemoryState

    3a. Cache Scoring
        For each app in cache:
            score = compute_cache_score(app, PredictionResult, MemoryState)
        Sort by score ascending → eviction order

    3b. Eviction Check
        If ram_free < PRESSURE_THRESHOLD:
            evict lowest-scored apps until ram_free >= PRESSURE_THRESHOLD

    3c. Preload Decision
        If prediction.confidence > PRELOAD_CONF_THRESHOLD
        AND ram_free > predicted_app.memory_requirement
        AND battery > BATTERY_MIN_THRESHOLD
        AND thrashing_detector.is_safe():
            → preload predicted_app

    3d. Thrashing Check
        Update thrashing_detector with latest eviction/reload events
        If thrashing detected:
            → enter safe mode (disable preloading, tighten eviction)

    Output → MemoryActionPlan {
        evictions: List[app],
        preloads: List[app],
        retentions: List[app],
        mode: NORMAL | SAFE
    }

STEP 4: Execution
──────────────────
    Apply MemoryActionPlan to MemoryState
    Log all actions to database
    Update metrics (cache hits, misses, latency, thrashing count)

STEP 5: Next Event
──────────────────
    Advance to next app-switch event in session
    Record whether the opened app was in cache (warm) or not (cold)
    Compute launch latency accordingly
    Repeat from STEP 1
```

---

## 3. Low-Level Design (LLD)

### 3.1 Class Diagram

```
┌─────────────────────────────────────────────┐
│              ContextSnapshot                 │
├─────────────────────────────────────────────┤
│ + current_app: str                          │
│ + prev_apps: List[str]                      │
│ + hour: int                                 │
│ + day_of_week: int                          │
│ + battery: float                            │
│ + is_charging: bool                         │
│ + ram_free_mb: float                        │
│ + ram_total_mb: float                       │
│ + network_type: str                         │
│ + ai_task_active: bool                      │
│ + screen_state: str                         │
│ + timestamp: datetime                       │
├─────────────────────────────────────────────┤
│ + to_feature_vector() → np.ndarray          │
│ + to_dict() → dict                          │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│              FeatureEngine                   │
├─────────────────────────────────────────────┤
│ - label_encoders: Dict[str, LabelEncoder]   │
│ - scaler: StandardScaler                    │
├─────────────────────────────────────────────┤
│ + fit(sessions: List[Session]) → self       │
│ + transform(ctx: ContextSnapshot) → ndarray │
│ + get_feature_names() → List[str]           │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│           PredictionResult                   │
├─────────────────────────────────────────────┤
│ + predicted_app: str                        │
│ + confidence: float                         │
│ + top_k: List[Tuple[str, float]]            │
│ + model_used: str                           │
│ + timestamp: datetime                       │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            MarkovPredictor                   │
├─────────────────────────────────────────────┤
│ - transition_matrix: Dict[str, Dict[str, f]]│
│ - app_counts: Dict[str, int]                │
├─────────────────────────────────────────────┤
│ + fit(sessions: List[Session]) → self       │
│ + predict(current_app: str)                 │
│       → PredictionResult                    │
│ + get_transition_prob(from, to) → float     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           XGBoostPredictor                   │
├─────────────────────────────────────────────┤
│ - model: XGBClassifier                      │
│ - feature_engine: FeatureEngine             │
│ - class_labels: List[str]                   │
├─────────────────────────────────────────────┤
│ + fit(X: ndarray, y: ndarray) → self        │
│ + predict(ctx: ContextSnapshot)             │
│       → PredictionResult                    │
│ + predict_proba(ctx) → Dict[str, float]     │
│ + feature_importance() → Dict[str, float]   │
│ + save(path: str) → None                    │
│ + load(path: str) → self                    │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│              AppCacheEntry                   │
├─────────────────────────────────────────────┤
│ + app_name: str                             │
│ + memory_mb: float                          │
│ + cache_mb: float                           │
│ + last_accessed: datetime                   │
│ + access_count: int                         │
│ + category: str                             │
│ + importance: float                         │
│ + cold_launch_ms: int                       │
│ + warm_launch_ms: int                       │
│ + is_ai_workload: bool                      │
│ + ai_cache_mb: float                        │
├─────────────────────────────────────────────┤
│ + time_since_access() → float               │
│ + total_memory() → float                    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              MemoryState                     │
├─────────────────────────────────────────────┤
│ + total_ram_mb: float                       │
│ + cached_apps: Dict[str, AppCacheEntry]     │
│ + foreground_app: str                       │
│ + system_reserved_mb: float                 │
├─────────────────────────────────────────────┤
│ + free_ram() → float                        │
│ + used_ram() → float                        │
│ + utilization() → float                     │
│ + is_app_cached(app: str) → bool            │
│ + add_to_cache(entry: AppCacheEntry) → None │
│ + remove_from_cache(app: str) → AppCacheEntry│
│ + get_all_entries() → List[AppCacheEntry]   │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│          CacheScoringEngine                  │
├─────────────────────────────────────────────┤
│ - w_recency: float = 0.25                   │
│ - w_frequency: float = 0.20                 │
│ - w_prediction: float = 0.25                │
│ - w_importance: float = 0.15                │
│ - w_context: float = 0.15                   │
├─────────────────────────────────────────────┤
│ + compute_score(                            │
│       entry: AppCacheEntry,                 │
│       prediction: PredictionResult,         │
│       memory_state: MemoryState             │
│   ) → float                                │
│ + rank_all(                                 │
│       memory_state: MemoryState,            │
│       prediction: PredictionResult          │
│   ) → List[Tuple[str, float]]              │
│ + get_eviction_candidates(                  │
│       memory_state, prediction,             │
│       required_mb: float                    │
│   ) → List[str]                            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│          PreloadDecisionEngine               │
├─────────────────────────────────────────────┤
│ - confidence_threshold: float = 0.6         │
│ - battery_min: float = 20.0                 │
│ - ram_headroom_mb: float = 500.0            │
│ - preload_history: List[PreloadEvent]       │
├─────────────────────────────────────────────┤
│ + should_preload(                           │
│       prediction: PredictionResult,         │
│       memory_state: MemoryState,            │
│       context: ContextSnapshot,             │
│       thrashing_status: bool                │
│   ) → Tuple[bool, str]                     │
│ + record_outcome(app, was_used) → None      │
│ + success_rate() → float                    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│          ThrashingDetector                   │
├─────────────────────────────────────────────┤
│ - event_window: deque (max_size=50)         │
│ - reload_threshold: int = 3                 │
│ - window_seconds: float = 120.0             │
│ - cache_miss_rate_threshold: float = 0.5    │
│ - is_safe_mode: bool = False                │
├─────────────────────────────────────────────┤
│ + record_event(event: MemoryEvent) → None   │
│ + is_thrashing() → bool                     │
│ + is_safe() → bool                          │
│ + get_thrashing_score() → float             │
│ + enter_safe_mode() → None                  │
│ + exit_safe_mode() → None                   │
│ + reset() → None                            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│          MemoryActionPlan                    │
├─────────────────────────────────────────────┤
│ + evictions: List[str]                      │
│ + preloads: List[str]                       │
│ + retentions: List[str]                     │
│ + compressions: List[str]                   │
│ + mode: Enum(NORMAL, SAFE)                  │
│ + timestamp: datetime                       │
│ + reasoning: Dict[str, str]                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│         MemoryDecisionController             │
├─────────────────────────────────────────────┤
│ - cache_scorer: CacheScoringEngine          │
│ - preload_engine: PreloadDecisionEngine     │
│ - thrashing_detector: ThrashingDetector     │
│ - pressure_threshold_mb: float = 800.0      │
├─────────────────────────────────────────────┤
│ + decide(                                   │
│       context: ContextSnapshot,             │
│       prediction: PredictionResult,         │
│       memory_state: MemoryState             │
│   ) → MemoryActionPlan                     │
│ + apply_plan(                               │
│       plan: MemoryActionPlan,               │
│       memory_state: MemoryState             │
│   ) → MemoryState                          │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            SimulationEngine                  │
├─────────────────────────────────────────────┤
│ - predictor: XGBoostPredictor               │
│ - controller: MemoryDecisionController      │
│ - memory_state: MemoryState                 │
│ - metrics: MetricsCollector                 │
│ - app_profiles: Dict[str, AppProfile]       │
├─────────────────────────────────────────────┤
│ + run_session(session: UserSession) → Result│
│ + run_benchmark(                            │
│       sessions: List[UserSession],          │
│       policies: List[Policy]                │
│   ) → BenchmarkResult                      │
│ + step(event: AppSwitchEvent) → StepResult  │
│ + reset() → None                            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            MetricsCollector                  │
├─────────────────────────────────────────────┤
│ - launch_times: List[float]                 │
│ - cache_hits: int                           │
│ - cache_misses: int                         │
│ - eviction_count: int                       │
│ - preload_count: int                        │
│ - preload_used: int                         │
│ - thrashing_events: int                     │
│ - memory_snapshots: List[MemorySnapshot]    │
├─────────────────────────────────────────────┤
│ + record_launch(app, latency, was_cached)   │
│ + record_eviction(app)                      │
│ + record_preload(app, was_used_later)       │
│ + record_thrashing_event()                  │
│ + record_memory_snapshot(state)             │
│ + cache_hit_rate() → float                  │
│ + avg_launch_latency() → float              │
│ + preload_success_rate() → float            │
│ + to_dict() → dict                          │
│ + to_dataframe() → pd.DataFrame             │
└─────────────────────────────────────────────┘
```

### 3.2 Interaction Sequence (LLD)

```
User opens new app (AppSwitchEvent)
    │
    ├──▶ ContextCollector.capture()
    │        │
    │        ├── Read current_app, prev_apps from app_history_buffer
    │        ├── Read battery, ram_free, network from SystemMonitor
    │        ├── Compute hour, day_of_week from system clock
    │        └── Return ContextSnapshot
    │
    ├──▶ FeatureEngine.transform(context_snapshot)
    │        │
    │        ├── Encode categorical features (app names, network, category)
    │        ├── Normalize numerical features (battery, ram, hour)
    │        ├── Compute derived features (ram_utilization, is_peak_hour)
    │        └── Return feature_vector: np.ndarray [1 x N_features]
    │
    ├──▶ XGBoostPredictor.predict(context_snapshot)
    │        │
    │        ├── Get feature_vector from FeatureEngine
    │        ├── Run model.predict_proba(feature_vector)
    │        ├── Extract top-K predictions with probabilities
    │        ├── Primary prediction = argmax(probabilities)
    │        ├── Confidence = max(probabilities)
    │        └── Return PredictionResult
    │
    ├──▶ MemoryDecisionController.decide(context, prediction, memory_state)
    │        │
    │        ├── CacheScoringEngine.rank_all(memory_state, prediction)
    │        │       │
    │        │       ├── For each app in cache:
    │        │       │     recency    = 1.0 / (1 + seconds_since_last_access / 300)
    │        │       │     frequency  = min(access_count / max_access_count, 1.0)
    │        │       │     pred_reuse = prediction.get_probability(app_name) or 0.0
    │        │       │     importance = app_importance_lookup[app.category]
    │        │       │     context_rel = compute_context_relevance(app, context)
    │        │       │     pressure   = max(0, 1.0 - memory_state.free_ram / total_ram)
    │        │       │     
    │        │       │     score = 0.25 * recency
    │        │       │           + 0.20 * frequency
    │        │       │           + 0.25 * pred_reuse      ← KEY INNOVATION
    │        │       │           + 0.15 * importance
    │        │       │           + 0.15 * context_rel
    │        │       │           - 0.30 * pressure
    │        │       │
    │        │       └── Return sorted list [(app, score), ...] ascending
    │        │
    │        ├── Check if eviction needed:
    │        │       If free_ram < PRESSURE_THRESHOLD:
    │        │           evict apps from bottom of ranked list
    │        │           until free_ram >= PRESSURE_THRESHOLD
    │        │
    │        ├── PreloadDecisionEngine.should_preload(...)
    │        │       │
    │        │       ├── Check: confidence > 0.6?
    │        │       ├── Check: ram_free > app_memory + HEADROOM?
    │        │       ├── Check: battery > 20%?
    │        │       ├── Check: thrashing_detector.is_safe()?
    │        │       ├── Check: app not already in cache?
    │        │       └── Return (True/False, reason_string)
    │        │
    │        ├── ThrashingDetector.is_thrashing()
    │        │       │
    │        │       ├── Count reload events for same app within window
    │        │       ├── Compute cache_miss_rate over last N events
    │        │       ├── If reload_count > threshold OR miss_rate > threshold:
    │        │       │       enter_safe_mode()
    │        │       │       Return True
    │        │       └── Return False
    │        │
    │        └── Assemble MemoryActionPlan
    │
    ├──▶ MemoryActionExecutor.execute(plan, memory_state)
    │        │
    │        ├── For each eviction: remove app from cache, free memory
    │        ├── For each preload: load app into cache, allocate memory
    │        ├── Update memory_state
    │        └── Return updated MemoryState
    │
    └──▶ MetricsCollector.record(...)
             │
             ├── Was the opened app in cache? → cache_hit++ or cache_miss++
             ├── Launch latency = warm_launch_ms if cached, cold_launch_ms if not
             ├── Record eviction count, preload count
             └── Update running averages
```

---

## 4. ML Pipeline — Detailed

### 4.1 Feature Engineering

#### Raw Features (collected per app-switch event)

| Feature | Type | Source | Range |
|---|---|---|---|
| `current_app` | Categorical | App history buffer | 15-25 unique apps |
| `prev_app_1` | Categorical | History buffer [t-1] | Same set |
| `prev_app_2` | Categorical | History buffer [t-2] | Same set |
| `prev_app_3` | Categorical | History buffer [t-3] | Same set |
| `hour_of_day` | Integer | System clock | 0-23 |
| `day_of_week` | Integer | System clock | 0-6 (Mon-Sun) |
| `battery_level` | Float | System / simulated | 0-100 |
| `is_charging` | Binary | System / simulated | 0 or 1 |
| `ram_free_mb` | Float | System / simulated | 0 - total_ram |
| `network_type` | Categorical | System / simulated | wifi, mobile, none |
| `ai_task_active` | Binary | System / simulated | 0 or 1 |
| `app_category` | Categorical | Lookup table | social, productivity, media, etc. |

#### Derived Features (computed during feature engineering)

| Feature | Computation | Rationale |
|---|---|---|
| `ram_utilization` | `1.0 - (ram_free / ram_total)` | Normalized pressure indicator |
| `is_peak_hour` | `1 if hour in [8-10, 12-14, 18-22] else 0` | Usage patterns differ by time |
| `is_weekend` | `1 if day_of_week >= 5 else 0` | Weekend usage differs |
| `session_depth` | Count of app switches in current session | Early vs deep session behavior differs |
| `time_since_last_switch` | Seconds since previous app switch | Rapid switching = different pattern |
| `battery_bucket` | `low (<20), medium (20-60), high (>60)` | Coarse battery state |
| `prev_app_category` | Category of prev_app_1 | Category-level transitions |

#### Encoding Strategy

| Feature Type | Encoding Method | Notes |
|---|---|---|
| App names | LabelEncoder → integer | XGBoost handles integers natively |
| App categories | LabelEncoder → integer | 8-10 categories |
| Network type | LabelEncoder → integer | 3 values |
| Hour of day | Kept as integer | XGBoost can learn non-linear splits |
| Numerical features | StandardScaler (optional) | XGBoost is tree-based, scaling not strictly required |

#### Target Variable

```
y = next_app (LabelEncoded integer)
```

Multi-class classification with 15-25 classes.

---

### 4.2 Model 1: Markov Chain (Baseline)

#### What It Is
A first-order Markov chain models app-to-app transition probabilities. Given the current app, it predicts the most likely next app based on historical transition frequencies.

#### Construction

```python
# Build transition matrix from training sessions
transition_counts = defaultdict(Counter)

for session in training_sessions:
    for i in range(len(session) - 1):
        current = session[i].app
        next_app = session[i + 1].app
        transition_counts[current][next_app] += 1

# Normalize to probabilities
transition_probs = {}
for app, next_counts in transition_counts.items():
    total = sum(next_counts.values())
    transition_probs[app] = {
        next_app: count / total
        for next_app, count in next_counts.items()
    }
```

#### Prediction

```python
def predict(self, current_app: str) -> PredictionResult:
    probs = self.transition_probs.get(current_app, {})
    if not probs:
        return PredictionResult(predicted_app="unknown", confidence=0.0)
    
    sorted_preds = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return PredictionResult(
        predicted_app=sorted_preds[0][0],
        confidence=sorted_preds[0][1],
        top_k=sorted_preds[:5],
        model_used="markov"
    )
```

#### Limitations
- Uses only current app → ignores all context (time, battery, RAM, history depth)
- Cannot learn complex patterns (Camera → Gallery only on weekends)
- Serves as the baseline to show XGBoost's improvement

#### Expected Accuracy: Top-1: 55-65% | Top-3: 75-80%

---

### 4.3 Model 2: XGBoost (Primary Model)

#### Why XGBoost
- Handles mixed feature types natively
- Fast inference (critical for mobile/memory management)
- Built-in feature importance (useful for ablation study and patent explanation)
- Doesn't require GPU
- Robust to class imbalance

#### Training

```python
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=num_apps,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    random_state=42
)

# Stratified K-Fold for robust evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    accuracies.append(accuracy_score(y_val, y_pred))

print(f"Mean accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
```

#### Prediction with Confidence

```python
def predict(self, context: ContextSnapshot) -> PredictionResult:
    features = self.feature_engine.transform(context)
    probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
    
    top_k_indices = np.argsort(probabilities)[::-1][:5]
    top_k = [
        (self.class_labels[i], float(probabilities[i]))
        for i in top_k_indices
    ]
    
    return PredictionResult(
        predicted_app=top_k[0][0],
        confidence=top_k[0][1],
        top_k=top_k,
        model_used="xgboost"
    )
```

#### Hyperparameter Tuning

```python
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3]
}

search = RandomizedSearchCV(
    XGBClassifier(objective='multi:softprob', num_class=num_apps),
    param_distributions=param_dist,
    n_iter=50, scoring='accuracy', cv=3, random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
```

#### Expected Accuracy: Top-1: 70-80% (synthetic), 60-70% (real) | Top-3: 85-92% (synthetic)

---

### 4.4 Model Evaluation

#### Metrics to Report

| Metric | What It Measures |
|---|---|
| **Top-1 Accuracy** | Is the predicted app exactly right? |
| **Top-3 Accuracy** | Is the correct app in top 3 predictions? |
| **Weighted F1** | Handles class imbalance |
| **Confusion Matrix** | Which apps are confused with which? |
| **Confidence Calibration** | When model says 70%, is it right 70% of the time? |

#### Ablation Study (Required for IEEE Paper)

| Experiment | Features Removed | Expected Impact |
|---|---|---|
| Full model | None | Best accuracy |
| No temporal | hour, day_of_week, is_weekend, is_peak | ~5-8% drop |
| No resource | battery, ram_free, is_charging | ~2-4% drop |
| No history | prev_app_1/2/3 | ~10-15% drop |
| No category | app_category, prev_app_category | ~2-3% drop |
| Current app only | Everything except current_app | Drops to Markov baseline |

---

## 5. Core Algorithms — Pseudocode

### 5.1 Cache Priority Scoring

```
FUNCTION compute_cache_score(entry, prediction, memory_state):
    
    # Recency: exponential decay based on time since last access
    seconds_idle = (now - entry.last_accessed).total_seconds()
    recency = 1.0 / (1.0 + seconds_idle / 300.0)
    
    # Frequency: normalized access count
    max_count = max(e.access_count for e in memory_state.all_entries())
    frequency = entry.access_count / max(max_count, 1)
    
    # Prediction reuse probability ← KEY INNOVATION
    pred_reuse = 0.0
    for app, prob in prediction.top_k:
        if app == entry.app_name:
            pred_reuse = prob
            break
    
    # App importance: lookup by category
    importance_map = {
        "communication": 0.9,
        "navigation": 0.85,
        "productivity": 0.8,
        "social": 0.6,
        "media": 0.5,
        "gaming": 0.3,
        "utility": 0.7,
    }
    importance = importance_map.get(entry.category, 0.5)
    
    # Context relevance
    context_relevance = compute_context_relevance(entry, context)
    
    # Memory pressure penalty
    utilization = 1.0 - (memory_state.free_ram() / memory_state.total_ram_mb)
    pressure_penalty = max(0, utilization - 0.7) * 2.0
    
    # Final weighted score
    score = (0.25 * recency
           + 0.20 * frequency
           + 0.25 * pred_reuse
           + 0.15 * importance
           + 0.15 * context_relevance
           - 0.30 * pressure_penalty)
    
    RETURN score
```

### 5.2 Preload Decision Gate

```
FUNCTION should_preload(prediction, memory_state, context, thrashing_detector):
    
    app = prediction.predicted_app
    conf = prediction.confidence
    
    IF conf < 0.6:                          RETURN (False, "low confidence")
    IF memory_state.is_app_cached(app):     RETURN (False, "already cached")
    
    required_mb = app_profiles[app].memory_mb + 500  # headroom
    IF memory_state.free_ram() < required_mb:
        RETURN (False, "insufficient RAM")
    
    IF context.battery < 20 AND NOT context.is_charging:
        RETURN (False, "low battery")
    
    IF NOT thrashing_detector.is_safe():
        RETURN (False, "thrashing active")
    
    RETURN (True, "preloading approved")
```

### 5.3 Thrashing Detection

```
FUNCTION detect_thrashing(event_window):
    
    # Remove events older than 120 seconds
    cutoff = now - 120s
    purge events before cutoff
    
    # Check 1: Same app evicted and reloaded
    evicted = [e.app for e in window IF e.type == "eviction"]
    reloaded = [e.app for e in window IF e.type == "cold_load"]
    reload_count = count(app in reloaded IF app in evicted)
    
    # Check 2: Cache miss rate
    all_loads = [e for e in window IF e.type in ("cold_load", "warm_load")]
    miss_rate = count(cold_loads) / max(len(all_loads), 1)
    
    # Check 3: Eviction spike rate
    eviction_rate = len(evicted) / 120.0
    
    # Combined thrashing score
    score = 0.4 * (reload_count / 3.0)
          + 0.4 * (miss_rate / 0.5)
          + 0.2 * (eviction_rate / 0.1)
    score = clamp(score, 0.0, 1.0)
    
    IF score > 0.7: enter_safe_mode()
    IF score < 0.3 AND is_safe_mode: exit_safe_mode()
    
    RETURN score > 0.7
```

### 5.4 Safe Mode

```
WHEN safe_mode IS ACTIVE:
    preload_enabled         = False
    foreground_protection   = True
    eviction_threshold      = 0.15  (normal: 0.0)
    target_free_ram        *= 1.5
    
    EXIT when thrashing_score < 0.3 for 60 continuous seconds
```

---

## 6. Database Schema

```sql
CREATE TABLE app_profiles (
    app_name        TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    cold_launch_ms  INTEGER NOT NULL,
    warm_launch_ms  INTEGER NOT NULL,
    memory_mb       REAL NOT NULL,
    cache_mb        REAL NOT NULL,
    importance      REAL DEFAULT 0.5,
    is_ai_workload  INTEGER DEFAULT 0,
    ai_cache_mb     REAL DEFAULT 0.0
);

CREATE TABLE sessions (
    session_id      TEXT PRIMARY KEY,
    persona         TEXT NOT NULL,
    start_time      TIMESTAMP NOT NULL,
    end_time        TIMESTAMP,
    event_count     INTEGER,
    ram_config_mb   INTEGER NOT NULL
);

CREATE TABLE app_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    timestamp       TIMESTAMP NOT NULL,
    app_name        TEXT NOT NULL,
    prev_app        TEXT,
    battery         REAL,
    is_charging     INTEGER,
    ram_free_mb     REAL,
    network_type    TEXT,
    ai_task_active  INTEGER DEFAULT 0,
    hour_of_day     INTEGER,
    day_of_week     INTEGER
);

CREATE TABLE predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER REFERENCES app_events(event_id),
    model_used      TEXT NOT NULL,
    predicted_app   TEXT NOT NULL,
    confidence      REAL NOT NULL,
    actual_app      TEXT NOT NULL,
    was_correct     INTEGER NOT NULL,
    top_3_apps      TEXT,
    timestamp       TIMESTAMP NOT NULL
);

CREATE TABLE memory_actions (
    action_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER REFERENCES app_events(event_id),
    action_type     TEXT NOT NULL,
    target_app      TEXT NOT NULL,
    reason          TEXT,
    ram_before_mb   REAL,
    ram_after_mb    REAL,
    policy          TEXT NOT NULL,
    mode            TEXT DEFAULT 'normal',
    timestamp       TIMESTAMP NOT NULL
);

CREATE TABLE benchmark_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT REFERENCES sessions(session_id),
    policy              TEXT NOT NULL,
    ram_config_mb       INTEGER NOT NULL,
    avg_launch_latency  REAL,
    cache_hit_rate      REAL,
    cache_miss_count    INTEGER,
    thrashing_events    INTEGER,
    eviction_count      INTEGER,
    preload_count       INTEGER,
    preload_success     INTEGER,
    memory_utilization  REAL,
    total_events        INTEGER,
    run_timestamp       TIMESTAMP NOT NULL
);

CREATE TABLE ablation_results (
    ablation_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name     TEXT NOT NULL,
    run_id              INTEGER REFERENCES benchmark_runs(run_id),
    features_removed    TEXT,
    accuracy_top1       REAL,
    accuracy_top3       REAL,
    avg_launch_latency  REAL,
    cache_hit_rate      REAL,
    thrashing_events    INTEGER,
    timestamp           TIMESTAMP NOT NULL
);
```

---

## 7. API Contracts

### 7.1 Internal Module Interfaces

```python
# Context Layer
def capture_context(app_history: List[str], system_state: dict) -> ContextSnapshot

# Prediction Layer
def predict_next_app(context: ContextSnapshot, model: str = "xgboost") -> PredictionResult

# Decision Layer
def make_memory_decision(context: ContextSnapshot, prediction: PredictionResult,
                         memory_state: MemoryState) -> MemoryActionPlan

# Execution Layer
def execute_plan(plan: MemoryActionPlan, memory_state: MemoryState) -> MemoryState

# Simulation Layer
def run_simulation(sessions: List[UserSession], policy: str,
                   ram_mb: int, seed: int) -> BenchmarkResult

def run_ablation(sessions: List[UserSession], experiment: str,
                 ram_mb: int, n_runs: int = 30) -> AblationResult

# Metrics Layer
def compute_kpis(results: List[BenchmarkResult]) -> KPIReport
def compare_policies(results: Dict[str, List[BenchmarkResult]]) -> ComparisonTable
```

### 7.2 Dashboard Endpoints (Streamlit + optional FastAPI)

```
GET  /api/context/current          → ContextSnapshot
GET  /api/prediction/latest        → PredictionResult
GET  /api/memory/state             → MemoryState
GET  /api/memory/actions?limit=50  → List[MemoryAction]
GET  /api/benchmark/results        → BenchmarkResult
GET  /api/benchmark/compare        → ComparisonTable
GET  /api/kpis                     → KPIReport
POST /api/simulation/run           → BenchmarkResult
```

---

## 8. Build Steps — Week by Week

### Week 1: Foundation + Data

| Day | Task | Output |
|---|---|---|
| 1-2 | Project setup: directory structure, git (PRIVATE repo), venv, requirements.txt | Working dev environment |
| 3-4 | Create `app_profiles.json` (15-20 apps) and `personas.json` (5 personas) with transition matrices | Configuration files |
| 5-7 | Build `generator.py`: generate 50,000 synthetic events across all personas with battery/RAM simulation | `data/generated/*.csv` + SQLite |

### Week 2: Prediction Models

| Day | Task | Output |
|---|---|---|
| 1-2 | Build `FeatureEngine`: encode categoricals, compute derived features, train-test split | `feature_engine.py` |
| 3 | Build `MarkovPredictor`: train, evaluate top-1/top-3 accuracy per persona | Baseline accuracy numbers |
| 4-5 | Build `XGBoostPredictor`: train with 5-fold CV, hyperparameter tune, extract feature importance | Trained model + accuracy |
| 6-7 | LSApp integration: download, preprocess, evaluate XGBoost on real data | Real-data accuracy comparison |

### Week 3: Memory Simulator

| Day | Task | Output |
|---|---|---|
| 1-2 | Build `MemoryState`, `AppCacheEntry`, `MetricsCollector`, event replay loop | Core simulation framework |
| 3-4 | Implement 3 baselines: `NoOptPolicy`, `LRUPolicy`, `StaticPreloadPolicy` | Baseline policies |
| 5-6 | Build `CacheScoringEngine`, `PreloadDecisionEngine`, `ThrashingDetector` → `AetherMemPolicy` | Core system |
| 7 | Initial benchmark: 4 policies × 5 personas × 6GB RAM, verify AetherMem wins | First results |

### Week 4: Rigorous Benchmarking

| Day | Task | Output |
|---|---|---|
| 1-2 | Multi-config: 4 policies × 5 personas × 3 RAM configs × 30 runs = 1,800 simulations | Full benchmark data |
| 3-4 | Ablation: 6 configs × 5 personas × 3 RAM × 30 runs = 2,700 simulations | Ablation data |
| 5-7 | Statistical analysis: mean ± std, paired t-tests, p-values, confidence intervals | Statistical validation |

### Week 5: Visualization + Dashboard

| Day | Task | Output |
|---|---|---|
| 1-3 | Publication figures: architecture diagram, benchmark bars, ablation bars, heatmaps, timelines | `visualization/figures/` |
| 4-5 | Streamlit dashboard: live simulation, memory map, benchmark comparison, KPI report | `dashboard/app.py` |

### Week 6-7: Patent Filing

| Day | Task | Output |
|---|---|---|
| 1-3 | Draft patent specification: title, field, background, summary, detailed description, figures | `patent/specification.md` |
| 4-5 | Prior art search: Google Patents, InPASS. Document differences. Refine claims. | `patent/prior_art_search.md` |
| 6-7 | File: Form 1 + Form 2 (provisional) + Form 5 + Form 28 via IPO e-filing | Patent application number |

### Week 8-9: IEEE Paper + Report

| Day | Task | Output |
|---|---|---|
| 1-3 | Write IEEE paper in LaTeX (6 pages): intro, related work, design, experiments, results, conclusion | `paper/main.tex` |
| 4-5 | Capstone report: expand paper into full report with appendices | Report document |
| 6-7 | Faculty review, formatting, submission to IEEE conference | Submitted paper |

---

## 9. Directory Structure

```
aethermem/
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── app_profiles.json
│   ├── personas.json
│   ├── hyperparams.yaml
│   └── simulation_config.yaml
│
├── data/
│   ├── generator.py
│   ├── lsapp_preprocessor.py
│   ├── feature_engine.py
│   ├── generated/
│   └── external/lsapp/
│
├── models/
│   ├── markov.py
│   ├── xgboost_predictor.py
│   ├── evaluate.py
│   ├── trained/
│   └── results/
│
├── core/
│   ├── context.py
│   ├── memory_state.py
│   ├── cache_scorer.py
│   ├── preload_engine.py
│   ├── thrashing_detector.py
│   ├── decision_controller.py
│   └── action_executor.py
│
├── policies/
│   ├── base_policy.py
│   ├── no_opt_policy.py
│   ├── lru_policy.py
│   ├── static_preload_policy.py
│   └── aethermem_policy.py
│
├── simulation/
│   ├── engine.py
│   ├── benchmark.py
│   ├── ablation.py
│   ├── metrics.py
│   └── results/
│
├── visualization/
│   ├── plot_benchmark.py
│   ├── plot_ablation.py
│   ├── plot_prediction.py
│   ├── plot_memory_timeline.py
│   └── figures/
│
├── dashboard/
│   └── app.py
│
├── patent/
│   ├── specification.md
│   ├── claims.md
│   ├── prior_art_search.md
│   └── figures/
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── tests/
│   ├── test_feature_engine.py
│   ├── test_cache_scorer.py
│   ├── test_preload_engine.py
│   ├── test_thrashing_detector.py
│   ├── test_simulation.py
│   └── test_policies.py
│
└── scripts/
    ├── run_all_benchmarks.py
    ├── run_ablation.py
    ├── generate_all_figures.py
    └── export_results.py
```

---

## 10. Verification Plan

### Unit Tests

| Test | Verifies |
|---|---|
| `test_feature_engine` | Feature vector shape, encoding consistency, no NaN values |
| `test_cache_scorer` | Score range [0,1], higher score for predicted apps, pressure penalty works |
| `test_preload_engine` | All 5 gates work independently and in combination |
| `test_thrashing_detector` | Thrashing detected on repeated evict-reload, safe mode activates/deactivates |
| `test_simulation` | Session replays correctly, metrics accumulate, all policies produce results |
| `test_policies` | Each policy produces valid MemoryActionPlan, no memory leaks |

### Benchmark Validation Criteria

| Criterion | Pass Condition |
|---|---|
| AetherMem latency < LRU latency | p < 0.05 (paired t-test) |
| AetherMem cache hit rate > LRU cache hit rate | p < 0.05 |
| AetherMem thrashing < LRU thrashing | p < 0.05 |
| XGBoost accuracy > Markov accuracy | > 10 percentage points |
| Ablation: removing prediction degrades cache hit rate | > 5% drop |
| Ablation: removing preload increases latency | > 5% increase |
| All 1,800+ benchmark runs complete | Zero crashes |
| Results reproducible with same seed | Identical output |

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
streamlit>=1.28.0
plotly>=5.15.0
joblib>=1.3.0
pyyaml>=6.0
pytest>=7.4.0
```
