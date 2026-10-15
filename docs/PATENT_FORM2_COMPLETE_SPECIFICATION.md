# INDIAN PATENT APPLICATION
# FORM 2 — COMPLETE SPECIFICATION
# (See Section 10 of the Patents Act 1970 and Rule 13 of the Patents Rules 2003)

---

## TITLE OF THE INVENTION

**ADAPTIVE MEMORY MANAGEMENT SYSTEM FOR AI-INTENSIVE MOBILE ENVIRONMENTS
USING MACHINE LEARNING-DRIVEN PROACTIVE APP CACHING WITH CALIBRATED
PROBABILITY GATING AND SEQUENTIAL PATTERN INTEGRATION**

---

## APPLICANT(S)

| Field | Details |
|-------|---------|
| Name | [YOUR FULL NAME] |
| Nationality | Indian |
| Address | [YOUR COMPLETE ADDRESS WITH PIN CODE] |
| Category | Natural Person |

---

## INVENTOR(S)

| Field | Details |
|-------|---------|
| Name | [YOUR FULL NAME] |
| Nationality | Indian |
| Address | [YOUR COMPLETE ADDRESS WITH PIN CODE] |

---

## PREAMBLE TO THE DESCRIPTION

The following specification particularly describes the nature of this invention
and the manner in which it is to be performed.

---

## FIELD OF THE INVENTION

The present invention relates to mobile computing systems, and more
specifically to an adaptive memory management system that employs machine
learning to proactively manage application caching in Android-based mobile
devices operating under high artificial intelligence workload conditions.

---

## BACKGROUND OF THE INVENTION / PRIOR ART

### 1. The Problem

Modern Android smartphones run increasingly complex AI workloads —
on-device large language models (LLMs), image segmentation, real-time
translation — that compete aggressively for RAM. Simultaneously, users
context-switch between 5-15 apps per session [Shin et al., 2012 — see
Literature Survey Reference A1]. When a background app is evicted from RAM
and the user opens it, the "cold-start" penalty is 800-2000ms [Apple Developer
Documentation, Reference B4].

Current solutions are **reactive and static**:
- Android's Low Memory Killer Daemon (LMKD) kills background apps only
  *after* RAM pressure exceeds a threshold [Reference B1].
- Samsung's Game Booster uses binary app-category heuristics, not
  per-user behavioral learning [Reference B3].
- Apple's Jetsam daemon saves app state to disk, adding 200-800ms I/O
  latency [Reference B4].
- Prior prediction systems produce raw probability scores with no OS
  integration [References A1-A5].

### 2. Specific Shortcomings of Prior Art

The following 20 peer-reviewed publications collectively establish the
limitations of existing approaches (full citations in the Literature Survey,
`docs/LITERATURE_SURVEY_20_PAPERS.md`):

**App Usage Prediction (References A1-A5):**
- All 5 works (Shin 2012, Xu 2024/MAPLE, Huang 2019, Zhao 2023, Pan 2023)
  produce prediction probabilities that are **never fed back into the OS
  memory manager**. The prediction exists in isolation.
- LLM-based approaches (MAPLE, AppUsage2Vec) require 5-200MB RAM for
  inference — self-defeating when the goal is memory conservation.
- None address probability calibration: XGBoost raw probabilities have
  Expected Calibration Error (ECE) of 0.12 [Niculescu-Mizil 2005, Ref D2].

**Mobile Memory Management (References B1-B4):**
- All existing memory managers are **reactive** — they act after RAM
  pressure, not before it.
- None use ML prediction confidence as a weighted factor in eviction
  priority computation.
- None implement a prediction-confidence-gated preloading mechanism.

**Cache Eviction ML (References C1-C4):**
- Lykouris & Vassilvitskii [C2] prove that well-calibrated ML advice
  yields near-optimal cache hit rates, but provide no practical
  calibration implementation.
- HALP [C4] and related works assume fixed object sizes and are designed
  for web/CDN caches, not mobile app memory (50-800 MB per app).
- None model **negative memory pressure penalty** — large apps should
  have lower cache scores than small apps when RAM is scarce.

**Probability Calibration (References D1-D4):**
- Platt Scaling [D1] and calibration surveys [D2] are studied in the
  context of classification metrics — never as **safety mechanisms** for
  OS-level resource decisions.
- Concept drift detection approaches [D3, D4] do not address the
  **on-device retraining latency constraint** or provide rollback mechanisms.

**Graph Embeddings & Sequence Mining (References E1-E3):**
- Node2Vec [E1] and DeepWalk [E3] are designed for static social graphs —
  not dynamic, user-behavioral app transition graphs.
- PrefixSpan [E2] is an offline batch algorithm — never used as a
  real-time binary feature generator for ML inference.

### 3. Object of the Invention

The present invention addresses all of the above gaps by providing:

1. A **6-Factor Cache Scoring formula** that integrates a calibrated ML
   prediction probability as a first-class OS parameter.
2. A **Preload Gate** that uses Platt-calibrated probabilities as a
   safety threshold before triggering preloading actions.
3. A **Thrashing Detector with Safe Mode** that prevents OOM kills
   proactively using sliding-window cache hit rate monitoring.
4. An **on-device App2Vec embedding engine** (8-dimensional, 1.8 KB)
   based on Node2Vec graph walks over the App Transition Graph.
5. A **PrefixSpan sequential pattern feature extractor** that converts
   offline-mined patterns into real-time binary feature flags.
6. A **Concept Drift Monitor with Versioned Rollback** using KL-Divergence
   and Population Stability Index (PSI).

---

## SUMMARY OF THE INVENTION

The present invention provides an **Adaptive Memory Management (AMM)
System** for Android mobile devices comprising:

**(a)** A machine learning prediction engine based on XGBoost gradient
boosted trees, trained on multi-user app switch event sequences spanning
181,546 events across 15 user profiles over 7-14 days each.

**(b)** A feature extraction pipeline generating a 46-dimensional feature
vector comprising:
- 19 base contextual features (time, battery, RAM, session depth)
- 16 App2Vec graph embedding features (8-dim PCA embeddings for
  current and previous app)
- 11 PrefixSpan sequential pattern features (10 binary flags + 1
  confidence score)

**(c)** A Platt Scaling calibration layer reducing XGBoost's Expected
Calibration Error from 0.12 to 0.007 (94% reduction).

**(d)** A 6-Factor Cache Scoring formula:
```
S(app) = w_R * R + w_F * F + w_P * P + w_I * I + w_C * C - w_M * M
```
where:
- R = Recency score (w_R = 0.25)
- F = Frequency score (w_F = 0.20)
- P = Calibrated ML prediction probability (w_P = 0.30, highest weight)
- I = App importance score (w_I = 0.10)
- C = Contextual relevance score (w_C = 0.10)
- M = Memory footprint penalty (w_M = 0.30, negative coefficient)

**(e)** A 5-condition Preload Gate preventing unnecessary RAM consumption:
1. `confidence >= 0.30` (calibrated threshold)
2. `ram_free >= app_memory * 1.5` (safety margin)
3. `battery >= 0.15` (low battery guard)
4. `network_available = True` (connectivity guard)
5. `thrash_score < 0.70` (system stability guard)

**(f)** A Thrashing Detector monitoring sliding-window cache hit rates,
triggering Safe Mode (suspend preloading, evict only) when
`thrash_score > 0.70`.

**(g)** A Concept Drift Monitor using KL-Divergence (threshold > 0.10)
and PSI (threshold > 0.20) to detect behavioral shifts, with automatic
background retraining and versioned model rollback.

**(h)** An AOSP integration architecture demonstrating hook points in
`ActivityManagerService.java` (Java layer) and `lmkd.cpp` (kernel-space
C daemon), connected via JNI bridge to a native `libamm_inference.so`.

---

## BRIEF DESCRIPTION OF DRAWINGS

**Figure 1:** System architecture showing the three-layer AMM pipeline
(Data Layer → Prediction Layer → Memory Management Layer).

**Figure 2:** 6-Factor Cache Score computation flow with weight
assignments and clipping to [0, 1].

**Figure 3:** Preload Gate 5-condition decision tree.

**Figure 4:** App2Vec training pipeline: App Transition Graph →
Node2Vec walks → Word2Vec → PCA compression (64→8 dim).

**Figure 5:** PrefixSpan offline mining → real-time binary feature
flag injection flow.

**Figure 6:** Concept Drift Monitor: sliding window → KL/PSI
computation → retraining trigger → rollback decision.

**Figure 7:** AOSP hook points in ActivityManagerService.java and lmkd.cpp.

**Figure 8:** Benchmark results: 4 policies × 3 RAM configurations.

---

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

### Embodiment 1: Core Prediction Engine

The AMM system is implemented in Python for the prototype and is
designed for C++/ONNX Runtime deployment in Android AOSP.

**1.1 Data Collection and Feature Engineering**

The system collects the following contextual signals at each app-switch event:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `current_app` | categorical | 28-45 apps | Active application |
| `prev_app_1` | categorical | 28-45 apps | Previous application |
| `hour` | integer | 0-23 | Hour of day |
| `day_of_week` | integer | 0-6 | Day of week |
| `battery` | float | 0.0-1.0 | Battery level |
| `ram_free_mb` | float | 0-12288 | Free RAM in MB |
| `ram_total_mb` | float | 0-12288 | Total RAM in MB |
| `is_charging` | boolean | {0,1} | Charging status |
| `ai_task_active` | boolean | {0,1} | AI workload running |
| `network_type` | integer | {0,1,2} | WiFi/Cellular/None |

Derived features computed on-the-fly:
- `ram_pressure = 1 - (ram_free / ram_total)` — single unified pressure metric
- `hour_sin, hour_cos` — circular encoding of time (prevents hour 23 ≠ hour 0)
- `day_sin, day_cos` — circular encoding of weekday
- `is_weekend` — binary flag for Saturday/Sunday
- `is_peak_hour` — binary flag for 19:00-22:00 (social app peak hours)

**1.2 App2Vec Graph Embedding Engine**

The App2Vec module constructs a directed weighted graph G = (V, E, W) where:
- V = set of unique apps (|V| = 28-45)
- E = app transition edges
- W(u,v) = number of transitions from app u to app v

Node2Vec biased random walks are performed with:
- Walk length: 30 steps
- Walks per node: 200
- Return parameter p: 1.0 (controls backtracking tendency)
- In-out parameter q: 0.5 (DFS-biased, explores app neighborhoods)

Word2Vec skip-gram training on 5,600 walks produces 64-dimensional embeddings.
PCA compression (64→8 dimensions) retains 74-80% variance while reducing
XGBoost overfitting risk. Final embedding matrix: 28×8 = 224 floats ≈ 1.8 KB.

Validation (from experimental results on 181,546-event merged dataset):
- cos(WhatsApp, Telegram) = 0.749 (both messaging apps — correctly HIGH)
- cos(YouTube, Netflix) = 0.683 (both entertainment — correctly HIGH)
- cos(BGMI, Settings) = 0.383 (gaming vs system — correctly LOW)

**1.3 PrefixSpan Sequential Pattern Mining**

The PrefixSpan algorithm [Pei et al., 2001 — Reference E2] is applied
offline on 20,000 sampled events segmented into 617 sessions (30-minute
gap threshold). With minimum support = 3%, 129,021 patterns are found;
the top 10 by session support are retained.

Top discovered patterns (from experimental results):
1. WhatsApp → WhatsApp: 47.2% session support
2. WhatsApp → YouTube: 46.7% session support
3. YouTube → WhatsApp: 46.5% session support
4. Instagram → WhatsApp: 42.3% session support

At runtime, pattern matching is O(K × H) where K=10 patterns and H=4
history depth — computationally negligible.

**1.4 XGBoost Classifier with Platt Scaling**

Model configuration:
- Algorithm: XGBoost gradient-boosted trees
- Estimators: 200 trees
- Max depth: 6
- Objective: multi:softprob (multi-class softmax)
- Input: 46-dimensional feature vector
- Output: Softmax probability distribution over 28-45 apps

Post-hoc calibration via Platt Scaling:
- A Logistic Regression model is fitted over XGBoost output probabilities
  on a held-out calibration set (10% of training data)
- ECE before calibration: ~0.12 (industry benchmark for XGBoost)
- ECE after calibration: 0.007 (94% reduction)
- This makes the "60% confidence" threshold actually correspond to a
  true 60% empirical frequency

**1.5 6-Factor Cache Score**

At each memory management decision point, a cache score S ∈ [0,1] is
computed for each background app:

```
S(app) = clip(
    0.25 × R + 0.20 × F + 0.30 × P + 0.10 × I + 0.10 × C - 0.30 × M,
    0.0, 1.0
)
```

Weight rationale:
- w_P = 0.30 (highest): ML prediction is the novel contribution; it
  receives highest weight reflecting calibrated reliability
- w_R = 0.25: Recency is the next strongest signal (LRU foundation)
- w_M = 0.30 (negative): Memory penalty prevents keeping large apps
  (e.g., BGMI: 400 MB) at the cost of multiple small apps

Apps with S > 0.50 are preserved in RAM when trimApplications() is called.
Apps with S ≤ 0.50 are eligible for standard LRU eviction.

**1.6 Preload Gate**

Before preloading a predicted app into RAM, all 5 conditions must be met:

```
Condition 1: P_calibrated >= 0.30
Condition 2: RAM_free >= app_memory_MB × 1.5
Condition 3: battery >= 0.15
Condition 4: network_type != "none"
Condition 5: thrash_score < 0.70
```

This prevents a naive prediction system from consuming RAM unnecessarily
when the prediction is uncertain (Condition 1), when RAM is too scarce
(Condition 2), when device is low-power (Condition 3), when streaming
resources unavailable (Condition 4), or when system is already thrashing
(Condition 5). No prior work implements all 5 checks simultaneously.

**1.7 Thrashing Detector and Safe Mode**

A sliding window of N=50 recent cache hit/miss events is maintained.
Thrash score is defined as:

```
thrash_score = 1 - (cache_hits / window_size)
```

When `thrash_score > 0.70` (i.e., >30% hit rate degradation):
1. System enters **Safe Mode** — all preloading immediately suspended
2. Eviction-only operation using 6-Factor Score ordering (not LRU)
3. Safe Mode exits when `thrash_score < 0.40` for 3 consecutive windows

**1.8 Concept Drift Monitor**

App usage patterns change over time (students: term-time vs vacation,
professionals: work hours vs evenings). The drift monitor:

1. Maintains sliding windows W_old (days 0-6) and W_new (days 7-13)
2. Computes KL-Divergence: `KL(P_new || P_old)` over app frequency distributions
3. Computes PSI: `Σ (P_new_i - P_old_i) × ln(P_new_i / P_old_i)`
4. Triggers retraining when `KL > 0.10` OR `PSI > 0.20`
5. Retraining runs in background thread without blocking inference
6. Model versioning: if retrained model accuracy < baseline × 0.95, rollback

### Embodiment 2: Multi-User Dataset

The system was trained and validated on a novel multi-user dataset comprising:
- **181,546 total events** from 3 sources:
  - 50,000 synthetic events (5 personas × 10,000 events)
  - 131,179 multi-user simulated events (10 users × 7-14 days)
  - 367 real self-collected events (phone + laptop logging)
- **10 distinct user profiles**: student, professional, gamer, parent,
  developer, social media user, senior, healthcare worker, entrepreneur,
  fitness enthusiast
- **28-45 unique apps** covering 8 categories
- **Temporal train/test split**: 80% train (145,236 events), 20% test
  (36,310 events) — temporal split prevents data leakage

### Embodiment 3: AOSP Integration Architecture

The system is designed for integration into Android Open Source Project:

**Hook Point 1: `ActivityManagerService.java`**
The `trimApplications()` method is modified to call `AMMDecisionEngine.computeCacheScores()` before executing kill decisions. Apps with AMM score > 0.50 bypass the trim operation.

**Hook Point 2: `lmkd.cpp`**
The `find_and_kill_process()` function is modified to sort kill candidates by AMM score (ascending) rather than pure oom_score_adj, ensuring least-likely-needed apps are killed first.

**Native Library: `libamm_inference.so`**
A shared native library (compiled with Android NDK, AArch64) exposes:
- `extract_features()`: 46-dim feature vector construction
- `onnx_infer()`: ONNX Runtime model inference
- `compute_cache_score()`: 6-Factor Score computation
- `preload_gate()`: 5-condition safety check

The C++ prototype (`aosp_integration/cpp_prototype/inference.cpp`) demonstrates
all four functions compiling to a ~12 KB binary with <0.1ms total inference time.

---

## CLAIMS

**Claim 1 (Independent — System):**
An adaptive memory management system for Android mobile devices comprising:
- (a) a machine learning prediction module trained on app-switch event sequences;
- (b) a probability calibration module applying Platt Scaling to reduce Expected Calibration Error below 0.01;
- (c) a 6-Factor Cache Scoring module computing `S = w_R*R + w_F*F + w_P*P + w_I*I + w_C*C - w_M*M` wherein the ML prediction probability P receives the highest positive weight w_P;
- (d) a 5-condition Preload Gate module that conditions preloading on calibrated prediction probability, available RAM headroom, battery level, network availability, and system thrash score simultaneously.

**Claim 2 (Dependent on Claim 1 — App2Vec):**
The system of Claim 1, wherein the machine learning prediction module further comprises an App2Vec graph embedding engine that:
- constructs a directed weighted App Transition Graph from historical app-switch events;
- performs Node2Vec biased random walks with configurable return parameter p and in-out parameter q;
- trains a Word2Vec skip-gram model on generated walk sequences;
- compresses resulting embeddings from 64 to 8 dimensions via Principal Component Analysis;
- produces on-device embeddings of total size not exceeding 2 kilobytes.

**Claim 3 (Dependent on Claim 1 — PrefixSpan):**
The system of Claim 1, wherein the machine learning prediction module further comprises a PrefixSpan sequential pattern mining module that:
- mines frequent app usage subsequences from session-segmented historical data;
- retains the top-K patterns by session support frequency;
- generates binary pattern-match feature flags at inference time in O(K × H) operations, where H is the history window depth.

**Claim 4 (Dependent on Claim 1 — Thrashing):**
The system of Claim 1, further comprising a Thrashing Detector module that:
- maintains a sliding window of cache hit/miss events;
- computes a thrash score as the complement of the cache hit rate within the window;
- triggers a Safe Mode state suspending all preloading operations when thrash score exceeds a first threshold;
- exits Safe Mode when thrash score falls below a second threshold for a consecutive number of windows.

**Claim 5 (Dependent on Claim 1 — Drift):**
The system of Claim 1, further comprising a Concept Drift Monitor that:
- computes KL-Divergence and Population Stability Index between historical and recent app usage distributions;
- triggers background model retraining when either metric exceeds a configurable threshold;
- maintains versioned model snapshots;
- automatically rolls back to a previous model version if the retrained model does not improve held-out accuracy by a specified margin.

**Claim 6 (Independent — Method):**
A method for adaptive memory management in a mobile device comprising:
- (a) collecting app-switch events comprising contextual signals including time of day, battery level, RAM state, and previous applications used;
- (b) constructing a feature vector comprising base contextual features, graph embedding features from an App Transition Graph, and sequential pattern match features;
- (c) predicting the next application to be launched using a gradient-boosted tree classifier;
- (d) calibrating the predicted probability using post-hoc Platt Scaling;
- (e) computing a multi-factor cache score for each background application wherein the calibrated predicted probability is a weighted input;
- (f) conditionally preloading the predicted application into RAM only when a set of safety conditions are simultaneously satisfied.

**Claim 7 (Dependent on Claim 6 — AOSP Hook):**
The method of Claim 6, wherein step (e) is executed by a native shared library loaded into the Android ActivityManagerService process via JNI, and wherein step (f) modifies the kill ordering of the Low Memory Killer Daemon by substituting the multi-factor cache score for the standard oom_score_adj priority.

**Claim 8 (Independent — Dataset):**
A multi-user mobile app usage dataset for training adaptive memory management systems comprising:
- event records from a plurality of user profiles each spanning a plurality of consecutive days;
- each record comprising at minimum: user identity, timestamp, current application, previous applications, battery level, RAM free, RAM total, charging status, network type, and session depth;
- user profiles exhibiting distinct behavioral signatures including circadian usage rhythms, weekday-weekend variation, and concept drift after a defined number of days;
- the dataset comprising no fewer than 100,000 events across no fewer than 5 distinct user profiles.

---

## ABSTRACT

An Adaptive Memory Management (AMM) system for Android mobile devices that
proactively manages application caching using machine learning predictions.
The system comprises: (1) a prediction engine based on XGBoost gradient-boosted
trees trained on 181,546 app-switch events across 15 user profiles; (2) an
App2Vec graph embedding engine producing 8-dimensional on-device embeddings
(1.8 KB) from the App Transition Graph using Node2Vec random walks; (3) a
PrefixSpan sequential pattern miner converting offline-discovered patterns into
real-time binary feature flags; (4) a Platt Scaling calibration module reducing
Expected Calibration Error from 0.12 to 0.007; (5) a novel 6-Factor Cache
Scoring formula integrating calibrated ML probability as the highest-weighted
OS scheduling parameter; (6) a 5-condition Preload Gate preventing RAM overuse
from incorrect predictions; (7) a Thrashing Detector with Safe Mode for proactive
OOM prevention; and (8) a Concept Drift Monitor with versioned model rollback.
The system integrates with AOSP via hooks in ActivityManagerService.java and
lmkd.cpp through a native JNI-bridged shared library. Experimental validation
on a 181,546-event merged dataset achieves >67% top-1 and >85% top-3 prediction
accuracy. The system addresses 7 distinct gaps in 20 prior-art publications
spanning app usage prediction, mobile memory management, cache eviction ML,
probability calibration, and graph embeddings domains.

---

## STATEMENT OF NOVELTY

The present invention is novel over the cited prior art [References A1-A5, B1-B4,
C1-C4, D1-D4, E1-E3] in the following respects:

1. **First integration of calibrated ML probability into OS cache scoring:**
   No prior work (A1-A5, B1-B4) feeds a calibrated prediction probability
   as a weighted coefficient into the OS eviction priority formula.

2. **First application of Platt Scaling as an OS safety mechanism:**
   Prior calibration work (D1, D2) studies ECE as a metric; this invention
   uses calibration as a gating condition preventing hardware resource misuse.

3. **First on-device App2Vec for OS scheduling:**
   Prior graph embedding works (E1, E3) produce 100-300 dimensional
   server-side embeddings; this invention produces 8-dimensional on-device
   embeddings (1.8 KB) suitable for OS-level use.

4. **First use of PrefixSpan patterns as real-time ML features:**
   PrefixSpan (E2) is an offline mining algorithm; this invention converts
   mined patterns into O(K) binary feature flags at runtime.

5. **First 5-condition Preload Gate in mobile memory management:**
   No prior work conditions app preloading on simultaneous satisfaction of
   prediction confidence, RAM headroom, battery, network, and thrash score.

6. **Novel Thrashing Detector with Safe Mode:**
   Existing memory managers (B1-B3) react to OOM after it occurs;
   this invention detects thrashing before OOM and enters a protective state.

7. **Novel Concept Drift Monitor with Versioned Rollback:**
   Drift detection works (D3, D4) do not provide mobile-optimized
   background retraining with automatic safety rollback.

---

*This specification is prepared for filing with the Indian Patent Office (IPO)
under the Patents Act 1970. Supporting evidence: `docs/LITERATURE_SURVEY_20_PAPERS.md`
(20 cited references), experimental results in `results/phase2_metrics.json`,
and C++ feasibility proof in `aosp_integration/cpp_prototype/inference.cpp`.*
