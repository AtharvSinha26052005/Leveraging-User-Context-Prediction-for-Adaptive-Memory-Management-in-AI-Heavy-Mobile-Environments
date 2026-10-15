# System Design — Leveraging User Context Prediction for Adaptive Memory Management in AI-Heavy Mobile Environments

## 1. System Architecture Diagram

```
+===========================================================================+
|                          MOBILE DEVICE (4-8 GB RAM)                       |
|                                                                           |
|  +---------------------+    +---------------------+    +--------------+   |
|  | MODULE 1            |    | MODULE 2            |    | MODULE 3     |   |
|  | Context Collection  |--->| Prediction Layer    |--->| Memory       |   |
|  | Layer               |    |                     |    | Decision     |   |
|  |                     |    |                     |    | Layer        |   |
|  | - AppHistoryBuffer  |    | - FeatureEngine     |    |              |   |
|  | - SystemMonitor     |    | - MarkovPredictor   |    | - CacheScorer|   |
|  | - ContextCollector  |    | - XGBoostPredictor  |    | - PreloadGate|   |
|  |                     |    | - ConfidenceCalib.  |    | - ThrashDet. |   |
|  +---------------------+    +---------------------+    +------+-------+   |
|                                                               |           |
|                                                               v           |
|                                                        +--------------+   |
|                                                        | MODULE 4     |   |
|                                                        | Action       |   |
|                                                        | Executor     |   |
|                                                        |              |   |
|                                                        | - Evict      |   |
|                                                        | - Preload    |   |
|                                                        | - Retain     |   |
|                                                        | - Compress   |   |
|                                                        +--------------+   |
|                                                                           |
|  +--------------------------------------------------------------------+  |
|  | MODULE 5: Simulation Engine                                         |  |
|  | - UserSession replay  - MemoryState tracking  - Metrics collection  |  |
|  | - Policy comparison   - Ablation experiments  - Statistical tests   |  |
|  +--------------------------------------------------------------------+  |
|                                                                           |
|  +--------------------------------------------------------------------+  |
|  | MODULE 6: Dashboard (Streamlit)                                     |  |
|  | - Real-time context   - Memory map   - KPI gauges   - Charts       |  |
|  +--------------------------------------------------------------------+  |
+===========================================================================+
```

---

## 2. Data Flow Diagram (DFD)

### 2.1 Level 0 — Context Diagram

```
                +------------------+
                |   User / Device  |
                |   (External)     |
                +--------+---------+
                         |
                   app_switch_event
                   system_state
                         |
                         v
            +------------------------+
            |                        |
            |   Adaptive Memory      |
            |   Management System    |
            |                        |
            +--------+---------------+
                     |
              memory_actions
              (evict / preload / retain)
                     |
                     v
            +------------------+
            | Device Memory    |
            | (RAM / Cache)    |
            +------------------+
```

### 2.2 Level 1 — System Decomposition

```
+------------------+
|   User / Device  |
+--------+---------+
         |
   app_switch_event, battery, ram, network, ai_state
         |
         v
+--------+---------+       ContextSnapshot        +-----------------------+
| 1.0              +----------------------------->| 2.0                   |
| Context          |                              | Prediction            |
| Collection       |                              | Engine                |
+------------------+                              +-----------+-----------+
                                                              |
                                                    PredictionResult
                                                    (app, confidence,
                                                     top_k)
                                                              |
                                                              v
+-----------------------+    MemoryActionPlan     +-----------+-----------+
| 4.0                   |<------------------------| 3.0                   |
| Action                |                         | Memory Decision       |
| Executor              |                         | Controller            |
+-----------+-----------+                         +-----------+-----------+
            |                                                 ^
      updated_memory_state                                    |
            |                                          MemoryState
            v                                          (current cache,
    +-------+--------+                                  free RAM)
    | D1: Memory     |--------------------------------+
    | State Store    |
    +----------------+

    +----------------+
    | D2: Event Log  |  <--- all modules log actions
    | (SQLite)       |
    +----------------+

    +----------------+
    | D3: Model      |  <--- trained XGBoost, Markov matrix
    | Store          |
    +----------------+
```

### 2.3 Level 2 — Prediction Engine Decomposition

```
ContextSnapshot
     |
     v
+----+------------------+
| 2.1 Feature           |
| Engineering            |
|                        |
| - LabelEncoder         |
| - StandardScaler       |
| - Derived features     |
+----+------------------+
     |
     | feature_vector [1 x N]
     |
     +--------------------+
     |                    |
     v                    v
+----+-------+    +------+--------+
| 2.2 Markov |    | 2.3 XGBoost   |
| Predictor  |    | Predictor     |
| (fallback) |    | (primary)     |
|            |    |               |
| P(next|cur)|    | predict_proba |
+----+-------+    +------+--------+
     |                    |
     v                    v
+----+--------------------+-------+
| 2.4 Model Selector              |
|                                  |
| IF XGBoost confidence > 0.3:    |
|     USE XGBoost result          |
| ELSE:                            |
|     USE Markov result            |
+-----------------+----------------+
                  |
                  v
          PredictionResult
```

---

## 3. Class Diagram (UML)

```
+-----------------------------+         +----------------------------+
|      <<dataclass>>          |         |      <<dataclass>>         |
|      ContextSnapshot        |         |      PredictionResult      |
+-----------------------------+         +----------------------------+
| - current_app: str          |         | - predicted_app: str       |
| - prev_apps: List[str]     |         | - confidence: float        |
| - hour: int                |         | - top_k: List[Tuple]       |
| - day_of_week: int         |         | - model_used: str          |
| - battery: float           |         | - timestamp: datetime      |
| - is_charging: bool        |         +----------------------------+
| - ram_free_mb: float       |
| - ram_total_mb: float      |
| - network_type: str        |         +----------------------------+
| - ai_task_active: bool     |         |      <<dataclass>>         |
| - screen_state: str        |         |      AppCacheEntry         |
| - timestamp: datetime      |         +----------------------------+
+-----------------------------+         | - app_name: str            |
| + to_feature_vector(): arr  |         | - memory_mb: float         |
| + to_dict(): dict           |         | - cache_mb: float          |
+-----------------------------+         | - last_accessed: datetime  |
              |                         | - access_count: int        |
              | uses                    | - category: str            |
              v                         | - importance: float        |
+-----------------------------+         | - cold_launch_ms: int      |
|      FeatureEngine          |         | - warm_launch_ms: int      |
+-----------------------------+         | - is_ai_workload: bool     |
| - label_encoders: Dict      |         +----------------------------+
| - scaler: StandardScaler    |         | + time_since_access(): f   |
+-----------------------------+         | + total_memory(): float    |
| + fit(sessions): self       |         +----------------------------+
| + transform(ctx): ndarray   |                    |
| + get_feature_names(): List |                    | stored in
+-----------------------------+                    v
              |                         +----------------------------+
              | feeds                   |      MemoryState           |
              v                         +----------------------------+
+-----------------------------+         | - total_ram_mb: float      |
|    <<interface>>            |         | - cached_apps: Dict        |
|    Predictor                |         | - foreground_app: str      |
+-----------------------------+         | - system_reserved_mb: float|
| + predict(ctx): Result      |         +----------------------------+
+-----------------------------+         | + free_ram(): float        |
       ^              ^                 | + used_ram(): float        |
       |              |                 | + is_app_cached(app): bool |
       |              |                 | + add_to_cache(entry)      |
+------+------+ +-----+------+         | + remove_from_cache(app)   |
| MarkovPred. | | XGBoostPred.|         +----------------------------+
+-------------+ +-------------+                    |
| - trans_mat | | - model     |                    | read/write
| - app_counts| | - feat_eng  |                    v
+-------------+ +-------------+         +----------------------------+
| + predict() | | + predict() |         | MemoryDecisionController   |
| + fit()     | | + fit()     |         +----------------------------+
+-------------+ +-------------+         | - cache_scorer             |
                                        | - preload_engine           |
                                        | - thrashing_detector       |
                                        +----------------------------+
                                        | + decide(ctx, pred, mem)   |
                                        |     -> MemoryActionPlan    |
                                        | + apply_plan(plan, mem)    |
                                        |     -> MemoryState         |
                                        +----------------------------+
                                                   |
                      +----------------------------+---+
                      |                                |
           +----------+----------+        +------------+---------+
           | CacheScoringEngine  |        | PreloadDecisionEngine|
           +---------------------+        +----------------------+
           | - w_recency: 0.25   |        | - conf_threshold: 0.6|
           | - w_frequency: 0.20 |        | - battery_min: 20.0  |
           | - w_prediction: 0.25|        | - ram_headroom: 500  |
           +---------------------+        +----------------------+
           | + compute_score()   |        | + should_preload()   |
           | + rank_all()        |        | + record_outcome()   |
           | + get_evict_cands() |        | + success_rate()     |
           +---------------------+        +----------------------+

           +---------------------+
           | ThrashingDetector   |
           +---------------------+
           | - event_window: deq |
           | - reload_thresh: 3  |
           | - window_sec: 120   |
           | - is_safe_mode: bool|
           +---------------------+
           | + record_event()    |
           | + is_thrashing()    |
           | + get_score(): float|
           | + enter_safe_mode() |
           | + exit_safe_mode()  |
           +---------------------+
```

---

## 4. Sequence Diagram — App Switch Event Processing

```
User          ContextCollector    FeatureEngine    XGBoostPred    DecisionCtrl     ActionExecutor
 |                  |                  |                |               |                |
 |--app_switch----->|                  |                |               |                |
 |                  |                  |                |               |                |
 |                  |--capture()       |                |               |                |
 |                  |  read battery    |                |               |                |
 |                  |  read ram_free   |                |               |                |
 |                  |  read app_hist   |                |               |                |
 |                  |                  |                |               |                |
 |                  |---ContextSnapshot--->             |               |                |
 |                  |                  |--transform()   |               |                |
 |                  |                  |  encode cats   |               |                |
 |                  |                  |  compute derived               |                |
 |                  |                  |  normalize     |               |                |
 |                  |                  |                |               |                |
 |                  |                  |--feature_vec-->|               |                |
 |                  |                  |               |--predict_proba |                |
 |                  |                  |               |  top-K apps    |                |
 |                  |                  |               |  confidence    |                |
 |                  |                  |               |               |                |
 |                  |                  |               |--PredResult-->|                |
 |                  |                  |               |               |--rank_all()    |
 |                  |                  |               |               |  score each app|
 |                  |                  |               |               |                |
 |                  |                  |               |               |--check_evict() |
 |                  |                  |               |               |  if ram < thres|
 |                  |                  |               |               |                |
 |                  |                  |               |               |--should_preload|
 |                  |                  |               |               |  5 gate checks |
 |                  |                  |               |               |                |
 |                  |                  |               |               |--is_thrashing()|
 |                  |                  |               |               |  sliding window|
 |                  |                  |               |               |                |
 |                  |                  |               |               |--ActionPlan--->|
 |                  |                  |               |               |               |--evict()
 |                  |                  |               |               |               |--preload()
 |                  |                  |               |               |               |--update_mem
 |                  |                  |               |               |               |
 |<---------app launched (warm or cold)-------------------------------------------|
 |                  |                  |               |               |                |
```

---

## 5. State Transition Diagram — System Modes

```
                    +------------------+
                    |                  |
       +----------->|   NORMAL MODE   |<-----------+
       |            |                  |            |
       |            +--------+---------+            |
       |                     |                      |
       |          thrashing_score > 0.7             |
       |                     |                      |
       |                     v                      |
       |            +--------+---------+            |
       |            |                  |            |
       |            |   SAFE MODE     |            |
       |            |                  |            |
       |            | - preload OFF   |            |
       |            | - eviction      |            |
       |            |   threshold     |            |
       |            |   raised (0.15) |            |
       |            | - target free   |            |
       |            |   RAM x 1.5     |            |
       |            |                  |            |
       |            +--------+---------+            |
       |                     |                      |
       |          thrashing_score < 0.3             |
       |          for 60 continuous seconds          |
       |                     |                      |
       +---------------------+                      |
                                                    |
       Note: Safe Mode can also exit on             |
       explicit reset() call  ------->--------------+
```

---

## 6. Activity Diagram — Cache Scoring Process

```
                        [START]
                           |
                           v
                 +-------------------+
                 | For each cached   |
                 | app 'a' in cache  |
                 +-------------------+
                           |
                           v
                +--------------------+
                | Compute R(a,t)     |
                | recency =          |
                | 1/(1 + dt/300)     |
                +--------------------+
                           |
                           v
                +--------------------+
                | Compute F(a,t)     |
                | frequency =        |
                | count(a)/max_count |
                +--------------------+
                           |
                           v
                +--------------------+
                | Compute P(a,t)     |
                | pred_reuse =       |
                | ML model prob for a|   <--- KEY INNOVATION
                +--------------------+
                           |
                           v
                +--------------------+
                | Compute I(a)       |
                | importance =       |
                | category_lookup(a) |
                +--------------------+
                           |
                           v
                +--------------------+
                | Compute C(a,t)     |
                | context_relevance  |
                +--------------------+
                           |
                           v
                +--------------------+
                | Compute M(t)       |
                | pressure_penalty   |
                +--------------------+
                           |
                           v
          +----------------------------------+
          | S(a,t) = 0.25*R + 0.20*F         |
          |        + 0.25*P + 0.15*I         |
          |        + 0.15*C - 0.30*M         |
          +----------------------------------+
                           |
                           v
                 +-------------------+
                 | Sort all apps by  |
                 | score ascending   |
                 +-------------------+
                           |
                           v
               +---------------------+
               |  ram_free <          |
               |  PRESSURE_THRESHOLD? |
               +-----+----------+----+
                     |          |
                   YES          NO
                     |          |
                     v          v
          +----------+--+   +--+----------+
          | Evict lowest |   | No eviction |
          | scored apps  |   | needed      |
          | until RAM OK |   +-------------+
          +--------------+
                     |
                     v
                  [END]
```

---

## 7. Deployment Diagram

```
+========================================================+
|                    Development Machine                   |
|                                                         |
|  +-----------+  +-----------+  +--------------------+  |
|  | Python    |  | SQLite    |  | Streamlit          |  |
|  | 3.10+     |  | Database  |  | Dashboard          |  |
|  |           |  |           |  | (localhost:8501)    |  |
|  | - XGBoost |  | - events  |  |                    |  |
|  | - sklearn |  | - actions |  | - Real-time viz    |  |
|  | - pandas  |  | - results |  | - Benchmark plots  |  |
|  | - scipy   |  |           |  | - KPI gauges       |  |
|  +-----------+  +-----------+  +--------------------+  |
|                                                         |
|  +--------------------------------------------------+  |
|  | Simulation Engine                                 |  |
|  |                                                   |  |
|  | - Replays synthetic user sessions                 |  |
|  | - Applies 4 memory policies                       |  |
|  | - Runs 1800+ benchmark simulations                |  |
|  | - Runs 2700+ ablation experiments                 |  |
|  | - Generates statistical reports                   |  |
|  +--------------------------------------------------+  |
|                                                         |
|  +--------------------------------------------------+  |
|  | Output Artifacts                                  |  |
|  |                                                   |  |
|  | - Trained models (XGBoost .json, Markov .pkl)     |  |
|  | - Benchmark results (CSV + SQLite)                |  |
|  | - Publication figures (PNG, PDF)                   |  |
|  | - Patent specification document                   |  |
|  | - IEEE paper (LaTeX)                              |  |
|  +--------------------------------------------------+  |
+========================================================+
```

---

## 8. ER Diagram — Database Schema

```
+------------------+       +------------------+       +-------------------+
| app_profiles     |       | sessions         |       | app_events        |
+------------------+       +------------------+       +-------------------+
| *app_name (PK)   |       | *session_id (PK) |       | *event_id (PK)    |
| category         |       | persona          |   +-->| session_id (FK)   |
| cold_launch_ms   |       | start_time       |   |   | timestamp         |
| warm_launch_ms   |       | end_time         |   |   | app_name          |
| memory_mb        |       | event_count      |   |   | prev_app          |
| cache_mb         |       | ram_config_mb    |   |   | battery           |
| importance       |       +--------+---------+   |   | is_charging       |
| is_ai_workload   |                |              |   | ram_free_mb       |
| ai_cache_mb      |                +--------------+   | network_type      |
+------------------+                                   | ai_task_active    |
                                                       | hour_of_day       |
                                                       | day_of_week       |
                                                       +--------+----------+
                                                                |
                                    +---------------------------+
                                    |                           |
                                    v                           v
                          +---------+--------+       +----------+---------+
                          | predictions      |       | memory_actions     |
                          +------------------+       +--------------------+
                          | *prediction_id   |       | *action_id (PK)    |
                          | event_id (FK)    |       | event_id (FK)      |
                          | model_used       |       | action_type        |
                          | predicted_app    |       | target_app         |
                          | confidence       |       | reason             |
                          | actual_app       |       | ram_before_mb      |
                          | was_correct      |       | ram_after_mb       |
                          | top_3_apps       |       | policy             |
                          | timestamp        |       | mode               |
                          +------------------+       | timestamp          |
                                                     +--------------------+

                          +------------------+       +--------------------+
                          | benchmark_runs   |       | ablation_results   |
                          +------------------+       +--------------------+
                          | *run_id (PK)     |       | *ablation_id (PK)  |
                          | session_id (FK)  |       | experiment_name    |
                          | policy           |       | run_id (FK)        |
                          | ram_config_mb    |       | features_removed   |
                          | avg_launch_lat.  |       | accuracy_top1      |
                          | cache_hit_rate   |       | accuracy_top3      |
                          | cache_miss_count |       | avg_launch_latency |
                          | thrashing_events |       | cache_hit_rate     |
                          | eviction_count   |       | thrashing_events   |
                          | preload_count    |       | timestamp          |
                          | preload_success  |       +--------------------+
                          | memory_util.     |
                          | total_events     |
                          | run_timestamp    |
                          +------------------+
```
