# Image Generation Prompts for Gemini

Copy-paste these into Gemini to generate your diagrams.

---

## Prompt 1: UML Class Diagram

Generate a clean, professional UML class diagram on a white grid background for a "Predictive Memory Management System". Use yellow/gold colored headers on each class box, white body, black text, and proper UML arrows.

Show these classes with attributes and methods:

1. **ContextSnapshot**: attributes: +current_app: string, +prev_apps: List, +hour: int, +battery: float, +ram_free_mb: float, +ai_task_active: bool. Methods: +to_feature_vector(), +to_dict()

2. **FeatureEngine**: attributes: -label_encoders: Dict, -scaler: StandardScaler. Methods: +fit(), +transform()

3. **PredictionResult**: attributes: +predicted_app: string, +confidence: float, +top_k: List, +model_used: string

4. **MarkovPredictor**: attributes: -transition_matrix: Dict. Methods: +predict(), +fit()

5. **XGBoostPredictor**: attributes: -model: XGBClassifier, -feature_engine: FeatureEngine. Methods: +predict(), +predict_proba(), +fit()

6. **CacheScoringEngine**: attributes: -w_recency: 0.25, -w_frequency: 0.20, -w_prediction: 0.25. Methods: +compute_score(), +rank_all()

7. **MemoryState**: attributes: +total_ram_mb: float, +cached_apps: Dict. Methods: +free_ram(), +is_app_cached(), +add_to_cache(), +remove_from_cache()

8. **ThrashingDetector**: attributes: -event_window: deque, -is_safe_mode: bool. Methods: +is_thrashing(), +get_score(), +enter_safe_mode()

9. **MemoryDecisionController**: attributes: -cache_scorer, -preload_engine, -thrashing_detector. Methods: +decide(), +apply_plan()

Show UML relationships: ContextSnapshot feeds into FeatureEngine, FeatureEngine feeds into XGBoostPredictor, both MarkovPredictor and XGBoostPredictor implement a Predictor interface (dashed arrows), both produce PredictionResult, CacheScoringEngine uses PredictionResult and MemoryState, MemoryDecisionController contains CacheScoringEngine and ThrashingDetector. Use proper UML notation with multiplicity. Clean, readable, professional engineering style.

---

## Prompt 2: Full End-to-End HLD Architecture Diagram

Generate a clean, professional end-to-end High Level Design (HLD) architecture diagram for a "Predictive Memory Management System for AI-Heavy Mobile Devices". White background, light grid lines. Black and white with yellow/gold accent on box headers only. Clean engineering style.

Show complete left-to-right data flow through these blocks:

**Block 1 — Context Collection Layer** (left side):
- Triggered by "App Switch Event" 
- Contains: AppHistoryBuffer, SystemMonitor, ContextCollector
- Collects: current_app, prev_apps, hour, battery, ram_free, ai_task_active
- Output arrow labeled "ContextSnapshot"

**Block 2 — Prediction Layer** (center-left):
- Contains: Feature Engineering (LabelEncoder, Derived Features), XGBoost Model (primary), Markov Model (fallback)
- Output arrow labeled "PredictionResult (app, confidence, top_k)"

**Block 3 — Memory Decision Layer** (center):
- Contains: Cache Scoring Engine with formula S = w1*R + w2*F + w3*P + w4*I + w5*C - w6*M
- Preload Decision Gate (5 conditions)
- Thrashing Detector (sliding window)
- Shows two modes: NORMAL and SAFE MODE
- Output arrow labeled "MemoryActionPlan"

**Block 4 — Action Executor** (center-right):
- Three actions: EVICT, PRELOAD, RETAIN
- Connects to Memory State (showing RAM bar with cached apps)
- Feedback arrow back to Context Collection

**Block 5 — Simulation Engine** (bottom, full width):
- Session Replay, 4 Policy Comparison, Ablation Study, Statistical Tests

**Block 6 — Dashboard** (bottom-right):
- Streamlit, KPI Gauges, Benchmark Charts

Show clear labeled arrows between all blocks. Include feedback loop from Memory State back to Context Collection. Professional technical documentation quality, not marketing style.

---

## Prompt 3: Data Flow Diagram (DFD Level 1)

Generate a clean Data Flow Diagram (DFD Level 1) for a "Predictive Memory Management System" on white background. Use standard DFD notation: circles for processes, parallel lines for data stores, rectangles for external entities, arrows for data flows with labels.

External entities (rectangles):
- "User / Mobile Device" (top)
- "Device Memory (RAM)" (right)

Processes (circles numbered):
- 1.0 Context Collection
- 2.0 Feature Engineering and Prediction
- 3.0 Memory Decision Making
- 4.0 Action Execution

Data Stores (parallel lines):
- D1: Event Log (SQLite)
- D2: Model Store (trained XGBoost, Markov)
- D3: App Profiles

Data flows (labeled arrows):
- User to 1.0: app_switch_event, system_state
- 1.0 to 2.0: ContextSnapshot
- 2.0 to 3.0: PredictionResult
- D2 to 2.0: trained_models
- D3 to 3.0: app_profiles
- 3.0 to 4.0: MemoryActionPlan
- 4.0 to Device Memory: evict/preload/retain
- 4.0 to D1: action_log
- Device Memory to 1.0: memory_state (feedback)

Clean, professional, black and white engineering diagram style.

---

## Prompt 4: Sequence Diagram

Generate a clean UML sequence diagram for the "App Switch Event Processing" flow in a mobile memory management system. White background, professional style.

Actors/Objects (left to right): User, ContextCollector, FeatureEngine, XGBoostPredictor, MemoryDecisionController, ActionExecutor

Sequence of messages:
1. User to ContextCollector: app_switch_event
2. ContextCollector self-call: capture() read battery, RAM, app history
3. ContextCollector to FeatureEngine: ContextSnapshot
4. FeatureEngine self-call: transform() encode, normalize, derive features
5. FeatureEngine to XGBoostPredictor: feature_vector
6. XGBoostPredictor self-call: predict_proba() get confidence scores
7. XGBoostPredictor to MemoryDecisionController: PredictionResult
8. MemoryDecisionController self-call: rank_all() score each cached app
9. MemoryDecisionController self-call: check_eviction() if RAM low
10. MemoryDecisionController self-call: should_preload() 5 gate checks
11. MemoryDecisionController self-call: is_thrashing() sliding window
12. MemoryDecisionController to ActionExecutor: MemoryActionPlan
13. ActionExecutor self-call: execute() evict, preload, update memory
14. ActionExecutor to User: app launched (warm or cold start)

Use standard UML sequence diagram notation with lifelines, activation bars, self-calls shown as nested boxes. Clean black and white style.
