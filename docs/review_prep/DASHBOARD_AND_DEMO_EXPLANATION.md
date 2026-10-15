# Dashboard and Live Demo Explanation Guide

This document explains the interactive components used to demonstrate the project to the review panel: The CLI Live Demo and the Streamlit Web Dashboard.

## 1. The CLI Live Demo
**Execution:** Run `python scripts/live_demo.py` in the terminal.
**Purpose:** A rapid, highly technical, 15-second pipeline execution that proves the architecture works end-to-end without UI overhead. It is the best way to open the presentation.

### What it shows (8 Steps):
1. **Data Overview:** Proves ingestion of 50,000 synthetic events across 5 distinct user personas (power user, student, etc.) plus the presence of real self-collected phone/laptop data.
2. **ML Prediction Engine:** Shows the XGBoost model predicting app switches in real-time. It prints a live terminal table showing `Current App → Predicted App` with top-1 and top-3 accuracy metrics (typically ~67%).
3. **Confidence Calibration (Platt Scaling):** Proves the model knows when it is uncertain. It displays the Expected Calibration Error (ECE) dropping from 12% (0.12) down to a mathematically safe 0.7% (0.007).
4. **4-Policy Simulation:** Runs a real-time memory simulation on a constrained 4GB RAM device, simultaneously testing NoOpt (FIFO), LRU (Android default), Static Preload, and the Proposed policy.
5. **Results:** Prints a visual comparison block proving the Proposed policy beats Android's LRU (e.g., +6.7% cache hit rate), directly translating to thousands of saved evictions.
6. **Full Benchmarks:** Outputs the pre-calculated average hit rates across all 120 simulated multi-persona runs.
7. **Online Learning (Concept Drift):** Proves the system adapts to changing human behavior over time by showing KL-Divergence detecting "drifts" and automatically triggering model retraining without catastrophic rollback.
8. **Architecture Summary:** Prints the mathematical formula for the 6-Factor Cache Score to anchor the presentation in hard OS engineering.

---

## 2. The Streamlit Web Dashboard
**Execution:** Run `python -m streamlit run dashboard/app.py` in the terminal and open `http://localhost:8501`.
**Purpose:** An interactive, highly visual exploration of the project's data, metrics, and ablation studies, allowing the panel to "look under the hood."

### Dashboard Pages (Sidebar Navigation):

#### 1. 📊 Overview
* **What it is:** The executive summary page.
* **Key Features:** Displays massive top-line metrics (Hit Rate, Avg Latency, Thrashing Events). Features a visual flow chart of the System Architecture and the 6-Factor Cache Scoring formula ($S = w_R R + \dots$). It also displays the total project scale (19+ source files, 120 benchmarks, 50 unit tests).

#### 2. 📈 Model Comparison
* **What it is:** Proves why XGBoost was chosen over other algorithms.
* **Key Features:** Interactive bar charts and a Multi-Axis Radar Chart comparing Markov Chains, Random Forests, XGBoost, and Neural Networks across Accuracy (Top-1/Top-3), F1-Score, Training Speed, and Inference Speed. A callout box specifically justifies XGBoost's 0.05ms inference time and ~200KB footprint as optimal for mobile OS deployment.

#### 3. ⚔️ Benchmark Results
* **What it is:** Deep dive into the 120 simulation runs.
* **Key Features:** Shows how different policies perform across varying RAM constraints (4GB, 6GB, 8GB) and different user personas (e.g., Power User vs. Casual). Proves that the ML approach shines brightest on constrained devices (4GB).

#### 4. 🔬 Ablation Study
* **What it is:** Mathematical proof of *why* the model works.
* **Key Features:** A Waterfall chart showing what happens to the Cache Hit Rate when specific components are removed. Removing the "Frequency" factor causes the largest drop, while removing "Prediction" causes the second largest. This proves to the panel that every factor in the 6-Factor equation is strictly necessary.

#### 5. 📉 Statistical Tests
* **What it is:** Academic proof that results are not due to random chance.
* **Key Features:** Displays Paired t-tests and Cohen's d effect sizes. Charts show that the p-values comparing the Proposed Policy to LRU are $< 0.001$, proving highly significant, real-world impact (Cohen's d $> 1.7$).

#### 6. 🌍 Data Validation — Cross-Referenced with Literature
* **What it is:** Justifies the use of the 50,000-event synthetic dataset.
* **Key Features:** Side-by-side bar charts comparing our synthetic data distributions (e.g., Avg Screen Time, % Social Media usage) against 6 published Kaggle datasets (32,000 real users). Proves our data mirrors published academic literature within 1 standard deviation.

#### 7. 📱 Self-Collected Data
* **What it is:** Real-world hardware validation.
* **Key Features:** Displays telemetry collected directly from a Realme 13 5G via ADB and a Windows laptop via `ctypes`. Shows interactive heatmaps of app transitions and line graphs of real battery/RAM drain over time, proving the data pipeline works on physical silicon.

#### 8. 🎮 Live Simulation
* **What it is:** A real-time, interactive memory monitor.
* **Key Features:** Allows the panel to select a Persona (e.g., Developer) and RAM size (e.g., 6GB), then click "Run." It plots two live line charts showing Free RAM (MB) and Cache Hit Rate evolving event-by-event as the 4 policies compete in real-time.

#### 9. ⚙️ Weight Tuner
* **What it is:** An interactive sandbox for the 6-Factor equation.
* **Key Features:** Sliders allowing the judges to manually change the weights for Recency, Frequency, Prediction, etc., and instantly see how their custom weights perform against the default OS baseline.

#### 10. 📋 Dataset Explorer
* **What it is:** Raw data transparency.
* **Key Features:** Allows filtering the 50,000-event dataset by persona or app, showing hour-by-hour usage line charts and a table of the 20 fundamental App Profiles defining cold-start latencies and memory footprints.
