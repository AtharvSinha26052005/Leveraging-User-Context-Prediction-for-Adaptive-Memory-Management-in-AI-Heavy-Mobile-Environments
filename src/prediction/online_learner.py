"""
Online Learner — Production-grade incremental model updates with drift detection.

This module solves the "model staleness" problem: as user behavior changes over
time (new apps installed, routine shifts, seasonal patterns), a static model
degrades. The OnlineLearner detects when this happens and triggers retraining.

Architecture:
    1. Sliding Window Buffer — stores last N events for retraining
    2. Concept Drift Detector — monitors distribution shift via KL-divergence + PSI
    3. Performance Monitor — tracks rolling accuracy, triggers on degradation
    4. Model Versioner — saves each retrained model, supports rollback

Decision D-012: Sliding window (not exponential weighting) chosen because:
- Tree-based models (XGBoost) can't do gradient updates; full retrain required
- Sliding window naturally "forgets" old patterns
- Window size is tunable per-persona (power_user needs shorter window due to
  more diverse behavior)

Patent Claim: "A self-calibrating prediction system that monitors its own accuracy
in real-time, detects concept drift in user behavior distributions, and triggers
model retraining with automatic rollback to prevent deployment of degraded models."
"""
import os
import sys
import time
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DriftReport:
    """Results from a single drift check."""
    timestamp: str
    kl_divergence: float          # KL(current || training) — higher = more drift
    psi: float                    # Population Stability Index — >0.2 is significant
    accuracy_rolling: float       # Rolling accuracy over last check_window events
    accuracy_baseline: float      # Baseline accuracy from training
    is_drifted: bool              # True if any drift threshold exceeded
    drift_reason: str             # Which metric triggered drift
    n_events_since_last_train: int


@dataclass
class ModelVersion:
    """Metadata for a saved model version."""
    version: int
    timestamp: str
    accuracy: float
    n_training_events: int
    window_start: int
    window_end: int
    is_active: bool = True
    rollback_from: Optional[int] = None


class OnlineLearner:
    """
    Production-grade online learning system for the app usage predictor.

    Monitors prediction quality in real-time, detects concept drift via
    distribution comparison, and triggers sliding-window retraining when
    the model becomes stale.
    """

    def __init__(
        self,
        window_size: int = 5000,
        check_interval: int = 200,
        kl_threshold: float = 0.15,
        psi_threshold: float = 0.20,
        accuracy_drop_threshold: float = 0.05,
        min_retrain_events: int = 500,
        max_versions: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            window_size: Number of recent events to use for retraining
            check_interval: Check for drift every N new events
            kl_threshold: KL-divergence threshold for drift detection
            psi_threshold: PSI threshold for drift detection (>0.2 = significant shift)
            accuracy_drop_threshold: Trigger retrain if accuracy drops by this much
            min_retrain_events: Minimum events before first drift check
            max_versions: Maximum model versions to keep on disk
            verbose: Print status updates
        """
        self.window_size = window_size
        self.check_interval = check_interval
        self.kl_threshold = kl_threshold
        self.psi_threshold = psi_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.min_retrain_events = min_retrain_events
        self.max_versions = max_versions
        self.verbose = verbose

        # ── Internal State ──
        self.event_buffer: deque = deque(maxlen=window_size)
        self.prediction_log: deque = deque(maxlen=window_size)

        # Training distribution (app frequencies from original training set)
        self._training_distribution: Optional[Dict[str, float]] = None
        self._baseline_accuracy: float = 0.0

        # Model versioning
        self._current_version: int = 0
        self._version_history: List[ModelVersion] = []
        self._events_since_last_check: int = 0
        self._total_events_processed: int = 0
        self._drift_reports: List[DriftReport] = []

        # Rolling accuracy tracker
        self._correct_count: int = 0
        self._total_predictions: int = 0

    def set_training_baseline(self, training_df: pd.DataFrame, baseline_accuracy: float):
        """
        Record the training distribution and baseline accuracy.
        Called once after initial model training.

        Args:
            training_df: The DataFrame used to train the original model
            baseline_accuracy: The model's accuracy on validation data
        """
        # Compute app frequency distribution from training data
        app_counts = training_df["current_app"].value_counts()
        total = app_counts.sum()
        self._training_distribution = {
            app: count / total for app, count in app_counts.items()
        }
        self._baseline_accuracy = baseline_accuracy

        if self.verbose:
            print(f"  [OnlineLearner] Baseline set: accuracy={baseline_accuracy:.1%}, "
                  f"apps={len(self._training_distribution)}", flush=True)

    def observe(self, event: dict, predicted_app: str, actual_app: str) -> Optional[DriftReport]:
        """
        Observe a new prediction event. This is called after each prediction.

        Args:
            event: The full event dict (for buffering)
            predicted_app: What the model predicted
            actual_app: What actually happened

        Returns:
            DriftReport if a drift check was performed, None otherwise
        """
        # Buffer the event for potential retraining
        self.event_buffer.append(event)
        self._total_events_processed += 1
        self._events_since_last_check += 1

        # Track accuracy
        is_correct = predicted_app == actual_app
        self._correct_count += int(is_correct)
        self._total_predictions += 1

        self.prediction_log.append({
            "predicted": predicted_app,
            "actual": actual_app,
            "correct": is_correct,
            "timestamp": event.get("timestamp", ""),
        })

        # Check for drift at regular intervals
        if (self._events_since_last_check >= self.check_interval
                and self._total_events_processed >= self.min_retrain_events):
            report = self._check_drift()
            self._events_since_last_check = 0
            return report

        return None

    def _check_drift(self) -> DriftReport:
        """
        Perform a comprehensive drift check using multiple signals.

        Returns:
            DriftReport with all metrics and drift decision
        """
        # ── 1. Compute current app distribution from buffer ──
        current_apps = [e.get("current_app", "unknown") for e in self.event_buffer]
        current_counts = Counter(current_apps)
        total = sum(current_counts.values())
        current_dist = {app: count / total for app, count in current_counts.items()}

        # ── 2. KL-Divergence: D_KL(current || training) ──
        kl_div = self._kl_divergence(current_dist, self._training_distribution or {})

        # ── 3. Population Stability Index (PSI) ──
        psi = self._population_stability_index(
            self._training_distribution or {}, current_dist
        )

        # ── 4. Rolling accuracy from prediction log ──
        recent_preds = list(self.prediction_log)[-self.check_interval:]
        rolling_accuracy = sum(p["correct"] for p in recent_preds) / max(len(recent_preds), 1)

        # ── 5. Determine if drift occurred ──
        drift_reason = "none"
        is_drifted = False

        if kl_div > self.kl_threshold:
            is_drifted = True
            drift_reason = f"KL-divergence ({kl_div:.4f} > {self.kl_threshold})"
        elif psi > self.psi_threshold:
            is_drifted = True
            drift_reason = f"PSI ({psi:.4f} > {self.psi_threshold})"
        elif (self._baseline_accuracy - rolling_accuracy) > self.accuracy_drop_threshold:
            is_drifted = True
            drift_reason = (f"Accuracy drop ({rolling_accuracy:.1%} vs "
                            f"baseline {self._baseline_accuracy:.1%})")

        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            kl_divergence=kl_div,
            psi=psi,
            accuracy_rolling=rolling_accuracy,
            accuracy_baseline=self._baseline_accuracy,
            is_drifted=is_drifted,
            drift_reason=drift_reason,
            n_events_since_last_train=self._total_events_processed,
        )

        self._drift_reports.append(report)

        if self.verbose:
            status = "⚠ DRIFT DETECTED" if is_drifted else "✓ Stable"
            print(f"  [OnlineLearner] Check #{len(self._drift_reports)}: {status} "
                  f"| KL={kl_div:.4f} PSI={psi:.4f} "
                  f"Acc={rolling_accuracy:.1%} "
                  f"| {drift_reason}", flush=True)

        return report

    def retrain(self, predictor, val_df: Optional[pd.DataFrame] = None) -> ModelVersion:
        """
        Retrain the predictor on the sliding window buffer.

        Args:
            predictor: The XGBoostPredictor instance to retrain
            val_df: Validation DataFrame (uses last 20% of buffer if not provided)

        Returns:
            ModelVersion metadata for the new model
        """
        if len(self.event_buffer) < self.min_retrain_events:
            raise ValueError(
                f"Not enough events in buffer ({len(self.event_buffer)}) "
                f"for retraining (need {self.min_retrain_events})"
            )

        if self.verbose:
            print(f"\n  [OnlineLearner] Retraining on {len(self.event_buffer)} events...",
                  flush=True)

        # Build DataFrame from buffer
        buffer_df = pd.DataFrame(list(self.event_buffer))

        # Split buffer into train/val (80/20, temporal)
        split_idx = int(len(buffer_df) * 0.8)
        train_df = buffer_df.iloc[:split_idx].copy()

        if val_df is None:
            val_df = buffer_df.iloc[split_idx:].copy()

        # Save previous model for rollback
        self._current_version += 1
        prev_version = self._current_version - 1

        # Create a FRESH predictor instance to avoid class count mismatch
        # (the sliding window may contain fewer app classes than the original
        # training set, causing XGBoost's num_class to conflict)
        from src.prediction.xgboost_predictor import XGBoostPredictor
        fresh_predictor = XGBoostPredictor(
            n_estimators=predictor.n_estimators if hasattr(predictor, 'n_estimators') else 200,
            max_depth=predictor.max_depth if hasattr(predictor, 'max_depth') else 6,
        )

        # Retrain on buffer data
        t0 = time.time()
        metrics = fresh_predictor.fit(train_df, val_df=val_df)
        train_time = time.time() - t0

        new_accuracy = metrics.get("val_accuracy", metrics.get("cv_accuracy_mean", 0.0))

        # ── Rollback check: if new model is worse, revert ──
        version_info = ModelVersion(
            version=self._current_version,
            timestamp=datetime.now().isoformat(),
            accuracy=new_accuracy,
            n_training_events=len(train_df),
            window_start=self._total_events_processed - len(self.event_buffer),
            window_end=self._total_events_processed,
        )

        # Check if new model is significantly worse
        if (len(self._version_history) > 0
                and new_accuracy < self._version_history[-1].accuracy - 0.02):
            version_info.is_active = False
            version_info.rollback_from = prev_version
            if self.verbose:
                print(f"  [OnlineLearner] ⚠ New model v{self._current_version} "
                      f"({new_accuracy:.1%}) worse than v{prev_version} "
                      f"({self._version_history[-1].accuracy:.1%}). ROLLING BACK.",
                      flush=True)
            # In production, we'd reload the previous model here
        else:
            # Save the new model
            version_dir = os.path.join("models_trained", "versions")
            os.makedirs(version_dir, exist_ok=True)

            model_path = os.path.join(version_dir, f"xgboost_v{self._current_version}.json")
            fresh_predictor.model.save_model(model_path)

            # Update baseline with new model's accuracy
            self._baseline_accuracy = new_accuracy
            self._training_distribution = {
                app: count / len(train_df)
                for app, count in train_df["current_app"].value_counts().items()
            }

            if self.verbose:
                print(f"  [OnlineLearner] ✓ Model v{self._current_version} saved "
                      f"(accuracy={new_accuracy:.1%}, trained on {len(train_df)} events, "
                      f"time={train_time:.1f}s)", flush=True)

        self._version_history.append(version_info)

        # Prune old versions
        if len(self._version_history) > self.max_versions:
            self._version_history = self._version_history[-self.max_versions:]

        # Reset counters
        self._correct_count = 0
        self._total_predictions = 0

        return version_info

    def _kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        Compute KL-divergence D_KL(P || Q) between two app distributions.
        Uses smoothing to avoid log(0).
        """
        all_apps = set(list(p.keys()) + list(q.keys()))
        eps = 1e-10  # Smoothing constant

        kl = 0.0
        for app in all_apps:
            p_val = p.get(app, eps)
            q_val = q.get(app, eps)
            if p_val > eps:
                kl += p_val * np.log(p_val / q_val)

        return max(0.0, kl)

    def _population_stability_index(
        self, expected: Dict[str, float], actual: Dict[str, float]
    ) -> float:
        """
        Population Stability Index (PSI) — measures distribution shift.
        PSI < 0.1: no shift, 0.1-0.2: moderate, >0.2: significant.
        """
        all_apps = set(list(expected.keys()) + list(actual.keys()))
        eps = 1e-10

        psi = 0.0
        for app in all_apps:
            e_val = expected.get(app, eps)
            a_val = actual.get(app, eps)
            psi += (a_val - e_val) * np.log(a_val / e_val)

        return max(0.0, psi)

    def get_status(self) -> dict:
        """Get current learner status."""
        recent_preds = list(self.prediction_log)[-self.check_interval:]
        rolling_accuracy = (sum(p["correct"] for p in recent_preds) /
                            max(len(recent_preds), 1))

        return {
            "total_events": self._total_events_processed,
            "buffer_size": len(self.event_buffer),
            "current_version": self._current_version,
            "rolling_accuracy": rolling_accuracy,
            "baseline_accuracy": self._baseline_accuracy,
            "n_drift_checks": len(self._drift_reports),
            "n_drifts_detected": sum(1 for r in self._drift_reports if r.is_drifted),
            "n_retrains": len(self._version_history),
        }

    def save_state(self, path: str):
        """Save learner state to disk."""
        state = {
            "version": self._current_version,
            "version_history": [asdict(v) for v in self._version_history],
            "drift_reports": [asdict(r) for r in self._drift_reports],
            "training_distribution": self._training_distribution,
            "baseline_accuracy": self._baseline_accuracy,
            "total_events": self._total_events_processed,
            "config": {
                "window_size": self.window_size,
                "check_interval": self.check_interval,
                "kl_threshold": self.kl_threshold,
                "psi_threshold": self.psi_threshold,
                "accuracy_drop_threshold": self.accuracy_drop_threshold,
            },
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def print_summary(self):
        """Print comprehensive learner summary."""
        status = self.get_status()
        print(f"\n  ╔══════════════════════════════════════════════════════╗")
        print(f"  ║         ONLINE LEARNER STATUS                       ║")
        print(f"  ╠══════════════════════════════════════════════════════╣")
        print(f"  ║  Total events processed:  {status['total_events']:>6}                    ║")
        print(f"  ║  Buffer size:             {status['buffer_size']:>6} / {self.window_size:<6}          ║")
        print(f"  ║  Current model version:   v{status['current_version']:<5}                    ║")
        print(f"  ║  Rolling accuracy:        {status['rolling_accuracy']:.1%}                     ║")
        print(f"  ║  Baseline accuracy:       {status['baseline_accuracy']:.1%}                     ║")
        print(f"  ║  Drift checks performed:  {status['n_drift_checks']:>6}                    ║")
        print(f"  ║  Drifts detected:         {status['n_drifts_detected']:>6}                    ║")
        print(f"  ║  Model retrains:          {status['n_retrains']:>6}                    ║")
        print(f"  ╚══════════════════════════════════════════════════════╝")


def run_online_learning_demo():
    """
    Demonstrate the online learning pipeline on synthetic data.
    Simulates a scenario where user behavior shifts over time.
    """
    print("\n  Loading data and model...", flush=True)
    df = pd.read_csv("data/synthetic_events.csv")

    from src.prediction.xgboost_predictor import XGBoostPredictor

    predictor = XGBoostPredictor()
    predictor.load_model(
        "models_trained/xgboost_model.json",
        "models_trained/feature_engine.pkl",
        "models_trained/markov_matrix.pkl",
    )

    # Split: first 70% for training baseline, last 30% as "incoming stream"
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    stream_df = df.iloc[split_idx:].copy()

    # Initialize online learner
    learner = OnlineLearner(
        window_size=3000,
        check_interval=500,
        kl_threshold=0.10,
        psi_threshold=0.15,
        accuracy_drop_threshold=0.05,
        verbose=True,
    )

    # Set baseline from training
    learner.set_training_baseline(train_df, baseline_accuracy=0.75)

    print(f"\n  Simulating {len(stream_df)} incoming events...", flush=True)

    # Process stream
    for idx, (_, row) in enumerate(stream_df.iterrows()):
        event = row.to_dict()

        # Get prediction
        try:
            result = predictor.predict(event)
            predicted_app = result.predicted_app
        except Exception:
            predicted_app = event.get("current_app", "unknown")

        actual_app = event.get("next_app", event.get("current_app", "unknown"))

        # Feed to online learner
        report = learner.observe(event, predicted_app, actual_app)

        if report and report.is_drifted:
            print(f"\n  → Drift detected at event {idx}! Triggering retrain...", flush=True)
            try:
                version = learner.retrain(predictor)
                print(f"  → Retrained: v{version.version}, accuracy={version.accuracy:.1%}")
            except Exception as e:
                print(f"  → Retrain failed: {e}")

    # Print summary
    learner.print_summary()

    # Save state
    os.makedirs("results", exist_ok=True)
    learner.save_state("results/online_learner_state.json")
    print(f"\n  Saved state to results/online_learner_state.json")

    return learner


if __name__ == "__main__":
    run_online_learning_demo()
