"""
Confidence Calibrator — Calibrates raw XGBoost probabilities using Platt Scaling.

Raw XGBoost outputs are not true probabilities. A model saying "78% confident"
doesn't mean it's correct 78% of the time at that confidence level. This module
applies post-hoc calibration so that predicted probabilities match observed
frequencies, which is critical for the preload gate's confidence threshold.

Decision D-011: Platt scaling chosen over isotonic regression because:
- Works well with small validation sets (we have ~7500 test events)
- Monotonic transformation preserves ranking
- Single sigmoid fit = no overfitting risk

References:
- Platt, J. (1999). "Probabilistic outputs for SVMs"
- Niculescu-Mizil & Caruana (2005). "Predicting good probabilities with supervised learning"
"""
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


@dataclass
class CalibrationMetrics:
    """Stores calibration quality metrics."""
    method: str
    ece_before: float      # Expected Calibration Error before calibration
    ece_after: float       # Expected Calibration Error after calibration
    mce_before: float      # Maximum Calibration Error before
    mce_after: float       # Maximum Calibration Error after
    brier_before: float    # Brier score before
    brier_after: float     # Brier score after
    n_bins: int = 10
    reliability_data: Dict = field(default_factory=dict)

    def improvement(self) -> float:
        """Percentage improvement in ECE."""
        if self.ece_before == 0:
            return 0.0
        return (self.ece_before - self.ece_after) / self.ece_before * 100


class ConfidenceCalibrator:
    """
    Post-hoc confidence calibration for the XGBoost predictor.

    Wraps the raw model output with a calibration layer that maps
    raw probabilities to calibrated probabilities using Platt scaling
    (logistic regression on logits) or isotonic regression.

    Patent Relevance: The preload gate uses a confidence threshold (θ = 0.6).
    Calibrated confidence means this threshold has real meaning — "60% confident"
    actually corresponds to ~60% accuracy at that confidence level.
    """

    def __init__(self, method: str = "platt"):
        """
        Args:
            method: 'platt' (logistic sigmoid) or 'isotonic' (non-parametric)
        """
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Method must be 'platt' or 'isotonic', got '{method}'")

        self.method = method
        self.is_fitted = False

        # Per-class calibrators (one per app class)
        self._calibrators: Dict[int, object] = {}

        # Global calibrator (maps max_prob → calibrated_prob)
        self._global_calibrator = None

        # Stored metrics from last fit
        self.metrics: Optional[CalibrationMetrics] = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationMetrics:
        """
        Fit the calibrator on validation data.

        Args:
            y_true: True labels (integer class indices), shape (n_samples,)
            y_prob: Raw predicted probabilities, shape (n_samples, n_classes)

        Returns:
            CalibrationMetrics with before/after quality measures
        """
        n_samples, n_classes = y_prob.shape

        # ── Compute pre-calibration metrics ──
        raw_max_probs = y_prob.max(axis=1)
        raw_predictions = y_prob.argmax(axis=1)
        raw_correct = (raw_predictions == y_true).astype(float)

        ece_before = self._expected_calibration_error(raw_max_probs, raw_correct)
        mce_before = self._max_calibration_error(raw_max_probs, raw_correct)
        brier_before = self._brier_score(y_true, y_prob)

        # ── Fit per-class calibrators ──
        for cls in range(n_classes):
            cls_probs = y_prob[:, cls]
            cls_labels = (y_true == cls).astype(float)

            if cls_labels.sum() < 5:
                # Too few samples for this class — skip calibration
                continue

            if self.method == "platt":
                # Platt scaling: fit logistic regression on raw probabilities
                calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                calibrator.fit(cls_probs.reshape(-1, 1), cls_labels)
            else:
                # Isotonic regression: non-parametric monotonic fit
                calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                calibrator.fit(cls_probs, cls_labels)

            self._calibrators[cls] = calibrator

        # ── Fit global calibrator (for max-confidence calibration) ──
        if self.method == "platt":
            self._global_calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            self._global_calibrator.fit(raw_max_probs.reshape(-1, 1), raw_correct)
        else:
            self._global_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            self._global_calibrator.fit(raw_max_probs, raw_correct)

        self.is_fitted = True

        # ── Compute post-calibration metrics ──
        calibrated_probs = self.calibrate(y_prob)
        cal_max_probs = calibrated_probs.max(axis=1)
        cal_predictions = calibrated_probs.argmax(axis=1)
        cal_correct = (cal_predictions == y_true).astype(float)

        ece_after = self._expected_calibration_error(cal_max_probs, cal_correct)
        mce_after = self._max_calibration_error(cal_max_probs, cal_correct)
        brier_after = self._brier_score(y_true, calibrated_probs)

        # ── Build reliability diagram data ──
        frac_pos_before, mean_pred_before = calibration_curve(
            raw_correct, raw_max_probs, n_bins=10, strategy='uniform'
        )
        frac_pos_after, mean_pred_after = calibration_curve(
            cal_correct, cal_max_probs, n_bins=10, strategy='uniform'
        )

        self.metrics = CalibrationMetrics(
            method=self.method,
            ece_before=ece_before,
            ece_after=ece_after,
            mce_before=mce_before,
            mce_after=mce_after,
            brier_before=brier_before,
            brier_after=brier_after,
            reliability_data={
                "before": {"mean_predicted": mean_pred_before.tolist(),
                           "fraction_positive": frac_pos_before.tolist()},
                "after": {"mean_predicted": mean_pred_after.tolist(),
                          "fraction_positive": frac_pos_after.tolist()},
            },
        )

        return self.metrics

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities.

        Args:
            y_prob: Raw predicted probabilities, shape (n_samples, n_classes)

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            return y_prob  # Passthrough if not fitted

        n_samples, n_classes = y_prob.shape
        calibrated = np.zeros_like(y_prob)

        for cls in range(n_classes):
            if cls in self._calibrators:
                if self.method == "platt":
                    calibrated[:, cls] = self._calibrators[cls].predict_proba(
                        y_prob[:, cls].reshape(-1, 1)
                    )[:, 1]
                else:
                    calibrated[:, cls] = self._calibrators[cls].predict(y_prob[:, cls])
            else:
                calibrated[:, cls] = y_prob[:, cls]

        # Re-normalize so rows sum to 1.0
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # avoid division by zero
        calibrated = calibrated / row_sums

        return calibrated

    def calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Calibrate a single confidence value (the max probability from a prediction).
        Used by the preload gate to get a calibrated confidence for threshold comparison.

        Args:
            raw_confidence: Raw max probability from XGBoost (0.0 to 1.0)

        Returns:
            Calibrated confidence (0.0 to 1.0)
        """
        if not self.is_fitted or self._global_calibrator is None:
            return raw_confidence

        if self.method == "platt":
            return float(self._global_calibrator.predict_proba(
                np.array([[raw_confidence]])
            )[0, 1])
        else:
            return float(self._global_calibrator.predict(
                np.array([raw_confidence])
            )[0])

    def _expected_calibration_error(self, confidences: np.ndarray,
                                     accuracies: np.ndarray, n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE): weighted average of |accuracy - confidence|
        across binned confidence ranges.
        """
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue

            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)

            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def _max_calibration_error(self, confidences: np.ndarray,
                                accuracies: np.ndarray, n_bins: int = 10) -> float:
        """Maximum Calibration Error: worst-case bin miscalibration."""
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue

            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            mce = max(mce, abs(bin_accuracy - bin_confidence))

        return mce

    def _brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Multi-class Brier score: mean squared error of probability estimates."""
        n_samples, n_classes = y_prob.shape
        one_hot = np.zeros_like(y_prob)
        one_hot[np.arange(n_samples), y_true] = 1.0
        return float(np.mean((y_prob - one_hot) ** 2))

    def save(self, path: str):
        """Save calibrator to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrators': self._calibrators,
                'global_calibrator': self._global_calibrator,
                'is_fitted': self.is_fitted,
                'metrics': self.metrics,
            }, f)

    def load(self, path: str):
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.method = data['method']
        self._calibrators = data['calibrators']
        self._global_calibrator = data['global_calibrator']
        self.is_fitted = data['is_fitted']
        self.metrics = data.get('metrics')

    def print_report(self):
        """Print calibration quality report."""
        if self.metrics is None:
            print("  No calibration metrics available. Call fit() first.")
            return

        m = self.metrics
        print(f"\n  ╔══════════════════════════════════════════════════════╗")
        print(f"  ║  CONFIDENCE CALIBRATION REPORT ({m.method.upper()})          ║")
        print(f"  ╠══════════════════════════════════════════════════════╣")
        print(f"  ║  Expected Calibration Error (ECE):                  ║")
        print(f"  ║    Before: {m.ece_before:.4f}    After: {m.ece_after:.4f}  ({m.improvement():+.1f}%)   ║")
        print(f"  ║  Max Calibration Error (MCE):                       ║")
        print(f"  ║    Before: {m.mce_before:.4f}    After: {m.mce_after:.4f}              ║")
        print(f"  ║  Brier Score:                                       ║")
        print(f"  ║    Before: {m.brier_before:.4f}    After: {m.brier_after:.4f}              ║")
        print(f"  ╚══════════════════════════════════════════════════════╝")


def run_calibration():
    """Standalone calibration runner."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pandas as pd

    print("\n  Loading data and model...", flush=True)
    df = pd.read_csv("data/synthetic_events.csv")

    # Load trained model
    from src.prediction.xgboost_predictor import XGBoostPredictor
    predictor = XGBoostPredictor()
    predictor.load_model(
        "models_trained/xgboost_model.json",
        "models_trained/feature_engine.pkl",
        "models_trained/markov_matrix.pkl",
    )

    # Use test split for calibration fitting
    split_idx = int(len(df) * 0.7)
    val_df = df.iloc[split_idx:].copy()

    # Get raw probabilities
    print("  Getting raw predictions on validation set...", flush=True)
    fe = predictor.feature_engine
    X_val = fe.transform(val_df)
    y_val = fe.get_target(val_df)

    y_prob = predictor.model.predict_proba(X_val)
    y_true = y_val

    # Fit calibrator
    print("  Fitting Platt scaling calibrator...", flush=True)
    calibrator = ConfidenceCalibrator(method="platt")
    metrics = calibrator.fit(y_true, y_prob)
    calibrator.print_report()

    # Save
    os.makedirs("models_trained", exist_ok=True)
    calibrator.save("models_trained/confidence_calibrator.pkl")
    print(f"\n  Saved to models_trained/confidence_calibrator.pkl")

    return calibrator


if __name__ == "__main__":
    run_calibration()
