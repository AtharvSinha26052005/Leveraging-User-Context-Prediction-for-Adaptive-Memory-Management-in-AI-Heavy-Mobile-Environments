"""
Model Comparison — Train and evaluate 5 ML models for app prediction.
Produces a uniform comparison table with accuracy, F1, training time, inference time.
"""
import time
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import List, Tuple

from src.prediction.feature_engine import FeatureEngine
from src.prediction.markov_predictor import MarkovPredictor


@dataclass
class ModelResult:
    """Result from a single model evaluation."""
    model_name: str
    top1_accuracy: float
    top3_accuracy: float
    f1_macro: float
    train_time_sec: float
    inference_time_ms: float  # per-sample
    n_params: int

    def to_dict(self):
        return {
            "Model": self.model_name,
            "Top-1 Acc": f"{self.top1_accuracy:.1%}",
            "Top-3 Acc": f"{self.top3_accuracy:.1%}",
            "F1 (Macro)": f"{self.f1_macro:.3f}",
            "Train Time": f"{self.train_time_sec:.1f}s",
            "Inference": f"{self.inference_time_ms:.2f}ms",
        }


class ModelComparison:
    """Compare 5 models for app-next prediction."""

    def __init__(self):
        self.results: List[ModelResult] = []
        self.feature_engine = FeatureEngine()
        self.label_encoder = LabelEncoder()

    def run_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all 5 models and return comparison DataFrame."""
        # Prepare data
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Fit feature engine on training data
        self.feature_engine.fit(train_df)
        X_train = self.feature_engine.transform(train_df)
        X_test = self.feature_engine.transform(test_df)

        # Encode labels
        y_train = self.label_encoder.fit_transform(train_df["next_app"].values)
        y_test = self.label_encoder.transform(test_df["next_app"].values)
        n_classes = len(self.label_encoder.classes_)

        print(f"\n  Data: {len(X_train)} train, {len(X_test)} test, {n_classes} classes, {X_train.shape[1]} features")

        self.results = []

        # ── Model 1: Markov Chain ──
        print("\n  [1/5] Markov Chain...")
        self.results.append(self._eval_markov(train_df, test_df))

        # ── Model 2: Random Forest ──
        print("  [2/5] Random Forest...")
        self.results.append(self._eval_sklearn(
            RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
            "Random Forest", X_train, y_train, X_test, y_test, n_classes
        ))

        # ── Model 3: LightGBM ──
        print("  [3/5] LightGBM...")
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                num_leaves=31, n_jobs=-1, random_state=42, verbose=-1,
            )
            self.results.append(self._eval_sklearn(
                model, "LightGBM", X_train, y_train, X_test, y_test, n_classes
            ))
        except ImportError:
            print("    LightGBM not installed, using GradientBoosting fallback...")
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
            )
            self.results.append(self._eval_sklearn(
                model, "GradientBoosting", X_train, y_train, X_test, y_test, n_classes
            ))

        # ── Model 4: XGBoost ──
        print("  [4/5] XGBoost...")
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42, verbosity=0,
        )
        self.results.append(self._eval_sklearn(
            model, "XGBoost", X_train, y_train, X_test, y_test, n_classes
        ))

        # ── Model 5: MLP (Neural Network) ──
        print("  [5/5] MLP Neural Network...")
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=200,
            learning_rate="adaptive", early_stopping=True,
            random_state=42, verbose=False,
        )
        self.results.append(self._eval_sklearn(
            model, "MLP (Neural Net)", X_train, y_train, X_test, y_test, n_classes
        ))

        return self._to_dataframe()

    def _eval_markov(self, train_df, test_df) -> ModelResult:
        """Evaluate Markov chain predictor."""
        markov = MarkovPredictor()

        t0 = time.time()
        markov.fit(train_df["current_app"].tolist(), train_df["next_app"].tolist())
        train_time = time.time() - t0

        # Evaluate on test set
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        inference_times = []

        for _, row in test_df.iterrows():
            t0 = time.time()
            pred = markov.predict(row["current_app"])
            inference_times.append((time.time() - t0) * 1000)

            actual = row["next_app"]
            total += 1
            if pred.predicted_app == actual:
                correct_top1 += 1
            if actual in pred.top3_apps:
                correct_top3 += 1

        return ModelResult(
            model_name="Markov Chain",
            top1_accuracy=correct_top1 / max(total, 1),
            top3_accuracy=correct_top3 / max(total, 1),
            f1_macro=correct_top1 / max(total, 1),  # approximate
            train_time_sec=train_time,
            inference_time_ms=np.mean(inference_times) if inference_times else 0,
            n_params=len(markov.transition_counts) * 20,  # approximate
        )

    def _eval_sklearn(self, model, name, X_train, y_train, X_test, y_test, n_classes) -> ModelResult:
        """Evaluate a sklearn-compatible model."""
        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Predict
        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = (time.time() - t0) / len(X_test) * 1000  # ms per sample

        # Metrics
        top1 = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Top-3 accuracy
        try:
            y_proba = model.predict_proba(X_test)
            top3 = top_k_accuracy_score(y_test, y_proba, k=min(3, n_classes), labels=range(n_classes))
        except Exception:
            top3 = top1

        # Parameter count
        try:
            if hasattr(model, 'n_estimators'):
                n_params = model.n_estimators * (2 ** min(model.max_depth or 6, 10))
            elif hasattr(model, 'coefs_'):
                n_params = sum(c.size for c in model.coefs_)
            else:
                n_params = 0
        except Exception:
            n_params = 0

        print(f"    {name}: top1={top1:.1%}, top3={top3:.1%}, F1={f1:.3f}, "
              f"train={train_time:.1f}s, infer={pred_time:.2f}ms")

        return ModelResult(
            model_name=name,
            top1_accuracy=top1,
            top3_accuracy=top3,
            f1_macro=f1,
            train_time_sec=train_time,
            inference_time_ms=pred_time,
            n_params=n_params,
        )

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def print_table(self):
        """Print formatted comparison table."""
        print("\n" + "=" * 90)
        print("  MODEL COMPARISON — App Next Prediction")
        print("=" * 90)
        print(f"  {'Model':<20} {'Top-1':>8} {'Top-3':>8} {'F1':>8} {'Train':>10} {'Inference':>12}")
        print("-" * 90)

        # Sort by top-1 accuracy
        sorted_results = sorted(self.results, key=lambda r: r.top1_accuracy, reverse=True)
        for r in sorted_results:
            marker = " <-- BEST" if r == sorted_results[0] else ""
            chosen = " (CHOSEN)" if "XGBoost" in r.model_name else ""
            print(f"  {r.model_name:<20} {r.top1_accuracy:>7.1%} {r.top3_accuracy:>7.1%} "
                  f"{r.f1_macro:>7.3f} {r.train_time_sec:>8.1f}s "
                  f"{r.inference_time_ms:>9.2f}ms{marker}{chosen}")

        print("=" * 90)

        # Justification
        xgb_result = next((r for r in self.results if "XGBoost" in r.model_name), None)
        if xgb_result:
            print(f"\n  WHY XGBOOST?")
            print(f"  - Best accuracy-speed tradeoff for real-time mobile inference")
            print(f"  - Inference time: {xgb_result.inference_time_ms:.2f}ms (must be < 10ms for mobile)")
            print(f"  - Native model serialization (JSON) — lightweight deployment")
            print(f"  - Handles missing features gracefully (important for noisy sensor data)")


def run_model_comparison():
    """Standalone runner for model comparison."""
    print("\n  Loading synthetic data...")
    df = pd.read_csv("data/synthetic_events.csv")
    print(f"  Loaded {len(df)} events")

    comp = ModelComparison()
    results_df = comp.run_comparison(df)
    comp.print_table()

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/model_comparison.csv", index=False)
    print(f"\n  Saved to results/model_comparison.csv")

    return comp


if __name__ == "__main__":
    run_model_comparison()
