"""
XGBoost Predictor — primary ML model for app prediction.
Uses 19 contextual features to predict the next app with confidence scores.
"""
import numpy as np
import json
from typing import List, Tuple, Optional
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from src.models.prediction import PredictionResult
from src.prediction.feature_engine import FeatureEngine
from src.prediction.markov_predictor import MarkovPredictor


class XGBoostPredictor:
    """
    Primary predictor using XGBoost gradient boosted trees.
    Falls back to Markov chain when confidence is below threshold.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        fallback_threshold: float = 0.3,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        self.feature_engine: Optional[FeatureEngine] = None
        self.markov_fallback: Optional[MarkovPredictor] = None
        self.fallback_threshold = fallback_threshold
        self.is_fitted = False

        # Training metrics
        self.train_accuracy: float = 0.0
        self.cv_accuracy: float = 0.0
        self.top3_accuracy: float = 0.0

    def fit(self, train_df, val_df=None) -> dict:
        """Train XGBoost + Markov fallback on training data."""
        print("    Training feature engine...")
        self.feature_engine = FeatureEngine()
        self.feature_engine.fit(train_df)

        X_train = self.feature_engine.transform(train_df)
        y_train = self.feature_engine.get_target(train_df)

        # Train Markov fallback
        print("    Training Markov fallback...")
        self.markov_fallback = MarkovPredictor()
        self.markov_fallback.fit(
            train_df["current_app"].tolist(),
            train_df["next_app"].tolist(),
        )

        # Train XGBoost
        print(f"    Training XGBoost ({self.model.n_estimators} trees, depth {self.model.max_depth})...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate on training set
        y_pred_train = self.model.predict(X_train)
        self.train_accuracy = accuracy_score(y_train, y_pred_train)

        # Cross-validation
        print("    Running 5-fold stratified CV...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="accuracy")
        self.cv_accuracy = cv_scores.mean()

        # Top-3 accuracy
        y_prob = self.model.predict_proba(X_train)
        n_classes = y_prob.shape[1]
        k = min(3, n_classes)
        self.top3_accuracy = top_k_accuracy_score(y_train, y_prob, k=k)

        metrics = {
            "train_accuracy": round(self.train_accuracy, 4),
            "cv_accuracy_mean": round(self.cv_accuracy, 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
            "n_classes": n_classes,
            "n_train_samples": len(train_df),
            "n_features": X_train.shape[1],
        }

        # Validation set evaluation
        if val_df is not None and len(val_df) > 0:
            X_val = self.feature_engine.transform(val_df)
            y_val = self.feature_engine.get_target(val_df)
            y_pred_val = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)
            y_prob_val = self.model.predict_proba(X_val)
            val_top3 = top_k_accuracy_score(y_val, y_prob_val, k=k)
            metrics["val_accuracy"] = round(val_acc, 4)
            metrics["val_top3_accuracy"] = round(val_top3, 4)
            print(f"    Validation: top-1 = {val_acc:.4f}, top-3 = {val_top3:.4f}")

        print(f"    Training complete: CV = {self.cv_accuracy:.4f}, Top-3 = {self.top3_accuracy:.4f}")
        return metrics

    def predict(self, event_row: dict, top_k: int = 5) -> PredictionResult:
        """Predict next app from a single event dict."""
        import pandas as pd

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Transform single event
        df = pd.DataFrame([event_row])
        X = self.feature_engine.transform(df)

        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        classes = self.feature_engine.target_classes

        # Build top-k list
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_list = [(classes[i], float(probs[i])) for i in top_indices]

        predicted_app = top_k_list[0][0]
        confidence = top_k_list[0][1]

        # Fallback to Markov if confidence is too low
        if confidence < self.fallback_threshold and self.markov_fallback:
            markov_result = self.markov_fallback.predict(
                event_row.get("current_app", ""), top_k=top_k
            )
            return PredictionResult(
                predicted_app=markov_result.predicted_app,
                confidence=markov_result.confidence,
                top_k=markov_result.top_k,
                model_used="markov_fallback",
            )

        return PredictionResult(
            predicted_app=predicted_app,
            confidence=confidence,
            top_k=top_k_list,
            model_used="xgboost",
        )

    def predict_batch(self, df) -> List[PredictionResult]:
        """Predict for a batch of events."""
        X = self.feature_engine.transform(df)
        probs = self.model.predict_proba(X)
        classes = self.feature_engine.target_classes

        results = []
        for i in range(len(df)):
            row_probs = probs[i]
            top_indices = np.argsort(row_probs)[::-1][:5]
            top_k_list = [(classes[j], float(row_probs[j])) for j in top_indices]

            results.append(PredictionResult(
                predicted_app=top_k_list[0][0],
                confidence=top_k_list[0][1],
                top_k=top_k_list,
                model_used="xgboost",
            ))
        return results

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get feature importance rankings."""
        if not self.is_fitted:
            return []
        importances = self.model.feature_importances_
        names = FeatureEngine.FEATURE_NAMES
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return pairs

    def save(self, model_path: str, engine_path: str, markov_path: str) -> None:
        """Save model, feature engine, and markov fallback."""
        self.model.save_model(model_path)
        self.feature_engine.save(engine_path)
        if self.markov_fallback:
            self.markov_fallback.save(markov_path)

    def load_model(self, model_path: str, engine_path: str, markov_path: str) -> "XGBoostPredictor":
        """Load all components."""
        self.model.load_model(model_path)
        self.feature_engine = FeatureEngine.load(engine_path)
        self.markov_fallback = MarkovPredictor()
        self.markov_fallback.load_model(markov_path)
        self.is_fitted = True
        return self
