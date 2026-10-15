"""
Markov Chain Predictor — baseline model.
Predicts the next app based only on the current app using a transition probability matrix.
"""
import numpy as np
import pickle
from collections import defaultdict
from typing import List, Tuple, Optional
from datetime import datetime

from src.models.prediction import PredictionResult


class MarkovPredictor:
    """First-order Markov chain: P(next_app | current_app)."""

    def __init__(self):
        self.transition_counts: dict = defaultdict(lambda: defaultdict(int))
        self.total_counts: dict = defaultdict(int)
        self.app_list: List[str] = []
        self.is_fitted = False

    def fit(self, current_apps: List[str], next_apps: List[str]) -> "MarkovPredictor":
        """Build transition matrix from observed (current → next) pairs."""
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)

        for curr, nxt in zip(current_apps, next_apps):
            self.transition_counts[curr][nxt] += 1
            self.total_counts[curr] += 1

        self.app_list = sorted(set(list(current_apps) + list(next_apps)))
        self.is_fitted = True
        return self

    def predict(self, current_app: str, top_k: int = 5) -> PredictionResult:
        """Predict next app given current app."""
        if not self.is_fitted or current_app not in self.transition_counts:
            # Uniform fallback for unseen apps
            prob = 1.0 / max(len(self.app_list), 1)
            top_k_list = [(app, prob) for app in self.app_list[:top_k]]
            return PredictionResult(
                predicted_app=self.app_list[0] if self.app_list else "unknown",
                confidence=prob,
                top_k=top_k_list,
                model_used="markov_fallback",
            )

        counts = self.transition_counts[current_app]
        total = self.total_counts[current_app]

        # Build probability distribution
        probs = []
        for app in self.app_list:
            p = counts.get(app, 0) / total if total > 0 else 0
            probs.append((app, p))

        # Sort by probability descending
        probs.sort(key=lambda x: x[1], reverse=True)
        top_k_list = probs[:top_k]

        return PredictionResult(
            predicted_app=top_k_list[0][0],
            confidence=top_k_list[0][1],
            top_k=top_k_list,
            model_used="markov",
        )

    def get_transition_matrix(self) -> np.ndarray:
        """Return the full transition matrix as a numpy array."""
        n = len(self.app_list)
        matrix = np.zeros((n, n))
        app_idx = {app: i for i, app in enumerate(self.app_list)}

        for curr, nexts in self.transition_counts.items():
            total = self.total_counts[curr]
            if total > 0 and curr in app_idx:
                for nxt, count in nexts.items():
                    if nxt in app_idx:
                        matrix[app_idx[curr]][app_idx[nxt]] = count / total
        return matrix

    def save(self, path: str) -> None:
        data = {
            "transition_counts": dict(self.transition_counts),
            "total_counts": dict(self.total_counts),
            "app_list": self.app_list,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, path: str) -> "MarkovPredictor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.transition_counts = defaultdict(lambda: defaultdict(int), data["transition_counts"])
        for k, v in self.transition_counts.items():
            self.transition_counts[k] = defaultdict(int, v)
        self.total_counts = defaultdict(int, data["total_counts"])
        self.app_list = data["app_list"]
        self.is_fitted = True
        return self
