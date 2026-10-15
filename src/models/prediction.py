"""
Data models for ML prediction outputs.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple


@dataclass
class PredictionResult:
    """Output of the prediction layer — which app the user will open next."""
    predicted_app: str
    confidence: float                          # 0.0 – 1.0
    top_k: List[Tuple[str, float]]             # [(app_name, probability), ...]
    model_used: str                            # "xgboost", "markov", "fallback"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def top3_apps(self) -> List[str]:
        return [app for app, _ in self.top_k[:3]]

    @property
    def is_confident(self) -> bool:
        """True if confidence is above the preload threshold."""
        return self.confidence >= 0.6

    def get_app_probability(self, app_name: str) -> float:
        """Get the predicted probability for a specific app."""
        for app, prob in self.top_k:
            if app == app_name:
                return prob
        return 0.0
