"""
Cache Scoring Engine — computes the 6-factor composite priority score for each cached app.

S(a,t) = w_R * Recency + w_F * Frequency + w_P * Prediction
       + w_I * Importance + w_C * Context - w_M * MemoryPressure
"""
import math
from datetime import datetime
from typing import List, Tuple, Optional

from src.models.memory import AppCacheEntry, MemoryState
from src.models.prediction import PredictionResult


class CacheScoringEngine:
    """Computes and ranks cache priority scores using the 6-factor formula."""

    def __init__(
        self,
        w_recency: float = 0.25,
        w_frequency: float = 0.20,
        w_prediction: float = 0.25,
        w_importance: float = 0.15,
        w_context: float = 0.15,
        w_pressure: float = 0.30,
        recency_decay: float = 0.001,      # lambda for exp decay
        pressure_threshold: float = 0.70,   # RAM usage fraction to start pressure
    ):
        self.w_recency = w_recency
        self.w_frequency = w_frequency
        self.w_prediction = w_prediction
        self.w_importance = w_importance
        self.w_context = w_context
        self.w_pressure = w_pressure
        self.recency_decay = recency_decay
        self.pressure_threshold = pressure_threshold

    def compute_score(
        self,
        entry: AppCacheEntry,
        now: datetime,
        prediction: PredictionResult,
        memory_state: MemoryState,
        hour: int = 12,
        max_access_count: int = 1,
    ) -> float:
        """
        Compute the composite cache priority score for a single app.
        Higher score = more important = keep in cache.
        """
        # ── R: Recency (exponential decay) ──
        dt = entry.time_since_access(now)
        R = math.exp(-self.recency_decay * dt)

        # ── F: Frequency (normalized by max in cache) ──
        F = entry.access_count / max(max_access_count, 1)
        F = min(F, 1.0)

        # ── P: Prediction (ML model's probability for this app) ──
        P = prediction.get_app_probability(entry.app_name)

        # ── I: Importance (static category-based weight) ──
        I = entry.importance

        # ── C: Context relevance ──
        C = self._compute_context_relevance(entry, hour)

        # ── M: Memory pressure penalty ──
        M = self._compute_pressure_penalty(memory_state)

        # ── Composite score ──
        score = (
            self.w_recency * R
            + self.w_frequency * F
            + self.w_prediction * P
            + self.w_importance * I
            + self.w_context * C
            - self.w_pressure * M
        )

        return round(score, 6)

    def rank_all(
        self,
        memory_state: MemoryState,
        prediction: PredictionResult,
        now: datetime,
        hour: int = 12,
    ) -> List[Tuple[str, float]]:
        """
        Score all cached apps and return sorted list (ascending = evict first).
        Returns: [(app_name, score), ...] — lowest score first.
        """
        if not memory_state.cached_apps:
            return []

        max_access = max(e.access_count for e in memory_state.cached_apps.values())

        scores = []
        for app_name, entry in memory_state.cached_apps.items():
            # Don't score the foreground app — it's always kept
            if app_name == memory_state.foreground_app:
                scores.append((app_name, 999.0))
                continue

            score = self.compute_score(entry, now, prediction, memory_state, hour, max_access)
            scores.append((app_name, score))

        # Sort ascending: lowest score = first to evict
        scores.sort(key=lambda x: x[1])
        return scores

    def get_eviction_candidates(
        self,
        memory_state: MemoryState,
        prediction: PredictionResult,
        now: datetime,
        hour: int = 12,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Get apps eligible for eviction (lowest scores, excluding foreground)."""
        ranked = self.rank_all(memory_state, prediction, now, hour)
        # Exclude foreground (score=999) and apps above min_score threshold
        return [(app, score) for app, score in ranked if score < 999 and score <= min_score + 0.5]

    def _compute_context_relevance(self, entry: AppCacheEntry, hour: int) -> float:
        """How relevant is this app to the current context (time-based)."""
        category = entry.category

        # Morning (7-9): navigation, productivity high
        if 7 <= hour <= 9:
            if category in ("navigation", "productivity"):
                return 0.9
            elif category in ("messaging",):
                return 0.7
            else:
                return 0.3

        # Work hours (9-17): productivity high
        elif 9 < hour <= 17:
            if category in ("productivity",):
                return 0.9
            elif category in ("messaging", "finance"):
                return 0.7
            elif category in ("entertainment", "gaming"):
                return 0.2
            else:
                return 0.4

        # Evening (17-22): entertainment, social high
        elif 17 < hour <= 22:
            if category in ("entertainment", "social"):
                return 0.9
            elif category in ("messaging",):
                return 0.7
            elif category in ("productivity",):
                return 0.3
            else:
                return 0.5

        # Night (22-7): social, messaging
        else:
            if category in ("social", "messaging"):
                return 0.8
            elif category in ("entertainment",):
                return 0.6
            else:
                return 0.2

    def _compute_pressure_penalty(self, memory_state: MemoryState) -> float:
        """Memory pressure penalty — increases as RAM fills up."""
        utilization = memory_state.ram_utilization

        if utilization < self.pressure_threshold:
            return 0.0  # no pressure
        elif utilization < 0.90:
            # Linear ramp from 0 to 0.5
            return 0.5 * (utilization - self.pressure_threshold) / (0.90 - self.pressure_threshold)
        else:
            # Critical — full penalty
            return 1.0
