"""
Unit tests for CacheScoringEngine.

Verifies:
- Score is computed for cached apps
- Predicted apps score higher than non-predicted
- Recency, frequency, importance factors work
"""
import sys
import os
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.cache_scorer import CacheScoringEngine
from src.models.memory import MemoryState, AppCacheEntry
from src.models.prediction import PredictionResult
from src.data.app_profiles import get_profile


def make_entry(app="WhatsApp", last_delta_s=60, access_count=10):
    """Create an AppCacheEntry using real app profile data."""
    now = datetime.now()
    profile = get_profile(app)
    return AppCacheEntry(
        app_name=app, memory_mb=profile.memory_mb,
        last_accessed=now - timedelta(seconds=last_delta_s),
        access_count=access_count, category=profile.category,
        importance=profile.importance,
    )


def make_prediction(app="WhatsApp", confidence=0.8):
    return PredictionResult(
        predicted_app=app, confidence=confidence,
        top_k=[(app, confidence)], model_used="xgboost",
    )


def make_memory_state(total_ram=6144, reserved=1500):
    """MemoryState takes total_ram_mb; free_ram_mb is computed."""
    return MemoryState(total_ram_mb=total_ram, system_reserved_mb=reserved)


class TestCacheScorer:
    """Test suite for the 6-factor cache scoring formula."""

    def setup_method(self):
        self.scorer = CacheScoringEngine()
        self.now = datetime.now()

    def test_score_is_numeric(self):
        """compute_score should return a float."""
        entry = make_entry()
        pred = make_prediction()
        mem = make_memory_state()
        score = self.scorer.compute_score(entry, self.now, pred, mem, hour=12)
        assert isinstance(score, (int, float))

    def test_predicted_app_scores_higher(self):
        """An app matching the prediction should score higher."""
        entry = make_entry(app="WhatsApp", last_delta_s=60)
        mem = make_memory_state()

        pred_match = make_prediction(app="WhatsApp", confidence=0.9)
        pred_miss = make_prediction(app="YouTube", confidence=0.9)

        score_predicted = self.scorer.compute_score(entry, self.now, pred_match, mem, hour=12)
        score_not_predicted = self.scorer.compute_score(entry, self.now, pred_miss, mem, hour=12)

        assert score_predicted >= score_not_predicted, (
            f"Predicted ({score_predicted}) should >= non-predicted ({score_not_predicted})"
        )

    def test_recent_app_scores_higher(self):
        """Recently accessed app should score higher than stale app."""
        mem = make_memory_state()
        pred = make_prediction(app="Other")

        entry_recent = make_entry(app="Gmail", last_delta_s=5)
        entry_stale = make_entry(app="Gmail", last_delta_s=3600)

        score_recent = self.scorer.compute_score(entry_recent, self.now, pred, mem, hour=12)
        score_stale = self.scorer.compute_score(entry_stale, self.now, pred, mem, hour=12)

        assert score_recent > score_stale, (
            f"Recent ({score_recent}) should beat stale ({score_stale})"
        )

    def test_frequent_app_scores_higher(self):
        """Frequently used app should score higher."""
        mem = make_memory_state()
        pred = make_prediction(app="Other")

        entry_frequent = make_entry(app="Instagram", access_count=80)
        entry_rare = make_entry(app="Instagram", access_count=2)

        score_frequent = self.scorer.compute_score(
            entry_frequent, self.now, pred, mem, hour=15, max_access_count=80
        )
        score_rare = self.scorer.compute_score(
            entry_rare, self.now, pred, mem, hour=15, max_access_count=80
        )

        assert score_frequent > score_rare, (
            f"Frequent ({score_frequent}) should beat rare ({score_rare})"
        )

    def test_get_eviction_candidates_returns_list(self):
        """get_eviction_candidates should return a list."""
        mem = make_memory_state()
        profile = get_profile("WhatsApp")
        mem.add_to_cache(profile, self.now)

        pred = make_prediction(app="YouTube")
        candidates = self.scorer.get_eviction_candidates(mem, pred, self.now)

        assert isinstance(candidates, list)

    def test_rank_all_returns_list(self):
        """rank_all should return a list of (app, score) tuples."""
        mem = make_memory_state()

        for app in ["WhatsApp", "YouTube"]:
            profile = get_profile(app)
            mem.add_to_cache(profile, self.now - timedelta(seconds=30))

        pred = make_prediction(app="WhatsApp")
        ranked = self.scorer.rank_all(mem, pred, self.now)

        assert isinstance(ranked, list)
        assert len(ranked) == 2
