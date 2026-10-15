"""
Unit tests for ThrashingDetector.

Verifies:
- Safe mode activates when thrashing score exceeds threshold
- Score is bounded [0, 1]
- Sliding window correctly ages out old events
"""
import sys
import os
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.thrashing_detector import ThrashingDetector


class TestThrashingDetector:
    """Test suite for thrashing detection and Safe Mode."""

    def setup_method(self):
        self.detector = ThrashingDetector(
            window_seconds=120.0,
            enter_threshold=0.70,
            exit_threshold=0.30,
            stability_period=10.0,
        )
        self.now = datetime.now()

    def test_initial_state_normal(self):
        """Detector should start in NORMAL mode with zero score."""
        # mode is an enum, compare with .value
        assert self.detector.mode.value == "normal"
        assert self.detector.get_score(self.now) == 0.0

    def test_score_bounded(self):
        """Thrashing score should always be in [0, 1]."""
        for i in range(50):
            t = self.now + timedelta(seconds=i * 0.1)
            self.detector.record_eviction(f"app_{i % 5}", t)
            self.detector.record_reload(f"app_{i % 5}", t)
            self.detector.record_cache_miss(f"app_{i % 5}", t)

        score = self.detector.get_score(self.now + timedelta(seconds=5))
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"

    def test_safe_mode_activates_on_thrashing(self):
        """Sustained high evict-reload should raise the score."""
        for i in range(30):
            t = self.now + timedelta(seconds=i * 0.5)
            self.detector.record_eviction(f"app_{i % 3}", t)
            self.detector.record_reload(f"app_{i % 3}", t)
            self.detector.record_cache_miss(f"app_{i % 3}", t)

        score = self.detector.get_score(self.now + timedelta(seconds=15))
        assert score > 0.0, f"Score should be elevated, got {score}"

    def test_no_events_means_zero_score(self):
        """With no events recorded, score should be zero."""
        score = self.detector.get_score(self.now)
        assert score == 0.0

    def test_eviction_increases_score(self):
        """Recording evictions should increase the thrashing score."""
        score_before = self.detector.get_score(self.now)

        for i in range(10):
            self.detector.record_eviction(f"app_{i}", self.now + timedelta(seconds=i))

        score_after = self.detector.get_score(self.now + timedelta(seconds=10))
        assert score_after >= score_before, (
            f"Score should increase after evictions: {score_before} → {score_after}"
        )

    def test_old_events_age_out(self):
        """Events outside the sliding window should not affect score."""
        past = self.now - timedelta(seconds=200)

        for i in range(20):
            self.detector.record_eviction(f"app_{i}", past + timedelta(seconds=i))

        score = self.detector.get_score(self.now)
        assert score == 0.0, f"Old events should have aged out, got score={score}"

    def test_reset_clears_state(self):
        """Reset should clear all events and return to NORMAL."""
        for i in range(10):
            self.detector.record_eviction("app", self.now + timedelta(seconds=i))

        self.detector.reset()
        assert self.detector.mode.value == "normal"
        assert self.detector.get_score(self.now) == 0.0
