"""
Unit tests for PreloadDecisionEngine.

Verifies all 5 gates work independently and in combination.
"""
import sys
import os
import pytest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.preload_gate import PreloadDecisionEngine
from src.models.prediction import PredictionResult
from src.models.memory import MemoryState
from src.data.app_profiles import get_profile


def make_prediction(app="WhatsApp", confidence=0.8):
    return PredictionResult(
        predicted_app=app, confidence=confidence,
        top_k=[(app, confidence)], model_used="xgboost",
    )


def make_memory_state(total_ram=6144, reserved=1500, cached_apps=None):
    """Create MemoryState; free_ram is computed as total - reserved - cached."""
    ms = MemoryState(total_ram_mb=total_ram, system_reserved_mb=reserved)
    now = datetime.now()
    if cached_apps:
        for app in cached_apps:
            profile = get_profile(app)
            ms.add_to_cache(profile, now)
    return ms


def make_low_ram_state():
    """Create a MemoryState with almost no free RAM."""
    ms = MemoryState(total_ram_mb=4096, system_reserved_mb=3700)
    return ms


class TestPreloadGate:
    """Test suite for the 5-condition preload gate."""

    def setup_method(self):
        self.gate = PreloadDecisionEngine(
            confidence_threshold=0.60,
            ram_headroom_mb=500.0,
            min_battery=0.20,
            thrashing_limit=0.70,
        )

    def test_all_gates_pass(self):
        """When all conditions are met, preload should be approved."""
        pred = make_prediction(confidence=0.8)
        mem = make_memory_state()

        should, reason = self.gate.should_preload(
            pred, mem, battery=0.75, thrashing_score=0.1
        )
        assert should is True
        assert "all 5 gates passed" in reason

    def test_gate1_confidence_blocks(self):
        """Low confidence should block preload."""
        pred = make_prediction(confidence=0.3)
        mem = make_memory_state()

        should, reason = self.gate.should_preload(
            pred, mem, battery=0.75, thrashing_score=0.1
        )
        assert should is False
        assert "confidence" in reason

    def test_gate2_ram_blocks(self):
        """Insufficient RAM should block preload."""
        pred = make_prediction(confidence=0.8)
        mem = make_low_ram_state()

        should, reason = self.gate.should_preload(
            pred, mem, battery=0.75, thrashing_score=0.1
        )
        assert should is False
        assert "RAM" in reason

    def test_gate3_battery_blocks(self):
        """Low battery should block preload."""
        pred = make_prediction(confidence=0.8)
        mem = make_memory_state()

        should, reason = self.gate.should_preload(
            pred, mem, battery=0.10, thrashing_score=0.1
        )
        assert should is False
        assert "battery" in reason

    def test_gate4_thrashing_blocks(self):
        """High thrashing score should block preload."""
        pred = make_prediction(confidence=0.8)
        mem = make_memory_state()

        should, reason = self.gate.should_preload(
            pred, mem, battery=0.75, thrashing_score=0.85
        )
        assert should is False
        assert "thrashing" in reason

    def test_gate5_already_cached_blocks(self):
        """App already in cache should block preload."""
        pred = make_prediction(app="WhatsApp", confidence=0.8)
        mem = make_memory_state(cached_apps=["WhatsApp"])

        should, reason = self.gate.should_preload(
            pred, mem, battery=0.75, thrashing_score=0.1
        )
        assert should is False
        assert "already in cache" in reason

    def test_outcome_tracking(self):
        """Preload success rate tracking should work correctly."""
        self.gate.reset_stats()
        self.gate.record_outcome(was_used=True)
        self.gate.record_outcome(was_used=True)
        self.gate.record_outcome(was_used=False)

        assert self.gate.total_preloads == 3
        assert self.gate.successful_preloads == 2
        assert abs(self.gate.success_rate - 2/3) < 0.01

    def test_boundary_confidence(self):
        """Confidence exactly at threshold should pass gate 1."""
        pred = make_prediction(confidence=0.60)
        mem = make_memory_state()

        should, _ = self.gate.should_preload(
            pred, mem, battery=0.75, thrashing_score=0.1
        )
        assert should is True
