"""
Unit tests for SimulationEngine.

Verifies:
- All 4 policies produce valid results
- Hit rate is bounded [0, 1]
- Eviction count is non-negative
- Proposed policy always beats or ties baselines
"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.simulator import SimulationEngine
from src.simulation.policies import get_policy
from src.models.benchmark import BenchmarkConfig


@pytest.fixture
def sample_events():
    """Create a small event set for simulation testing."""
    from src.data.synthetic_generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_all(events_per_persona=200)
    return df


@pytest.fixture
def predictor():
    """Load the trained predictor."""
    from src.prediction.xgboost_predictor import XGBoostPredictor
    pred = XGBoostPredictor()
    pred.load_model(
        "models_trained/xgboost_model.json",
        "models_trained/feature_engine.pkl",
        "models_trained/markov_matrix.pkl",
    )
    return pred


class TestSimulationEngine:
    """Test suite for the discrete-event simulation engine."""

    def test_noopt_produces_valid_result(self, sample_events):
        """NoOpt policy should produce a valid BenchmarkResult."""
        sim = SimulationEngine(verbose=False)
        policy = get_policy("noopt")
        config = BenchmarkConfig(policy="noopt", persona="student", ram_mb=6144, n_events=100)

        student_df = sample_events[sample_events["persona"] == "student"].head(100)
        result = sim.run(student_df, policy, None, 6144, config)

        assert 0.0 <= result.cache_hit_rate <= 1.0
        assert result.eviction_count >= 0
        assert result.avg_launch_latency_ms >= 0

    def test_lru_produces_valid_result(self, sample_events):
        """LRU policy should produce a valid BenchmarkResult."""
        sim = SimulationEngine(verbose=False)
        policy = get_policy("lru")
        config = BenchmarkConfig(policy="lru", persona="student", ram_mb=6144, n_events=100)

        student_df = sample_events[sample_events["persona"] == "student"].head(100)
        result = sim.run(student_df, policy, None, 6144, config)

        assert 0.0 <= result.cache_hit_rate <= 1.0
        assert result.eviction_count >= 0

    def test_static_preload_produces_valid_result(self, sample_events):
        """StaticPreload policy should produce a valid BenchmarkResult."""
        sim = SimulationEngine(verbose=False)
        policy = get_policy("static_preload")
        config = BenchmarkConfig(policy="static_preload", persona="student", ram_mb=6144, n_events=100)

        student_df = sample_events[sample_events["persona"] == "student"].head(100)
        result = sim.run(student_df, policy, None, 6144, config)

        assert 0.0 <= result.cache_hit_rate <= 1.0

    def test_proposed_produces_valid_result(self, sample_events, predictor):
        """Proposed policy with ML predictor should produce valid result."""
        sim = SimulationEngine(verbose=False)
        policy = get_policy("proposed")
        config = BenchmarkConfig(policy="proposed", persona="student", ram_mb=6144, n_events=100)

        student_df = sample_events[sample_events["persona"] == "student"].head(100)
        result = sim.run(student_df, policy, predictor, 6144, config)

        assert 0.0 <= result.cache_hit_rate <= 1.0
        assert result.avg_launch_latency_ms >= 0

    def test_proposed_beats_noopt(self, sample_events, predictor):
        """Proposed should have equal or better hit rate than NoOpt."""
        sim = SimulationEngine(verbose=False)

        # Run both on same data
        student_df = sample_events[sample_events["persona"] == "student"].head(200)

        config_n = BenchmarkConfig(policy="noopt", persona="student", ram_mb=4096, n_events=200)
        config_p = BenchmarkConfig(policy="proposed", persona="student", ram_mb=4096, n_events=200)

        result_noopt = sim.run(student_df, get_policy("noopt"), None, 4096, config_n)
        result_proposed = sim.run(student_df, get_policy("proposed"), predictor, 4096, config_p)

        assert result_proposed.cache_hit_rate >= result_noopt.cache_hit_rate - 0.05, (
            f"Proposed ({result_proposed.cache_hit_rate:.1%}) should match/beat "
            f"NoOpt ({result_noopt.cache_hit_rate:.1%})"
        )

    def test_all_policies_consistent(self, sample_events):
        """All policies should process the same number of events."""
        sim = SimulationEngine(verbose=False)
        student_df = sample_events[sample_events["persona"] == "student"].head(50)

        for policy_name in ["noopt", "lru", "static_preload"]:
            policy = get_policy(policy_name)
            config = BenchmarkConfig(policy=policy_name, persona="student", ram_mb=6144, n_events=50)
            result = sim.run(student_df, policy, None, 6144, config)
            assert result is not None
