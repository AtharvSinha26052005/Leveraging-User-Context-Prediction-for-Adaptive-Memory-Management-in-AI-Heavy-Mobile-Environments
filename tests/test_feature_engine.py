"""
Unit tests for FeatureEngine.

Verifies:
- Correct number of features (19)
- No NaN values in output
- Shape consistency
- Encoding correctness
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.feature_engine import FeatureEngine


@pytest.fixture
def sample_data():
    """Create a small but realistic dataset matching synthetic generator output."""
    data = {
        "current_app": ["WhatsApp", "YouTube", "Instagram", "Gmail", "WhatsApp"] * 20,
        "prev_app_1": ["YouTube", "WhatsApp", "YouTube", "WhatsApp", "Instagram"] * 20,
        "prev_app_2": ["Instagram", "Instagram", "WhatsApp", "YouTube", "YouTube"] * 20,
        "prev_app_3": ["Gmail", "Gmail", "Gmail", "Instagram", "Gmail"] * 20,
        "hour": [9, 12, 15, 18, 21] * 20,
        "day_of_week": [0, 1, 2, 3, 4] * 20,
        "battery": [85, 72, 60, 45, 30] * 20,
        "is_charging": [0, 0, 0, 1, 1] * 20,
        "ram_free_mb": [2000, 1800, 1500, 1200, 800] * 20,
        "ram_total_mb": [6144] * 100,
        "network_type": ["wifi", "mobile", "wifi", "mobile", "wifi"] * 20,
        "ai_task_active": [0, 0, 1, 0, 0] * 20,
        "app_category": ["messaging", "entertainment", "social", "productivity", "messaging"] * 20,
        "persona": ["student"] * 100,
        "session_id": ["s1"] * 50 + ["s2"] * 50,
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="5min").strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "next_app": ["YouTube", "Instagram", "Gmail", "WhatsApp", "YouTube"] * 20,
    }
    return pd.DataFrame(data)


@pytest.fixture
def fitted_engine(sample_data):
    """Return a fitted FeatureEngine."""
    fe = FeatureEngine()
    fe.fit(sample_data)
    return fe


class TestFeatureEngine:
    """Test suite for the 19-feature extraction pipeline."""

    def test_fit_does_not_crash(self, sample_data):
        """FeatureEngine.fit() should complete without errors."""
        fe = FeatureEngine()
        fe.fit(sample_data)
        # After fit + get_target, n_classes should be populated
        y = fe.get_target(sample_data)
        assert len(np.unique(y)) > 0

    def test_transform_produces_correct_shape(self, fitted_engine, sample_data):
        """Output should have (n_samples, n_features) shape."""
        X = fitted_engine.transform(sample_data)
        assert X.ndim == 2
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == len(fitted_engine.FEATURE_NAMES)

    def test_no_nan_values(self, fitted_engine, sample_data):
        """Transformed features should contain no NaN values."""
        X = fitted_engine.transform(sample_data)
        assert not np.any(np.isnan(X)), "Found NaN values in feature matrix"

    def test_no_inf_values(self, fitted_engine, sample_data):
        """Transformed features should contain no infinite values."""
        X = fitted_engine.transform(sample_data)
        assert not np.any(np.isinf(X)), "Found infinite values in feature matrix"

    def test_target_encoding(self, fitted_engine, sample_data):
        """Target values should be non-negative integers."""
        y = fitted_engine.get_target(sample_data)
        assert np.all(y >= 0)

    def test_feature_count_is_19(self, fitted_engine):
        """Should produce exactly 19 features as documented."""
        n_features = len(fitted_engine.FEATURE_NAMES)
        assert n_features == 19, f"Expected 19 features, got {n_features}"

    def test_save_load_produces_same_shape(self, fitted_engine, sample_data, tmp_path):
        """Saved and loaded engine should produce same shape output."""
        X_before = fitted_engine.transform(sample_data)

        save_path = str(tmp_path / "feature_engine.pkl")
        fitted_engine.save(save_path)

        loaded_fe = FeatureEngine()
        loaded_fe.load(save_path)

        X_after = loaded_fe.transform(sample_data)
        assert X_before.shape == X_after.shape
