"""
Unit tests for memory management policies.

Verifies:
- Each policy instantiates correctly
- Policy factory returns correct types
- Proposed policy has all sub-components
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.policies import (
    get_policy, NoOptPolicy, LRUPolicy, StaticPreloadPolicy, ProposedPolicy
)


class TestPolicyFactory:
    """Test the policy factory function."""

    def test_get_noopt(self):
        policy = get_policy("noopt")
        assert isinstance(policy, NoOptPolicy)

    def test_get_lru(self):
        policy = get_policy("lru")
        assert isinstance(policy, LRUPolicy)

    def test_get_static_preload(self):
        policy = get_policy("static_preload")
        assert isinstance(policy, StaticPreloadPolicy)

    def test_get_proposed(self):
        policy = get_policy("proposed")
        assert isinstance(policy, ProposedPolicy)

    def test_invalid_policy_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_policy("nonexistent_policy")


class TestNoOptPolicy:
    """Test NoOpt (FIFO) baseline policy."""

    def test_instantiation(self):
        policy = NoOptPolicy()
        assert policy is not None

    def test_has_on_app_switch(self):
        """Policy must have an on_app_switch method."""
        policy = NoOptPolicy()
        assert hasattr(policy, "on_app_switch")


class TestLRUPolicy:
    """Test LRU baseline policy."""

    def test_instantiation(self):
        policy = LRUPolicy()
        assert policy is not None

    def test_has_on_app_switch(self):
        policy = LRUPolicy()
        assert hasattr(policy, "on_app_switch")


class TestStaticPreloadPolicy:
    """Test StaticPreload baseline policy."""

    def test_instantiation(self):
        policy = StaticPreloadPolicy()
        assert policy is not None

    def test_has_on_app_switch(self):
        policy = StaticPreloadPolicy()
        assert hasattr(policy, "on_app_switch")


class TestProposedPolicy:
    """Test the proposed ML-integrated policy."""

    def test_instantiation(self):
        policy = ProposedPolicy()
        assert policy is not None

    def test_has_controller(self):
        """Proposed policy should have a decision controller."""
        policy = ProposedPolicy()
        assert hasattr(policy, "controller")

    def test_controller_has_scorer(self):
        """Decision controller should have a cache scorer."""
        policy = ProposedPolicy()
        assert hasattr(policy.controller, "scorer")

    def test_controller_has_preload_engine(self):
        """Decision controller should have a preload engine."""
        policy = ProposedPolicy()
        assert hasattr(policy.controller, "preload_engine")

    def test_controller_has_thrashing_detector(self):
        """Decision controller should have a thrashing detector."""
        policy = ProposedPolicy()
        assert hasattr(policy.controller, "thrashing_detector")
