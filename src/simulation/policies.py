"""
Memory management policies — 4 strategies compared in benchmarks.
1. NoOptPolicy — FIFO, no intelligence
2. LRUPolicy — evict least recently used
3. StaticPreloadPolicy — LRU + preload top-3 frequent apps
4. ProposedPolicy — full Predict-Decide-Act pipeline
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict

from src.models.memory import MemoryState, AppCacheEntry, MemoryActionPlan, MemoryAction, ActionType, SystemMode
from src.models.prediction import PredictionResult
from src.models.context import AppProfile
from src.data.app_profiles import get_profile


class BasePolicy(ABC):
    """Abstract base for all memory policies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def on_app_switch(
        self,
        event: dict,
        memory_state: MemoryState,
        prediction: Optional[PredictionResult],
        now: datetime,
    ) -> MemoryActionPlan:
        """Process an app-switch event. Returns actions taken."""
        pass

    def reset(self) -> None:
        """Reset any internal state."""
        pass

    def _ensure_ram(self, memory_state: MemoryState, needed_mb: float,
                    evict_fn, now: datetime, foreground: str) -> List[MemoryAction]:
        """Generic helper: evict apps until we have enough free RAM."""
        actions = []
        while memory_state.free_ram_mb < needed_mb and len(memory_state.cached_apps) > 1:
            candidate = evict_fn(memory_state, foreground, now)
            if candidate is None:
                break
            entry = memory_state.remove_from_cache(candidate)
            if entry:
                actions.append(MemoryAction(
                    action_type=ActionType.EVICT,
                    target_app=candidate,
                    reason=f"{self.name}_eviction",
                    memory_freed_mb=entry.memory_mb,
                ))
        return actions


class NoOptPolicy(BasePolicy):
    """No optimization — FIFO eviction. Represents a phone with no cache management."""

    def __init__(self):
        super().__init__("noopt")

    def on_app_switch(self, event, memory_state, prediction, now):
        plan = MemoryActionPlan()
        app = event["current_app"]
        profile = get_profile(app)

        # Evict oldest (FIFO) if not enough RAM
        def fifo_evict(mem, fg, t):
            oldest = None
            oldest_time = None
            for name, entry in mem.cached_apps.items():
                if name == fg:
                    continue
                if oldest_time is None or entry.last_accessed < oldest_time:
                    oldest = name
                    oldest_time = entry.last_accessed
            return oldest

        plan.actions = self._ensure_ram(memory_state, profile.total_memory_mb + 200,
                                        fifo_evict, now, app)
        memory_state.add_to_cache(profile, now)
        memory_state.foreground_app = app
        return plan


class LRUPolicy(BasePolicy):
    """LRU eviction — standard Android approach. Evicts least recently used."""

    def __init__(self):
        super().__init__("lru")

    def on_app_switch(self, event, memory_state, prediction, now):
        plan = MemoryActionPlan()
        app = event["current_app"]
        profile = get_profile(app)

        def lru_evict(mem, fg, t):
            lru_app = None
            lru_time = None
            for name, entry in mem.cached_apps.items():
                if name == fg:
                    continue
                if lru_time is None or entry.last_accessed < lru_time:
                    lru_app = name
                    lru_time = entry.last_accessed
            return lru_app

        plan.actions = self._ensure_ram(memory_state, profile.total_memory_mb + 200,
                                        lru_evict, now, app)
        memory_state.add_to_cache(profile, now)
        memory_state.foreground_app = app
        return plan


class StaticPreloadPolicy(BasePolicy):
    """LRU + static preloading of top-3 most frequent apps."""

    def __init__(self):
        super().__init__("static_preload")
        self.app_frequency: Dict[str, int] = {}

    def on_app_switch(self, event, memory_state, prediction, now):
        plan = MemoryActionPlan()
        app = event["current_app"]
        profile = get_profile(app)

        # Track frequency
        self.app_frequency[app] = self.app_frequency.get(app, 0) + 1

        # LRU eviction
        def lru_evict(mem, fg, t):
            lru_app = None
            lru_time = None
            for name, entry in mem.cached_apps.items():
                if name == fg:
                    continue
                if lru_time is None or entry.last_accessed < lru_time:
                    lru_app = name
                    lru_time = entry.last_accessed
            return lru_app

        plan.actions = self._ensure_ram(memory_state, profile.total_memory_mb + 200,
                                        lru_evict, now, app)
        memory_state.add_to_cache(profile, now)
        memory_state.foreground_app = app

        # Preload top-3 frequent apps (if enough RAM)
        top3 = sorted(self.app_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
        for freq_app, _ in top3:
            if freq_app == app:
                continue
            if memory_state.is_app_cached(freq_app):
                continue
            freq_profile = get_profile(freq_app)
            if memory_state.free_ram_mb > freq_profile.total_memory_mb + 500:
                memory_state.add_to_cache(freq_profile, now)
                plan.actions.append(MemoryAction(
                    action_type=ActionType.PRELOAD,
                    target_app=freq_app,
                    reason="static_top3_preload",
                ))

        return plan

    def reset(self):
        self.app_frequency = {}


class ProposedPolicy(BasePolicy):
    """
    Our proposed system — full Predict-Decide-Act pipeline.
    Uses ML-integrated scoring, 5-gate preloading, and thrashing detection.
    """

    def __init__(self):
        super().__init__("proposed")
        from src.memory.decision_controller import MemoryDecisionController
        self.controller = MemoryDecisionController()

    def on_app_switch(self, event, memory_state, prediction, now):
        if prediction is None:
            # Fallback to LRU if no prediction available
            lru = LRUPolicy()
            return lru.on_app_switch(event, memory_state, prediction, now)

        return self.controller.decide(event, prediction, memory_state, now)

    def reset(self):
        self.controller.reset()


def get_policy(name: str) -> BasePolicy:
    """Factory function to get a policy by name."""
    policies = {
        "noopt": NoOptPolicy,
        "lru": LRUPolicy,
        "static_preload": StaticPreloadPolicy,
        "proposed": ProposedPolicy,
    }
    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Choose from {list(policies.keys())}")
    return policies[name]()
