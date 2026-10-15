"""
Memory Decision Controller — orchestrates scoring, preloading, and thrashing detection.
This is the "brain" that produces a MemoryActionPlan for each app-switch event.
"""
from datetime import datetime
from typing import Optional

from src.models.memory import (
    MemoryState, MemoryActionPlan, MemoryAction, ActionType, SystemMode
)
from src.models.prediction import PredictionResult
from src.models.context import AppProfile
from src.memory.cache_scorer import CacheScoringEngine
from src.memory.preload_gate import PreloadDecisionEngine
from src.memory.thrashing_detector import ThrashingDetector
from src.data.app_profiles import get_profile


class MemoryDecisionController:
    """
    Orchestrates the Predict-Decide-Act pipeline:
    1. Score all cached apps
    2. Decide evictions (if RAM is low)
    3. Check preload gate for predicted app
    4. Monitor for thrashing
    5. Produce a MemoryActionPlan
    """

    def __init__(
        self,
        target_free_ram_mb: float = 500.0,
        safe_mode_free_multiplier: float = 1.5,
    ):
        self.scorer = CacheScoringEngine()
        self.preload_engine = PreloadDecisionEngine()
        self.thrashing_detector = ThrashingDetector()

        self.target_free_ram_mb = target_free_ram_mb
        self.safe_mode_free_multiplier = safe_mode_free_multiplier

    def decide(
        self,
        event: dict,
        prediction: PredictionResult,
        memory_state: MemoryState,
        now: datetime,
    ) -> MemoryActionPlan:
        """
        Make all memory decisions for a single app-switch event.
        Returns a complete action plan.
        """
        plan = MemoryActionPlan()
        hour = event.get("hour", 12)
        battery = event.get("battery", 0.5)

        # Get current thrashing state
        thrashing_score = self.thrashing_detector.get_score(now)
        plan.thrashing_score = thrashing_score
        plan.mode = SystemMode.SAFE if self.thrashing_detector.is_safe_mode else SystemMode.NORMAL

        # Determine target free RAM (higher in Safe Mode)
        target_free = self.target_free_ram_mb
        if self.thrashing_detector.is_safe_mode:
            target_free *= self.safe_mode_free_multiplier

        # ── Step 1: Handle the current app (bring it to foreground) ──
        current_app = event.get("current_app", "")
        app_profile = get_profile(current_app)

        if not memory_state.is_app_cached(current_app):
            # Cache miss — need to load this app
            self.thrashing_detector.record_cache_miss(current_app, now)
            # Check if this was a recently evicted app (thrashing signal)
            self.thrashing_detector.record_reload(current_app, now)

        # Add/update current app in cache
        memory_state.add_to_cache(app_profile, now)
        memory_state.foreground_app = current_app

        # ── Step 2: Evict if RAM is low ──
        while memory_state.free_ram_mb < target_free and len(memory_state.cached_apps) > 1:
            ranked = self.scorer.rank_all(memory_state, prediction, now, hour)

            # Find the lowest-scored non-foreground app
            evict_candidate = None
            for app_name, score in ranked:
                if app_name != memory_state.foreground_app and score < 999:
                    evict_candidate = (app_name, score)
                    break

            if evict_candidate is None:
                break  # nothing to evict

            app_name, score = evict_candidate
            entry = memory_state.cached_apps[app_name]
            freed = entry.memory_mb

            memory_state.remove_from_cache(app_name)
            self.thrashing_detector.record_eviction(app_name, now)

            plan.actions.append(MemoryAction(
                action_type=ActionType.EVICT,
                target_app=app_name,
                reason=f"score={score:.3f}, freed={freed:.0f}MB",
                score=score,
                memory_freed_mb=freed,
            ))

        # ── Step 3: Check preloading (skip in Safe Mode) ──
        if not self.thrashing_detector.is_safe_mode:
            should_preload, reason = self.preload_engine.should_preload(
                prediction, memory_state, battery, thrashing_score
            )

            if should_preload:
                preload_app = prediction.predicted_app
                preload_profile = get_profile(preload_app)

                # Only preload if we still have enough RAM after
                if memory_state.free_ram_mb > preload_profile.total_memory_mb + 200:
                    memory_state.add_to_cache(preload_profile, now)
                    plan.actions.append(MemoryAction(
                        action_type=ActionType.PRELOAD,
                        target_app=preload_app,
                        reason=f"confidence={prediction.confidence:.2f}, {reason}",
                        score=prediction.confidence,
                    ))

        # Update thrashing mode in plan
        plan.mode = SystemMode.SAFE if self.thrashing_detector.is_safe_mode else SystemMode.NORMAL
        plan.thrashing_score = self.thrashing_detector.get_score(now)

        return plan

    def reset(self) -> None:
        """Reset all stateful components."""
        self.thrashing_detector.reset()
        self.preload_engine.reset_stats()
