"""
Simulation Engine — replays app-switch events through a policy and collects metrics.
This is the core benchmarking component.
"""
import time
from datetime import datetime
from typing import Optional, List

import pandas as pd

from src.models.memory import MemoryState, ActionType
from src.models.prediction import PredictionResult
from src.models.benchmark import BenchmarkConfig, BenchmarkResult
from src.simulation.policies import BasePolicy, get_policy
from src.data.app_profiles import get_profile


class SimulationEngine:
    """Discrete-event simulator that replays sessions through a memory policy."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(
        self,
        events_df: pd.DataFrame,
        policy: BasePolicy,
        predictor=None,
        ram_total_mb: float = 6144,
        config: Optional[BenchmarkConfig] = None,
    ) -> BenchmarkResult:
        """
        Run a simulation: replay events through a policy and collect metrics.

        Args:
            events_df: DataFrame of app-switch events
            policy: memory management policy to use
            predictor: XGBoostPredictor instance (None for non-proposed policies)
            ram_total_mb: total device RAM
            config: benchmark config metadata
        """
        if config is None:
            config = BenchmarkConfig(
                policy=policy.name,
                persona="mixed",
                ram_mb=ram_total_mb,
                n_events=len(events_df),
            )

        start_time = time.time()
        policy.reset()

        # Initialize memory state
        mem = MemoryState(total_ram_mb=ram_total_mb)

        # Tracking counters
        cache_hits = 0
        cache_misses = 0
        total_latency = 0.0
        eviction_count = 0
        preload_count = 0
        preload_used = set()      # preloaded apps that were actually accessed
        preloaded_apps = set()    # currently preloaded apps
        thrashing_events = 0
        safe_mode_activations = 0
        prediction_correct_top1 = 0
        prediction_correct_top3 = 0
        prediction_total = 0
        mem_util_sum = 0.0

        for idx, row in events_df.iterrows():
            event = row.to_dict()
            current_app = event["current_app"]
            actual_next = event.get("next_app", "")
            now = datetime.fromisoformat(event["timestamp"])

            # ── Check cache hit/miss ──
            profile = get_profile(current_app)
            if mem.is_app_cached(current_app):
                cache_hits += 1
                total_latency += profile.warm_launch_ms
                # Track if this was a preloaded app
                if current_app in preloaded_apps:
                    preload_used.add(current_app)
                    preloaded_apps.discard(current_app)
            else:
                cache_misses += 1
                total_latency += profile.cold_launch_ms

            # ── Get prediction (for proposed policy) ──
            prediction = None
            if predictor is not None and policy.name == "proposed":
                try:
                    prediction = predictor.predict(event)
                    prediction_total += 1
                    if prediction.predicted_app == actual_next:
                        prediction_correct_top1 += 1
                    if actual_next in prediction.top3_apps:
                        prediction_correct_top3 += 1
                except Exception:
                    prediction = PredictionResult(
                        predicted_app=current_app,
                        confidence=0.0,
                        top_k=[(current_app, 0.5)],
                        model_used="fallback",
                    )

            # ── Execute policy ──
            plan = policy.on_app_switch(event, mem, prediction, now)

            # ── Collect action stats ──
            for action in plan.actions:
                if action.action_type == ActionType.EVICT:
                    eviction_count += 1
                elif action.action_type == ActionType.PRELOAD:
                    preload_count += 1
                    preloaded_apps.add(action.target_app)

            if hasattr(plan, 'thrashing_score') and plan.thrashing_score > 0.7:
                thrashing_events += 1

            mem_util_sum += mem.ram_utilization

            if self.verbose and idx % 2000 == 0:
                hit_rate = cache_hits / max(cache_hits + cache_misses, 1)
                print(f"    [{policy.name}] Event {idx}: "
                      f"hits={cache_hits}, misses={cache_misses}, "
                      f"rate={hit_rate:.1%}, "
                      f"free_ram={mem.free_ram_mb:.0f}MB, "
                      f"cached={len(mem.cached_apps)} apps")

        # ── Compile results ──
        total_events = cache_hits + cache_misses
        elapsed = time.time() - start_time

        # Count safe mode activations from proposed policy
        if hasattr(policy, 'controller') and hasattr(policy.controller, 'thrashing_detector'):
            safe_mode_activations = policy.controller.thrashing_detector.total_safe_mode_activations

        result = BenchmarkResult(
            config=config,
            cache_hit_rate=cache_hits / max(total_events, 1),
            avg_launch_latency_ms=total_latency / max(total_events, 1),
            total_cache_hits=cache_hits,
            total_cache_misses=cache_misses,
            total_events=total_events,
            eviction_count=eviction_count,
            preload_count=preload_count,
            preload_success=len(preload_used),
            preload_success_rate=len(preload_used) / max(preload_count, 1),
            thrashing_events=thrashing_events,
            safe_mode_activations=safe_mode_activations,
            prediction_accuracy_top1=prediction_correct_top1 / max(prediction_total, 1),
            prediction_accuracy_top3=prediction_correct_top3 / max(prediction_total, 1),
            avg_memory_utilization=mem_util_sum / max(total_events, 1),
            run_duration_sec=elapsed,
        )

        return result
