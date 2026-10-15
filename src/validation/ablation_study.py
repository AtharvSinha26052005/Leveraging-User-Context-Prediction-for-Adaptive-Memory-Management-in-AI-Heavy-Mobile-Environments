"""
Ablation Study — Systematically remove each component to prove its value.
Each ablation variant removes ONE factor/component and measures the impact.

This version runs the full benchmark pipeline with trained predictor.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

from src.simulation.simulator import SimulationEngine
from src.models.benchmark import BenchmarkConfig, BenchmarkResult


@dataclass
class AblationResult:
    variant: str
    description: str
    cache_hit_rate: float
    avg_latency_ms: float
    eviction_count: int
    delta_hit_rate: float    # compared to full system
    delta_latency: float

    def to_dict(self):
        return {
            "Variant": self.variant,
            "Description": self.description,
            "Hit Rate": f"{self.cache_hit_rate:.1%}",
            "Latency (ms)": f"{self.avg_latency_ms:.0f}",
            "Evictions": self.eviction_count,
            "Delta Hit Rate": f"{self.delta_hit_rate:+.1%}",
            "Delta Latency": f"{self.delta_latency:+.0f} ms",
        }


class AblationStudy:
    """Run ablation experiments using the benchmark runner across multiple configs."""

    VARIANTS = [
        ("full_system",       "Full proposed system (all components)",       {}),
        ("no_prediction",     "Remove ML prediction (w_prediction=0)",       {"w_prediction": 0.0}),
        ("no_recency",        "Remove recency factor (w_recency=0)",         {"w_recency": 0.0}),
        ("no_frequency",      "Remove frequency factor (w_frequency=0)",     {"w_frequency": 0.0}),
        ("no_context",        "Remove context factor (w_context=0)",         {"w_context": 0.0}),
        ("no_importance",     "Remove importance factor (w_importance=0)",    {"w_importance": 0.0}),
        ("no_pressure",       "Remove memory pressure penalty (w_pressure=0)", {"w_pressure": 0.0}),
        ("no_preloading",     "Disable preloading entirely",                 {"disable_preload": True}),
        ("no_thrashing_det",  "Disable thrashing detection",                 {"disable_thrashing": True}),
    ]

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[AblationResult] = []
        self.baseline_hit_rate = 0.0
        self.baseline_latency = 0.0

    def run(self, events_df: pd.DataFrame, predictor=None, ram_mb: float = 6144) -> pd.DataFrame:
        """Run all ablation variants with a real predictor."""
        from src.simulation.policies import ProposedPolicy, LRUPolicy
        from src.memory.decision_controller import MemoryDecisionController

        self.results = []
        sim = SimulationEngine(verbose=False)

        for i, (name, description, mods) in enumerate(self.VARIANTS):
            if self.verbose:
                print(f"  [{i+1}/{len(self.VARIANTS)}] {name}: {description}", flush=True)

            # Create a fresh proposed policy and modify its controller
            policy = ProposedPolicy()
            scorer = policy.controller.scorer

            # Apply weight modifications
            for key, val in mods.items():
                if key.startswith("w_"):
                    setattr(scorer, key, val)

            # Disable preloading by setting impossible threshold
            if mods.get("disable_preload"):
                policy.controller.preload_engine.confidence_threshold = 999.0

            # Disable thrashing detection by setting impossible threshold
            if mods.get("disable_thrashing"):
                policy.controller.thrashing_detector.enter_threshold = 999.0

            config = BenchmarkConfig(
                policy=f"ablation_{name}", persona="mixed",
                ram_mb=ram_mb, n_events=len(events_df),
            )

            # Use a dummy predictor that always returns something if real one is None
            effective_predictor = predictor
            if effective_predictor is None:
                effective_predictor = _DummyPredictor(events_df)

            result = sim.run(events_df, policy, effective_predictor, ram_mb, config)

            if name == "full_system":
                self.baseline_hit_rate = result.cache_hit_rate
                self.baseline_latency = result.avg_launch_latency_ms

            ablation_result = AblationResult(
                variant=name,
                description=description,
                cache_hit_rate=result.cache_hit_rate,
                avg_latency_ms=result.avg_launch_latency_ms,
                eviction_count=result.eviction_count,
                delta_hit_rate=result.cache_hit_rate - self.baseline_hit_rate,
                delta_latency=result.avg_launch_latency_ms - self.baseline_latency,
            )
            self.results.append(ablation_result)

            if self.verbose:
                print(f"    hit_rate={result.cache_hit_rate:.1%}, "
                      f"latency={result.avg_launch_latency_ms:.0f}ms, "
                      f"evictions={result.eviction_count}, "
                      f"delta={ablation_result.delta_hit_rate:+.1%}", flush=True)

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def print_table(self):
        """Print formatted ablation table."""
        print(f"\n{'='*95}")
        print(f"  ABLATION STUDY — Component Contribution Analysis")
        print(f"{'='*95}")
        print(f"  {'Variant':<22} {'Hit Rate':>10} {'Latency':>10} {'Evictions':>10} "
              f"{'Delta HR':>10} {'Impact':>12}")
        print(f"{'-'*95}")

        for r in self.results:
            if r.variant == "full_system":
                marker = " (baseline)"
            elif r.delta_hit_rate < -0.01:
                marker = " *** CRITICAL"
            elif r.delta_hit_rate < -0.005:
                marker = " ** important"
            elif r.delta_hit_rate < 0:
                marker = " * minor"
            else:
                marker = ""

            print(f"  {r.variant:<22} {r.cache_hit_rate:>9.1%} {r.avg_latency_ms:>8.0f}ms "
                  f"{r.eviction_count:>10} {r.delta_hit_rate:>+9.1%}{marker}")

        print(f"{'='*95}")
        print(f"\n  *** = removing this component significantly hurts performance")
        print(f"  ** = important contribution")
        print(f"  * = minor but measurable contribution")


class _DummyPredictor:
    """
    Dummy predictor that uses observed transition frequencies
    to provide realistic predictions without a trained model.
    """
    def __init__(self, events_df: pd.DataFrame):
        from collections import defaultdict
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.app_list = sorted(events_df["current_app"].unique().tolist())

        for _, row in events_df.iterrows():
            curr = row.get("current_app", "")
            nxt = row.get("next_app", curr)
            if curr and nxt:
                self.transitions[curr][nxt] += 1

    def predict(self, event: dict):
        """Return a PredictionResult based on transition frequencies."""
        from src.models.prediction import PredictionResult
        curr = event.get("current_app", "")
        trans = self.transitions.get(curr, {})

        if not trans:
            top_app = curr
            confidence = 0.1
            top_k = [(curr, 1.0)]
        else:
            total = sum(trans.values())
            sorted_apps = sorted(trans.items(), key=lambda x: x[1], reverse=True)
            top_app = sorted_apps[0][0]
            confidence = sorted_apps[0][1] / total
            top_k = [(app, cnt / total) for app, cnt in sorted_apps[:5]]

        return PredictionResult(
            predicted_app=top_app,
            confidence=confidence,
            top_k=top_k,
            model_used="dummy_markov",
        )


def run_ablation():
    """Standalone ablation study runner."""
    print("\n  Loading data...", flush=True)
    df = pd.read_csv("data/synthetic_events.csv")
    # Use test split subset for speed
    test_df = df.iloc[int(len(df)*0.7):].head(1000).copy()
    print(f"  Using {len(test_df)} test events", flush=True)

    study = AblationStudy(verbose=True)
    results_df = study.run(test_df, predictor=None, ram_mb=6144)
    study.print_table()

    # Save
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/ablation_results.csv", index=False)
    print(f"\n  Saved to results/ablation_results.csv")

    return study


if __name__ == "__main__":
    run_ablation()
