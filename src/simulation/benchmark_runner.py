"""
Benchmark Runner — runs the simulation engine across multiple configurations
and compiles comparative results.
"""
import time
from typing import List, Optional

import pandas as pd

from src.models.benchmark import BenchmarkConfig, BenchmarkResult
from src.simulation.simulator import SimulationEngine
from src.simulation.policies import get_policy


class BenchmarkRunner:
    """Orchestrates benchmark runs across policies, personas, and RAM configs."""

    POLICIES = ["noopt", "lru", "static_preload", "proposed"]
    PERSONAS = ["student", "professional", "casual", "power_user", "developer"]
    RAM_CONFIGS = [4096, 6144, 8192]

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def run_single(
        self,
        events_df: pd.DataFrame,
        policy_name: str,
        ram_mb: float,
        predictor=None,
        persona: str = "mixed",
        run_id: int = 0,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        config = BenchmarkConfig(
            policy=policy_name,
            persona=persona,
            ram_mb=ram_mb,
            n_events=len(events_df),
            run_id=run_id,
        )

        policy = get_policy(policy_name)
        sim = SimulationEngine(verbose=self.verbose)
        result = sim.run(events_df, policy, predictor, ram_mb, config)
        self.results.append(result)
        return result

    def run_all_policies(
        self,
        events_by_persona: dict,
        predictor=None,
        ram_configs: Optional[List[float]] = None,
        n_runs: int = 1,
    ) -> pd.DataFrame:
        """
        Run all policies × personas × RAM configs.
        
        Args:
            events_by_persona: dict mapping persona name to DataFrame of events
            predictor: trained XGBoostPredictor (used by proposed policy)
            ram_configs: list of RAM sizes in MB
            n_runs: number of repetitions per config
        """
        if ram_configs is None:
            ram_configs = self.RAM_CONFIGS

        self.results = []
        total_configs = len(self.POLICIES) * len(events_by_persona) * len(ram_configs) * n_runs
        done = 0
        start = time.time()

        for persona, events_df in events_by_persona.items():
            for ram_mb in ram_configs:
                for policy_name in self.POLICIES:
                    for run in range(n_runs):
                        done += 1
                        if self.verbose:
                            print(f"\n  [{done}/{total_configs}] "
                                  f"{policy_name} | {persona} | {int(ram_mb)}MB | run {run+1}")

                        result = self.run_single(
                            events_df, policy_name, ram_mb,
                            predictor=predictor,
                            persona=persona,
                            run_id=run,
                        )

                        if self.verbose:
                            print(f"    → hit_rate={result.cache_hit_rate:.1%}, "
                                  f"latency={result.avg_launch_latency_ms:.0f}ms, "
                                  f"evictions={result.eviction_count}, "
                                  f"thrashing={result.thrashing_events}")

        elapsed = time.time() - start
        if self.verbose:
            print(f"\n  Benchmark complete: {total_configs} runs in {elapsed:.1f}s")

        return self.to_dataframe()

    def run_quick_comparison(
        self,
        events_df: pd.DataFrame,
        predictor=None,
        ram_mb: float = 6144,
        persona: str = "mixed",
    ) -> pd.DataFrame:
        """Quick 4-policy comparison on a single config — good for demo."""
        self.results = []
        for policy_name in self.POLICIES:
            if self.verbose:
                print(f"\n  Running {policy_name}...")
            self.run_single(events_df, policy_name, ram_mb, predictor, persona)
            r = self.results[-1]
            if self.verbose:
                print(f"    → hit_rate={r.cache_hit_rate:.1%}, "
                      f"latency={r.avg_launch_latency_ms:.0f}ms")

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a DataFrame."""
        rows = [r.to_dict() for r in self.results]
        return pd.DataFrame(rows)

    def print_comparison_table(self, df: Optional[pd.DataFrame] = None) -> None:
        """Print a formatted comparison table to terminal."""
        if df is None:
            df = self.to_dataframe()

        # Aggregate by policy
        summary = df.groupby("policy").agg({
            "cache_hit_rate": "mean",
            "avg_launch_latency_ms": "mean",
            "eviction_count": "mean",
            "thrashing_events": "mean",
            "preload_count": "mean",
            "preload_success_rate": "mean",
        }).round(2)

        print("\n" + "=" * 85)
        print("  BENCHMARK RESULTS — Policy Comparison")
        print("=" * 85)
        print(f"  {'Policy':<18} {'Hit Rate':>10} {'Avg Latency':>12} "
              f"{'Evictions':>10} {'Thrashing':>10} {'Preloads':>10}")
        print("-" * 85)

        policy_order = ["noopt", "lru", "static_preload", "proposed"]
        for policy in policy_order:
            if policy in summary.index:
                row = summary.loc[policy]
                marker = " ◄" if policy == "proposed" else ""
                print(f"  {policy:<18} {row['cache_hit_rate']:>9.1%} "
                      f"{row['avg_launch_latency_ms']:>10.0f} ms "
                      f"{row['eviction_count']:>10.0f} "
                      f"{row['thrashing_events']:>10.0f} "
                      f"{row['preload_count']:>10.0f}{marker}")

        print("=" * 85)
