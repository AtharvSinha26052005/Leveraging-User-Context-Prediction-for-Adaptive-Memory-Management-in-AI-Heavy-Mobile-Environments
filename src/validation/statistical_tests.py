"""
Statistical Significance Tests — Prove our results are not due to random chance.
Paired t-test, Wilcoxon signed-rank, and Cohen's d effect size.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List


@dataclass
class StatTestResult:
    comparison: str
    metric: str
    mean_proposed: float
    mean_baseline: float
    improvement: float
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool

    def to_dict(self):
        return {
            "Comparison": self.comparison,
            "Metric": self.metric,
            "Proposed": f"{self.mean_proposed:.3f}",
            "Baseline": f"{self.mean_baseline:.3f}",
            "Improvement": f"{self.improvement:+.1%}",
            "t-statistic": f"{self.t_statistic:.3f}",
            "p-value": f"{self.p_value:.6f}",
            "Cohen's d": f"{self.cohens_d:.3f}",
            "Significant": "YES ***" if self.p_value < 0.001 else
                          "YES **" if self.p_value < 0.01 else
                          "YES *" if self.p_value < 0.05 else "NO",
        }


class StatisticalTests:
    """Run statistical significance tests on benchmark results."""

    def __init__(self):
        self.results: List[StatTestResult] = []

    def run_all_tests(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Run all statistical tests comparing proposed vs each baseline."""
        self.results = []
        baselines = ["noopt", "lru", "static_preload"]
        metrics = [
            ("cache_hit_rate", "Cache Hit Rate"),
            ("avg_launch_latency_ms", "Avg Latency (ms)"),
            ("eviction_count", "Eviction Count"),
        ]

        proposed_df = results_df[results_df["policy"] == "proposed"]

        for baseline_name in baselines:
            baseline_df = results_df[results_df["policy"] == baseline_name]

            for metric_col, metric_name in metrics:
                # Align data by persona+ram config
                proposed_vals = []
                baseline_vals = []

                for _, p_row in proposed_df.iterrows():
                    matching = baseline_df[
                        (baseline_df["persona"] == p_row["persona"]) &
                        (baseline_df["ram_mb"] == p_row["ram_mb"])
                    ]
                    if not matching.empty:
                        proposed_vals.append(p_row[metric_col])
                        baseline_vals.append(matching.iloc[0][metric_col])

                if len(proposed_vals) < 2:
                    continue

                proposed_arr = np.array(proposed_vals)
                baseline_arr = np.array(baseline_vals)

                # Paired t-test
                t_stat, p_val = stats.ttest_rel(proposed_arr, baseline_arr)

                # Cohen's d (paired)
                diff = proposed_arr - baseline_arr
                d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

                # For latency and evictions, lower is better
                improvement = (np.mean(proposed_arr) - np.mean(baseline_arr)) / (np.mean(baseline_arr) + 1e-10)
                if metric_col in ["avg_launch_latency_ms", "eviction_count"]:
                    improvement = -improvement  # invert so positive = better

                self.results.append(StatTestResult(
                    comparison=f"proposed vs {baseline_name}",
                    metric=metric_name,
                    mean_proposed=np.mean(proposed_arr),
                    mean_baseline=np.mean(baseline_arr),
                    improvement=improvement,
                    t_statistic=t_stat,
                    p_value=p_val,
                    cohens_d=abs(d),
                    significant=p_val < 0.05,
                ))

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def print_table(self):
        """Print formatted significance table."""
        print("\n" + "=" * 110)
        print("  STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 110)
        print(f"  {'Comparison':<28} {'Metric':<20} {'Proposed':>10} {'Baseline':>10} "
              f"{'Improve':>10} {'p-value':>12} {'Cohen d':>9} {'Sig?':>8}")
        print("-" * 110)

        for r in self.results:
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
            print(f"  {r.comparison:<28} {r.metric:<20} {r.mean_proposed:>10.3f} {r.mean_baseline:>10.3f} "
                  f"{r.improvement:>+9.1%} {r.p_value:>12.6f} {r.cohens_d:>9.3f} {sig:>8}")

        print("=" * 110)
        print("  *** p < 0.001 (highly significant)")
        print("  **  p < 0.01  (very significant)")
        print("  *   p < 0.05  (significant)")
        print(f"  Cohen's d: small > 0.2, medium > 0.5, large > 0.8")


def run_statistical_tests():
    """Standalone runner."""
    print("\n  Loading benchmark results...")
    df = pd.read_csv("results/benchmark_results.csv")
    print(f"  Loaded {len(df)} benchmark rows")

    tests = StatisticalTests()
    results_df = tests.run_all_tests(df)
    tests.print_table()

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/statistical_tests.csv", index=False)
    print(f"\n  Saved to results/statistical_tests.csv")

    return tests


if __name__ == "__main__":
    run_statistical_tests()
