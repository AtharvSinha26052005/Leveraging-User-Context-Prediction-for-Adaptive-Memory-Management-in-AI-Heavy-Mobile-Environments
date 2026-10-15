"""
Data models for benchmark configuration and results.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    policy: str                   # "noopt", "lru", "static_preload", "proposed"
    persona: str                  # "student", "professional", etc.
    ram_mb: float                 # 4096, 6144, 8192
    n_events: int = 10000
    run_id: int = 0

    @property
    def label(self) -> str:
        return f"{self.policy}_{self.persona}_{int(self.ram_mb)}"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig

    # Core KPIs
    cache_hit_rate: float = 0.0          # 0.0 – 1.0
    avg_launch_latency_ms: float = 0.0   # mean ms
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_events: int = 0

    # Eviction & Preload
    eviction_count: int = 0
    preload_count: int = 0
    preload_success: int = 0             # preloads that were actually used
    preload_success_rate: float = 0.0

    # Thrashing
    thrashing_events: int = 0
    safe_mode_activations: int = 0

    # Prediction (for proposed policy only)
    prediction_accuracy_top1: float = 0.0
    prediction_accuracy_top3: float = 0.0

    # Timing
    avg_memory_utilization: float = 0.0
    run_duration_sec: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "policy": self.config.policy,
            "persona": self.config.persona,
            "ram_mb": self.config.ram_mb,
            "run_id": self.config.run_id,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "avg_launch_latency_ms": round(self.avg_launch_latency_ms, 1),
            "total_cache_hits": self.total_cache_hits,
            "total_cache_misses": self.total_cache_misses,
            "total_events": self.total_events,
            "eviction_count": self.eviction_count,
            "preload_count": self.preload_count,
            "preload_success": self.preload_success,
            "preload_success_rate": round(self.preload_success_rate, 4),
            "thrashing_events": self.thrashing_events,
            "safe_mode_activations": self.safe_mode_activations,
            "prediction_accuracy_top1": round(self.prediction_accuracy_top1, 4),
            "prediction_accuracy_top3": round(self.prediction_accuracy_top3, 4),
            "avg_memory_utilization": round(self.avg_memory_utilization, 4),
            "run_duration_sec": round(self.run_duration_sec, 2),
            "timestamp": self.timestamp.isoformat(),
        }
