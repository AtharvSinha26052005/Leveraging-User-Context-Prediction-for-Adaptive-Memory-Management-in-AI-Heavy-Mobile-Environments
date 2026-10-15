"""
Data models for context, app profiles, and snapshots.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class AppProfile:
    """Static profile of an application — defines its resource footprint and characteristics."""
    app_name: str
    category: str                 # social, messaging, productivity, entertainment, etc.
    memory_mb: float              # RAM used when running (e.g., 150 MB)
    cache_mb: float               # Additional cache/data in RAM (e.g., 50 MB)
    cold_launch_ms: int           # Cold start time in ms (e.g., 2000)
    warm_launch_ms: int           # Warm start time in ms (e.g., 120)
    importance: float             # 0.0 – 1.0 priority score (communication=0.9, games=0.3)
    is_ai_workload: bool = False  # True for camera AI, LLMs, translation apps
    ai_cache_mb: float = 0.0     # Extra memory for AI model weights

    @property
    def total_memory_mb(self) -> float:
        return self.memory_mb + self.cache_mb + self.ai_cache_mb


@dataclass
class ContextSnapshot:
    """
    Captures the full device context at the moment of an app-switch event.
    This is the input to the prediction layer.
    """
    current_app: str
    prev_apps: List[str]          # last 3 apps (most recent first)
    hour: int                     # 0-23
    day_of_week: int              # 0=Mon, 6=Sun
    battery: float                # 0.0 – 1.0
    is_charging: bool
    ram_free_mb: float
    ram_total_mb: float
    network_type: str             # wifi, cellular, none
    ai_task_active: bool
    app_category: str             # category of current_app
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    persona: Optional[str] = None

    @property
    def ram_utilization(self) -> float:
        """RAM usage as a fraction 0.0 – 1.0."""
        if self.ram_total_mb <= 0:
            return 1.0
        return 1.0 - (self.ram_free_mb / self.ram_total_mb)

    @property
    def is_peak_hour(self) -> bool:
        """True during typical high-usage hours."""
        return self.hour in (8, 9, 12, 13, 18, 19, 20, 21)

    @property
    def is_weekend(self) -> bool:
        return self.day_of_week >= 5

    def to_dict(self) -> dict:
        return {
            "current_app": self.current_app,
            "prev_app_1": self.prev_apps[0] if len(self.prev_apps) > 0 else "",
            "prev_app_2": self.prev_apps[1] if len(self.prev_apps) > 1 else "",
            "prev_app_3": self.prev_apps[2] if len(self.prev_apps) > 2 else "",
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "battery": self.battery,
            "is_charging": self.is_charging,
            "ram_free_mb": self.ram_free_mb,
            "ram_total_mb": self.ram_total_mb,
            "network_type": self.network_type,
            "ai_task_active": self.ai_task_active,
            "app_category": self.app_category,
            "ram_utilization": self.ram_utilization,
            "is_peak_hour": self.is_peak_hour,
            "is_weekend": self.is_weekend,
            "timestamp": self.timestamp.isoformat(),
        }
