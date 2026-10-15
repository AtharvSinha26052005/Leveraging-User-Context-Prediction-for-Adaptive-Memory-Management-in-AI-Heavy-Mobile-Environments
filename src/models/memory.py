"""
Data models for memory state, cache entries, and action plans.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from src.models.context import AppProfile


class ActionType(Enum):
    EVICT = "evict"
    PRELOAD = "preload"
    RETAIN = "retain"


class SystemMode(Enum):
    NORMAL = "normal"
    SAFE = "safe"


@dataclass
class AppCacheEntry:
    """Represents an app currently residing in RAM cache."""
    app_name: str
    memory_mb: float
    last_accessed: datetime
    access_count: int = 1
    category: str = ""
    importance: float = 0.5
    is_ai_workload: bool = False
    cold_launch_ms: int = 1500
    warm_launch_ms: int = 120

    def time_since_access(self, now: datetime) -> float:
        """Seconds since last access."""
        return (now - self.last_accessed).total_seconds()


@dataclass
class MemoryAction:
    """A single memory action to take."""
    action_type: ActionType
    target_app: str
    reason: str
    score: float = 0.0          # cache priority score at time of action
    memory_freed_mb: float = 0.0


@dataclass
class MemoryActionPlan:
    """Complete plan of actions for a single app-switch event."""
    actions: List[MemoryAction] = field(default_factory=list)
    mode: SystemMode = SystemMode.NORMAL
    thrashing_score: float = 0.0

    @property
    def evictions(self) -> List[MemoryAction]:
        return [a for a in self.actions if a.action_type == ActionType.EVICT]

    @property
    def preloads(self) -> List[MemoryAction]:
        return [a for a in self.actions if a.action_type == ActionType.PRELOAD]


@dataclass
class MemoryState:
    """Current state of device RAM — what's cached, how much is free."""
    total_ram_mb: float
    system_reserved_mb: float = 1500.0   # OS + system services
    cached_apps: Dict[str, AppCacheEntry] = field(default_factory=dict)
    foreground_app: Optional[str] = None

    @property
    def used_by_apps_mb(self) -> float:
        return sum(e.memory_mb for e in self.cached_apps.values())

    @property
    def free_ram_mb(self) -> float:
        return self.total_ram_mb - self.system_reserved_mb - self.used_by_apps_mb

    @property
    def ram_utilization(self) -> float:
        used = self.system_reserved_mb + self.used_by_apps_mb
        return used / self.total_ram_mb if self.total_ram_mb > 0 else 1.0

    def is_app_cached(self, app_name: str) -> bool:
        return app_name in self.cached_apps

    def add_to_cache(self, profile: AppProfile, now: datetime) -> None:
        """Add or update an app in the cache."""
        if profile.app_name in self.cached_apps:
            entry = self.cached_apps[profile.app_name]
            entry.last_accessed = now
            entry.access_count += 1
        else:
            self.cached_apps[profile.app_name] = AppCacheEntry(
                app_name=profile.app_name,
                memory_mb=profile.total_memory_mb,
                last_accessed=now,
                access_count=1,
                category=profile.category,
                importance=profile.importance,
                is_ai_workload=profile.is_ai_workload,
                cold_launch_ms=profile.cold_launch_ms,
                warm_launch_ms=profile.warm_launch_ms,
            )

    def remove_from_cache(self, app_name: str) -> Optional[AppCacheEntry]:
        """Remove an app from cache. Returns the removed entry or None."""
        return self.cached_apps.pop(app_name, None)

    def get_cache_list(self) -> List[AppCacheEntry]:
        """All cached apps sorted by last access (oldest first)."""
        return sorted(self.cached_apps.values(), key=lambda e: e.last_accessed)
