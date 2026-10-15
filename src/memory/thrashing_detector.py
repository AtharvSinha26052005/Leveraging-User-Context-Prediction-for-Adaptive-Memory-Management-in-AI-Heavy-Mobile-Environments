"""
Thrashing Detector — monitors eviction patterns to detect memory thrashing.
When thrashing is detected, enters Safe Mode to stabilize the system.
"""
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from src.models.memory import SystemMode


@dataclass
class ThrashingEvent:
    """A single event in the thrashing detection window."""
    timestamp: datetime
    event_type: str      # "eviction", "reload", "cache_miss"
    app_name: str


class ThrashingDetector:
    """
    Sliding-window thrashing detector.

    Thrashing score = 0.4 × reload_rate + 0.35 × miss_rate + 0.25 × eviction_rate

    - Enter Safe Mode when score > enter_threshold (0.7)
    - Exit Safe Mode when score < exit_threshold (0.3) for stability_period seconds
    """

    def __init__(
        self,
        window_seconds: float = 120.0,
        enter_threshold: float = 0.70,
        exit_threshold: float = 0.30,
        stability_period: float = 60.0,
    ):
        self.window_seconds = window_seconds
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.stability_period = stability_period

        self.events: deque = deque()
        self.mode: SystemMode = SystemMode.NORMAL
        self.safe_mode_start: Optional[datetime] = None
        self.below_threshold_since: Optional[datetime] = None

        # Stats
        self.total_safe_mode_activations = 0
        self.recent_evicted_apps: deque = deque(maxlen=50)  # track recently evicted apps

    def record_eviction(self, app_name: str, now: datetime) -> None:
        """Record that an app was evicted from cache."""
        self.events.append(ThrashingEvent(now, "eviction", app_name))
        self.recent_evicted_apps.append((app_name, now))
        self._prune_old(now)
        self._update_mode(now)

    def record_cache_miss(self, app_name: str, now: datetime) -> None:
        """Record a cache miss (app not in cache when user opened it)."""
        self.events.append(ThrashingEvent(now, "cache_miss", app_name))
        self._prune_old(now)
        self._update_mode(now)

    def record_reload(self, app_name: str, now: datetime) -> None:
        """Record that a recently evicted app was reloaded (thrashing signal)."""
        # Check if this app was recently evicted
        for evicted_app, evicted_time in self.recent_evicted_apps:
            if evicted_app == app_name and (now - evicted_time).total_seconds() < 60:
                self.events.append(ThrashingEvent(now, "reload", app_name))
                break
        self._prune_old(now)
        self._update_mode(now)

    def get_score(self, now: datetime) -> float:
        """
        Compute the thrashing score for the current window.
        Score ranges from 0.0 (no thrashing) to ~1.0+ (severe thrashing).
        """
        self._prune_old(now)

        if not self.events:
            return 0.0

        total = len(self.events)
        reloads = sum(1 for e in self.events if e.event_type == "reload")
        misses = sum(1 for e in self.events if e.event_type == "cache_miss")
        evictions = sum(1 for e in self.events if e.event_type == "eviction")

        # Normalize rates
        reload_rate = reloads / max(evictions, 1)
        miss_rate = misses / max(total, 1)
        eviction_rate = min(evictions / 10.0, 1.0)  # normalize: 10+ evictions in window = 1.0

        score = 0.40 * reload_rate + 0.35 * miss_rate + 0.25 * eviction_rate
        return round(score, 4)

    @property
    def is_safe_mode(self) -> bool:
        return self.mode == SystemMode.SAFE

    def _prune_old(self, now: datetime) -> None:
        """Remove events outside the sliding window."""
        cutoff = now.timestamp() - self.window_seconds
        while self.events and self.events[0].timestamp.timestamp() < cutoff:
            self.events.popleft()

    def _update_mode(self, now: datetime) -> None:
        """Check if we should enter or exit Safe Mode."""
        score = self.get_score(now)

        if self.mode == SystemMode.NORMAL:
            if score >= self.enter_threshold:
                self.mode = SystemMode.SAFE
                self.safe_mode_start = now
                self.below_threshold_since = None
                self.total_safe_mode_activations += 1

        elif self.mode == SystemMode.SAFE:
            if score < self.exit_threshold:
                if self.below_threshold_since is None:
                    self.below_threshold_since = now
                elif (now - self.below_threshold_since).total_seconds() >= self.stability_period:
                    # Stable for long enough — exit Safe Mode
                    self.mode = SystemMode.NORMAL
                    self.safe_mode_start = None
                    self.below_threshold_since = None
            else:
                self.below_threshold_since = None

    def reset(self) -> None:
        """Force reset to Normal mode."""
        self.events.clear()
        self.mode = SystemMode.NORMAL
        self.safe_mode_start = None
        self.below_threshold_since = None
        self.recent_evicted_apps.clear()
