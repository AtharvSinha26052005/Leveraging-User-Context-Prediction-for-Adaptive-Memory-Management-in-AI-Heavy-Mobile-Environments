"""
Preload Decision Engine — 5-condition safety gate for preloading predicted apps.
All 5 conditions must pass before preloading is allowed.
"""
from src.models.memory import MemoryState
from src.models.prediction import PredictionResult


class PreloadDecisionEngine:
    """
    5-condition preload gate:
    1. ML confidence >= threshold
    2. Free RAM >= headroom_mb
    3. Battery >= min_battery
    4. Thrashing score < thrashing_limit
    5. App is NOT already cached
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        ram_headroom_mb: float = 500.0,
        min_battery: float = 0.20,
        thrashing_limit: float = 0.70,
    ):
        self.confidence_threshold = confidence_threshold
        self.ram_headroom_mb = ram_headroom_mb
        self.min_battery = min_battery
        self.thrashing_limit = thrashing_limit

        # Track preload outcomes
        self.total_preloads = 0
        self.successful_preloads = 0

    def should_preload(
        self,
        prediction: PredictionResult,
        memory_state: MemoryState,
        battery: float,
        thrashing_score: float,
    ) -> tuple[bool, str]:
        """
        Check all 5 conditions. Returns (should_preload, reason).
        All conditions must pass.
        """
        app = prediction.predicted_app

        # Gate 1: Confidence
        if prediction.confidence < self.confidence_threshold:
            return False, f"confidence {prediction.confidence:.2f} < {self.confidence_threshold}"

        # Gate 2: RAM headroom
        if memory_state.free_ram_mb < self.ram_headroom_mb:
            return False, f"free RAM {memory_state.free_ram_mb:.0f} MB < {self.ram_headroom_mb:.0f} MB"

        # Gate 3: Battery
        if battery < self.min_battery:
            return False, f"battery {battery:.0%} < {self.min_battery:.0%}"

        # Gate 4: Not thrashing
        if thrashing_score >= self.thrashing_limit:
            return False, f"thrashing score {thrashing_score:.2f} >= {self.thrashing_limit}"

        # Gate 5: Not already cached
        if memory_state.is_app_cached(app):
            return False, f"{app} already in cache"

        return True, "all 5 gates passed"

    def record_outcome(self, was_used: bool) -> None:
        """Record whether a preloaded app was actually used."""
        self.total_preloads += 1
        if was_used:
            self.successful_preloads += 1

    @property
    def success_rate(self) -> float:
        if self.total_preloads == 0:
            return 0.0
        return self.successful_preloads / self.total_preloads

    def reset_stats(self) -> None:
        self.total_preloads = 0
        self.successful_preloads = 0
