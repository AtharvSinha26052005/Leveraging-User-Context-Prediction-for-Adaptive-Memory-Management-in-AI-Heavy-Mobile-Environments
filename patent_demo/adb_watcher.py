"""
adb_watcher.py — Non-blocking ADB foreground app poller.

Runs in a daemon thread at ADB_POLL_MS intervals.
- Filters IGNORED_PACKAGES (transient overlays / system UI).
- Supports Auto-Tour mode (cycles AUTO_TOUR_SEQUENCE every AUTO_TOUR_INTERVAL_S).
- Exposes thread-safe properties: foreground_app, battery, ram, adb_connected.
- Gracefully degrades when ADB is unavailable (--no-adb mode).
- Issue 4: get_recent_apps() + kill_background_app() for USB debugging sync.
"""
import re
import subprocess
import sys
import os
import threading
import time

# ── Path setup (handles both direct execution and package import) ─────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_CAPSTONE = os.path.dirname(_HERE)
if _CAPSTONE not in sys.path:
    sys.path.insert(0, _CAPSTONE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    ADB_POLL_MS,
    IGNORED_PACKAGES,
    AUTO_TOUR_INTERVAL_S,
    AUTO_TOUR_SEQUENCE,
)

# Import PACKAGE_MAP from existing collector — fall back to empty dict if unavailable
try:
    from scripts.collect_phone_data import PACKAGE_MAP
except Exception:
    PACKAGE_MAP = {
        "com.whatsapp":                          "WhatsApp",
        "com.instagram.android":                 "Instagram",
        "com.google.android.youtube":            "YouTube",
        "com.google.android.gm":                 "Gmail",
        "com.android.chrome":                    "Chrome",
        "com.spotify.music":                     "Spotify",
        "com.android.camera":                    "Camera",
        "com.oplus.camera":                      "Camera",
        "com.battlegroundsmobilegame":           "BGMI",
        "org.telegram.messenger":                "Telegram",
        "com.discord":                           "Discord",
        "com.phonepe.app":                       "PhonePe",
        "com.google.android.apps.maps":          "Maps",
        "com.zhiliaoapp.musically":              "TikTok",
        "com.openai.chatgpt":                    "ChatGPT",
        "com.twitter.android":                   "Twitter",
        "com.netflix.mediaclient":               "Netflix",
        "com.android.settings":                  "Settings",
        "com.coloros.settings":                  "Settings",
        "com.coloros.calculator":                "Calculator",
        "com.coloros.launcher":                  "Home Screen",
        "com.realme.launcher":                   "Home Screen",
        "com.android.launcher3":                 "Home Screen",
        "com.amazon.mShop.android.shopping":     "Amazon",
        "com.flipkart.android":                  "Flipkart",
        "com.android.dialer":                    "Phone",
        "com.coloros.dialer":                    "Phone",
        "com.snapchat.android":                  "Snapchat",
        "com.google.android.apps.photos":        "Photos",
        "com.google.android.calculator":         "Calculator",
        "com.google.android.keep":               "Keep Notes",
    }

# ── Reverse map: friendly name → package (for am kill) ───────────────────────
REVERSE_PACKAGE_MAP: dict[str, str] = {}
for _pkg, _name in PACKAGE_MAP.items():
    if _name not in REVERSE_PACKAGE_MAP:   # first occurrence wins
        REVERSE_PACKAGE_MAP[_name] = _pkg

# Ensure common entries are correct even if PACKAGE_MAP was overridden
_HARDCODED_REVERSE = {
    "WhatsApp":   "com.whatsapp",
    "Instagram":  "com.instagram.android",
    "YouTube":    "com.google.android.youtube",
    "Gmail":      "com.google.android.gm",
    "Chrome":     "com.android.chrome",
    "Spotify":    "com.spotify.music",
    "Camera":     "com.oplus.camera",
    "BGMI":       "com.battlegroundsmobilegame",
    "Telegram":   "org.telegram.messenger",
    "Discord":    "com.discord",
    "PhonePe":    "com.phonepe.app",
    "Maps":       "com.google.android.apps.maps",
    "ChatGPT":    "com.openai.chatgpt",
    "Twitter":    "com.twitter.android",
    "Netflix":    "com.netflix.mediaclient",
    "Settings":   "com.android.settings",
    "Calculator": "com.coloros.calculator",
    "Amazon":     "com.amazon.mShop.android.shopping",
    "Flipkart":   "com.flipkart.android",
    "Phone":      "com.coloros.dialer",
    "Snapchat":   "com.snapchat.android",
}
REVERSE_PACKAGE_MAP.update(_HARDCODED_REVERSE)


def _package_to_name(package: str) -> str:
    """Convert package name to friendly app name."""
    if package in PACKAGE_MAP:
        return PACKAGE_MAP[package]
    parts = package.split(".")
    return parts[-1].capitalize() if parts else package


class AdbWatcher:
    """
    Polls ADB for the foreground app, battery, and RAM every ADB_POLL_MS ms.
    Thread-safe via a single threading.Lock.
    When adb_path is None, operates in demo mode (hotkey / auto-tour only).

    Issue 4 additions:
      - recent_apps property: ordered list of recent apps from phone Recents stack.
      - kill_background_app(name): fires am kill <pkg> in background daemon thread.
    """

    # How often to refresh recent-apps list (seconds)
    _RECENT_REFRESH_INTERVAL_S = 5.0

    def __init__(self, adb_path: str | None, poll_ms: int = ADB_POLL_MS) -> None:
        self._adb_path   = adb_path
        self._poll_ms    = poll_ms

        # Foreground app state
        self._current_app     = "Home Screen"
        self._current_package = ""

        # Telemetry defaults (used when ADB is unavailable)
        self._battery_pct  = 74
        self._is_charging  = False
        self._ram_total_mb = 8192.0
        self._ram_free_mb  = 3200.0

        # ADB health
        self._adb_ok       = (adb_path is not None)
        self._fail_streak  = 0

        # Auto-Tour state
        self._auto_tour    = False
        self._tour_idx     = 0
        self._last_tour_ts = time.monotonic()

        # Issue 4 — Recent apps list (ordered, phone Recents stack)
        self._recent_apps: list[str] = []
        self._last_recent_refresh    = 0.0   # monotonic timestamp

        # Thread safety
        self._lock = threading.Lock()

        # Start daemon poll thread (handles both ADB polling and auto-tour)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="AdbWatcher")
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def foreground_app(self) -> str:
        with self._lock:
            return self._current_app

    @property
    def battery(self) -> tuple[int, bool]:
        with self._lock:
            return self._battery_pct, self._is_charging

    @property
    def ram(self) -> tuple[float, float]:
        with self._lock:
            return self._ram_total_mb, self._ram_free_mb

    @property
    def adb_connected(self) -> bool:
        with self._lock:
            return self._adb_ok

    @property
    def recent_apps(self) -> list[str]:
        """Ordered list of recent apps from phone Recents stack (thread-safe)."""
        with self._lock:
            return list(self._recent_apps)

    def override_app(self, app_name: str) -> None:
        """Force-set the foreground app (called by HotkeyListener). Disables auto-tour."""
        with self._lock:
            self._current_app = app_name
            self._auto_tour   = False   # hotkey wins over auto-tour

    def toggle_auto_tour(self) -> None:
        """Toggle Auto-Tour mode on / off."""
        with self._lock:
            self._auto_tour    = not self._auto_tour
            self._tour_idx     = 0
            self._last_tour_ts = time.monotonic()

    # ── Issue 4: Recent apps & background kill ────────────────────────────────

    def get_recent_apps(self) -> list[str]:
        """
        Queries 'adb shell dumpsys activity recents' and returns an ordered
        list of up to 8 friendly app names. Returns [] if ADB unavailable.
        """
        if self._adb_path is None:
            return []
        try:
            raw = self._run_adb("shell", "dumpsys", "activity", "recents", timeout=3.0)
        except Exception:
            return []

        seen: list[str] = []
        seen_set: set[str] = set()

        # Match lines like:  baseIntent=... cmp=com.whatsapp/.Main
        # or:                realActivity=com.whatsapp/.Main
        pkg_re = re.compile(r'(?:baseIntent.*?cmp=|realActivity=)([a-zA-Z][a-zA-Z0-9_.]+)/[a-zA-Z]')

        for line in raw.splitlines():
            m = pkg_re.search(line)
            if not m:
                continue
            pkg = m.group(1)
            if pkg in IGNORED_PACKAGES:
                continue
            name = _package_to_name(pkg)
            if name == "Home Screen":
                continue
            if name not in seen_set:
                seen.append(name)
                seen_set.add(name)
            if len(seen) >= 8:
                break

        return seen

    def kill_background_app(self, app_name: str) -> None:
        """
        Fire-and-forget: sends 'adb shell am kill <package>' in a daemon thread
        to reclaim that app's cached background process from phone RAM.
        Safe — am kill only terminates background cached processes; it never
        affects the foreground app or crashes anything.
        """
        if not self._adb_ok or self._adb_path is None:
            return
        pkg = REVERSE_PACKAGE_MAP.get(app_name)
        if not pkg:
            return

        adb_path = self._adb_path   # capture before thread starts

        def _kill():
            try:
                subprocess.run(
                    [adb_path, "shell", "am", "kill", pkg],
                    capture_output=True, timeout=2,
                )
            except Exception:
                pass

        t = threading.Thread(target=_kill, daemon=True, name=f"AdbKill-{app_name}")
        t.start()

    # ── Daemon poll loop ──────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        # Fetch recent apps once at startup (before first render tick)
        if self._adb_path is not None:
            try:
                recents = self.get_recent_apps()
                with self._lock:
                    self._recent_apps = recents
                self._last_recent_refresh = time.monotonic()
            except Exception:
                pass

        while True:
            try:
                self._step()
            except Exception:
                pass
            time.sleep(self._poll_ms / 1000.0)

    def _step(self) -> None:
        now = time.monotonic()

        # ── Auto-Tour advance ─────────────────────────────────────────────────
        with self._lock:
            is_touring = self._auto_tour
            last_ts    = self._last_tour_ts
            idx        = self._tour_idx

        if is_touring and (now - last_ts) >= AUTO_TOUR_INTERVAL_S:
            next_app = AUTO_TOUR_SEQUENCE[idx % len(AUTO_TOUR_SEQUENCE)]
            with self._lock:
                self._current_app  = next_app
                self._tour_idx     = idx + 1
                self._last_tour_ts = now
            return   # skip ADB poll on tour-advance ticks

        # ── ADB polling (only if path provided) ───────────────────────────────
        if self._adb_path is None:
            return

        try:
            pkg = self._get_foreground()
            if pkg is not None:
                with self._lock:
                    self._current_app     = pkg
                    self._current_package = pkg

            batt_pct, is_charging = self._get_battery()
            with self._lock:
                self._battery_pct = batt_pct
                self._is_charging = is_charging

            total_mb, free_mb = self._get_ram()
            with self._lock:
                self._ram_total_mb = total_mb
                self._ram_free_mb  = free_mb

            with self._lock:
                self._adb_ok     = True
                self._fail_streak = 0

            # Refresh recent apps every _RECENT_REFRESH_INTERVAL_S seconds
            if now - self._last_recent_refresh >= self._RECENT_REFRESH_INTERVAL_S:
                recents = self.get_recent_apps()
                with self._lock:
                    self._recent_apps = recents
                self._last_recent_refresh = now

        except Exception:
            with self._lock:
                self._fail_streak += 1
                if self._fail_streak >= 5:
                    self._adb_ok = False

    # ── Private ADB helpers ───────────────────────────────────────────────────

    def _run_adb(self, *args: str, timeout: float = 5.0) -> str:
        """Run an ADB command and return stdout text. Raises on failure."""
        result = subprocess.run(
            [self._adb_path] + list(args),
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout

    def _get_foreground(self) -> str | None:
        """
        Returns the friendly app name for the current foreground package,
        or None if the package should be ignored / undetectable.
        """
        _PKG_RE = re.compile(r'([a-zA-Z][a-zA-Z0-9_.]+)/[a-zA-Z.]')

        # PRIMARY: dumpsys activity
        try:
            out = self._run_adb("shell", "dumpsys", "activity", "activities")
            for line in out.splitlines():
                if "ResumedActivity" in line or "topResumedActivity" in line:
                    m = _PKG_RE.search(line)
                    if m:
                        pkg = m.group(1)
                        if pkg in IGNORED_PACKAGES:
                            return None
                        return _package_to_name(pkg)
        except Exception:
            pass

        # FALLBACK: dumpsys window
        try:
            out = self._run_adb("shell", "dumpsys", "window", "displays")
            for line in out.splitlines():
                if "mCurrentFocus" in line or "mFocusedApp" in line:
                    m = _PKG_RE.search(line)
                    if m:
                        pkg = m.group(1)
                        if pkg in IGNORED_PACKAGES:
                            return None
                        return _package_to_name(pkg)
        except Exception:
            pass

        return None

    def _get_battery(self) -> tuple[int, bool]:
        """Parse battery level and charging state from ADB. Default (74, False) on error."""
        try:
            out = self._run_adb("shell", "dumpsys", "battery")
            level    = 74
            charging = False
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("level:"):
                    m = re.search(r'\d+', line)
                    if m:
                        level = int(m.group())
                elif line.startswith("status:"):
                    # status 2 = charging, 5 = full
                    charging = any(c in line for c in ("2", "5"))
            return level, charging
        except Exception:
            return 74, False

    def _get_ram(self) -> tuple[float, float]:
        """Parse MemTotal and MemAvailable from /proc/meminfo. Default (8192, 3200) on error."""
        try:
            out = self._run_adb("shell", "cat", "/proc/meminfo")
            total_kb = 8192 * 1024
            avail_kb = 3200 * 1024
            for line in out.splitlines():
                if line.startswith("MemTotal:"):
                    m = re.search(r'\d+', line)
                    if m:
                        total_kb = int(m.group())
                elif line.startswith("MemAvailable:"):
                    m = re.search(r'\d+', line)
                    if m:
                        avail_kb = int(m.group())
            return total_kb / 1024.0, avail_kb / 1024.0
        except Exception:
            return 8192.0, 3200.0
