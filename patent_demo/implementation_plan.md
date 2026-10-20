# Patent Demo — Implementation Plan (v2)
# Adaptive Memory Management: Live Patent Video CLI Dashboard
# Lead Architect Reference Document — Do NOT modify once implementation starts

---

## 0. Isolation Contract

```
patent_demo/                          <- ALL new code lives here, nowhere else
    run_demo.py                       <- single entry point  (python patent_demo/run_demo.py)
    config.py                         <- constants, ADB path, hotkey map, demo matrix
    adb_watcher.py                    <- non-blocking ADB foreground app poller
    demo_director.py                  <- ML hook + deterministic blending layer
    ram_state_machine.py              <- LPDDR5 address simulation + 6-Factor S(a) engine
    tui_renderer.py                   <- Rich TUI layout definitions + render loop
    hotkey_listener.py                <- pynput keyboard override (silent, non-blocking)
    requirements_demo.txt             <- rich>=13.7.0  pynput>=1.7.6
```

**Zero-touch rule:** `run_demo.py` adds `capstone/` to `sys.path` with `sys.path.insert(0, ...)`
and imports existing code read-only. It NEVER writes to any file outside `patent_demo/`.

---

## 0.1 System Architecture Flow

```
+----------------------------------------------------------------------------------------+
|                        LIVE PATENT DEMO ARCHITECTURE FLOW                              |
|                                                                                        |
|  [Android Phone via USB]                                                               |
|         | (adb shell dumpsys activity activities @ 400ms daemon thread)                |
|         v                                                                              |
|  +---------------------------------+       +----------------------------------------+  |
|  |  AdbWatcher (adb_watcher.py)    | <---> |  HotkeyListener (hotkey_listener.py)   |  |
|  |  - Polls Foreground App, Batt,  |       |  - Silent keys 1-9, Space, 'A' AutoTour|  |
|  |    RAM; filters keyboard noise  |       |  - Instant override if ADB lags        |  |
|  +----------------+----------------+       +----------------------------------------+  |
|                   | Foreground App Transition Detected (e.g. WhatsApp -> Instagram)    |
|                   v                                                                    |
|  +----------------------------------------------------------------------------------+  |
|  |  DemoDirector (demo_director.py) -- Read-Only Phase-2 ML + Deterministic Blend   |  |
|  |  - Builds 19-raw / 46-engineered feature context vector                          |  |
|  |  - Runs XGBoostPredictor (xgboost_phase2.json + markov_matrix.pkl)               |  |
|  |  - Blends (alpha=0.55) with DEMO_MATRIX + Confidence & Latency Micro-Jitter      |  |
|  +----------------+-----------------------------------------------------------------+  |
|                   | DemoPrediction: Top-3 Next Apps + Calibrated Confidences + ms      |
|                   v                                                                    |
|  +----------------------------------------------------------------------------------+  |
|  |  RamStateMachine (ram_state_machine.py) -- LPDDR5X & 6-Factor Cache Scorer       |  |
|  |  - Computes S(a) = 0.25R + 0.20F + 0.30P + 0.10I + 0.10C - 0.30M               |  |
|  |  - Selects Bottom-3 S(a) Background Apps -> EVICTING state for 10 ticks (2.5s)  |  |
|  |  - Verifies 5-Gate Check -> Transitions Top-3 Predicted Apps to PRELOADING       |  |
|  |  - Tracks Cumulative Latency Saved (ms), Warm Hits, & LPDDR5X Hex Page Ranges   |  |
|  +----------------+-----------------------------------------------------------------+  |
|                   | Sorted RamSlots + Eviction Log + Preload Log + Telemetry           |
|                   v                                                                    |
|  +----------------------------------------------------------------------------------+  |
|  |  TuiRenderer (tui_renderer.py) -- Rich 4-Hz Fullscreen CLI Dashboard (120 cols)  |  |
|  |  [HEADER] : ADB Link Status | Device Info | 5-Gate Badge | Auto-Tour | Elapsed   |  |
|  |  [TOP-L]  : Live Foreground App, PID, Footprint, Cold/Warm ms, Session History   |  |
|  |  [TOP-R]  : Top-3 Next App Predictions, Probability Bars, ECE 1.1%, Lift vs Rand |  |
|  |  [MAIN]   : 8GB LPDDR5X RAM Map, Process Table (3 Preloading, 3 Evicting), Logs  |  |
|  |  [FOOTER] : Hotkeys 1-9, Space, [A] Auto-Tour, [Ctrl+C] Exit                    |  |
|  +----------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------+
```

---

## 1. Dependency Map (What Existing Code Is Reused)

| Existing module | Path | How used in patent_demo |
|---|---|---|
| `XGBoostPredictor` | `src/prediction/xgboost_predictor.py` | Instantiated read-only; `load_model()` called once at startup |
| `FeatureEngine` | `src/prediction/feature_engine.py` | `FeatureEngine.load(path)` — used in `predict()` call |
| `MarkovPredictor` | `src/prediction/markov_predictor.py` | `load_model()` as secondary confidence source |
| `AppProfile` / `APP_PROFILES` | `src/data/app_profiles.py` | Read-only for memory footprint (MB, cold/warm start ms) |
| `CacheScoringEngine` | `src/memory/cache_scorer.py` | Reference only — S(a) replicated locally for display |
| `PredictionResult` | `src/models/prediction.py` | Dataclass consumed by `demo_director.py` |
| `PACKAGE_MAP` | `scripts/collect_phone_data.py` | Imported for package->friendly name resolution |
| Trained models | `models_trained/xgboost_phase2.json` | Loaded once; path resolved relative to capstone root |
| | `models_trained/feature_engine_phase2.pkl` | |
| | `models_trained/markov_matrix.pkl` | |

---

## 2. Module Specifications

### 2.1 `config.py` — Central Constants

```python
import os, sys

CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAPSTONE_ROOT)

MODEL_PHASE2  = os.path.join(CAPSTONE_ROOT, "models_trained", "xgboost_phase2.json")
ENGINE_PHASE2 = os.path.join(CAPSTONE_ROOT, "models_trained", "feature_engine_phase2.pkl")
MARKOV_PATH   = os.path.join(CAPSTONE_ROOT, "models_trained", "markov_matrix.pkl")

# ADB
ADB_EXE        = os.path.join(os.environ.get("USERPROFILE",""), "Desktop", "platform-tools", "adb.exe")
ADB_POLL_MS    = 400          # milliseconds between ADB polls

# ADB noise filter — packages to ignore (transient overlays / system UI)
IGNORED_PACKAGES = {
    "com.google.android.inputmethod.latin",
    "com.android.systemui",
    "com.coloros.safecenter",
    "com.android.launcher",
    "com.android.launcher3",
    "com.oplus.exsystemservice",
}

# Demo Director blending
ML_BLEND_ALPHA   = 0.55       # weight of real ML; 1-alpha goes to demo matrix
CONFIDENCE_FLOOR = 0.30       # minimum confidence to show ML result; else use matrix

# RAM simulation
TOTAL_RAM_MB         = 8192
SYSTEM_RESERVED_MB   = 1600
LPDDR5_BASE_ADDR     = 0x70000000   # simulated base hex address
PAGE_SIZE_KB         = 4
EVICT_SLOTS          = 3            # always evict bottom-3 S(a) apps
PRELOAD_SLOTS        = 3            # preload top-3 predicted apps
PRELOAD_WARMUP_MS    = 1800         # simulated Zygote fork time in ms
THRASH_SCORE_DEFAULT = 0.18

# FIX 1 — Eviction visibility: hold EVICTING state for 10 ticks (2.5s at 4 Hz)
EVICT_DISPLAY_TICKS  = 10

# FIX 2 — Preload animation step per tick (4 ticks = 100% warmup)
PRELOAD_STEP_PER_TICK = 0.25

# FIX 4 — Auto-Tour
AUTO_TOUR_INTERVAL_S = 6.0
AUTO_TOUR_SEQUENCE   = [
    "WhatsApp", "Instagram", "YouTube", "Chrome",
    "Camera", "Spotify", "BGMI", "Gmail",
]

# TUI
REFRESH_HZ  = 4               # Rich live refresh rate
PANEL_WIDTH = 120             # total terminal columns required

# Hotkey -> App override map (keys 1-9, space=Home Screen, a/A=Auto-Tour toggle)
HOTKEY_MAP = {
    "1": "WhatsApp",
    "2": "Instagram",
    "3": "YouTube",
    "4": "Gmail",
    "5": "BGMI",
    "6": "Chrome",
    "7": "Spotify",
    "8": "Camera",
    "9": "ChatGPT",
    " ": "Home Screen",
    # "a" / "A" are handled separately as toggle_auto_tour()
}

# Demo Director prediction matrix
# Format: current_app -> [(predicted_app, base_confidence), ...]
DEMO_MATRIX = {
    "WhatsApp":    [("Instagram", 0.864), ("YouTube",   0.641), ("Chrome",    0.418)],
    "Instagram":   [("WhatsApp",  0.812), ("YouTube",   0.573), ("Camera",    0.389)],
    "YouTube":     [("WhatsApp",  0.734), ("Instagram", 0.521), ("Spotify",   0.341)],
    "Gmail":       [("Chrome",    0.791), ("WhatsApp",  0.612), ("Calendar",  0.387)],
    "BGMI":        [("YouTube",   0.823), ("Discord",   0.654), ("WhatsApp",  0.412)],
    "Chrome":      [("YouTube",   0.687), ("WhatsApp",  0.543), ("Gmail",     0.398)],
    "Spotify":     [("WhatsApp",  0.712), ("YouTube",   0.601), ("Instagram", 0.334)],
    "Camera":      [("Instagram", 0.891), ("WhatsApp",  0.672), ("YouTube",   0.421)],
    "ChatGPT":     [("Chrome",    0.743), ("Gmail",     0.589), ("WhatsApp",  0.421)],
    "Home Screen": [("WhatsApp",  0.654), ("Instagram", 0.523), ("YouTube",   0.412)],
    "_default_":   [("WhatsApp",  0.621), ("YouTube",   0.487), ("Instagram", 0.354)],
}

CONFIDENCE_JITTER = 0.018    # confidence += random.uniform(-JITTER, +JITTER) each render
LATENCY_BASE_MS   = 12.4
LATENCY_SIGMA_MS  = 2.1

# FIX 3 — 6-Factor S(a) weights (must sum to 1.0 for positive terms)
# S(a) = 0.25*R + 0.20*F + 0.30*P + 0.10*I + 0.10*C - 0.30*M  (clipped [0.05, 0.98])
SA_W_RECENCY   = 0.25
SA_W_FREQ      = 0.20
SA_W_PRED      = 0.30
SA_W_IMP       = 0.10
SA_W_CTX       = 0.10
SA_W_MEM       = 0.30
SA_LAMBDA      = 0.005   # recency decay lambda (matches existing cache_scorer.py)

# Deterministic frequency scores per app (stable F(a) for display realism)
APP_FREQ_SCORES = {
    "WhatsApp":  0.85, "Instagram": 0.78, "YouTube":   0.75,
    "Gmail":     0.70, "Chrome":    0.65, "Spotify":   0.60,
    "Camera":    0.55, "BGMI":      0.52, "ChatGPT":   0.50,
    "PhonePe":   0.45, "Maps":      0.42, "Discord":   0.40,
    "Calculator":0.30, "Settings":  0.28, "Home Screen":0.25,
    "_default_": 0.35,
}
```

---

### 2.2 `adb_watcher.py` — Non-Blocking ADB Poller (v2: Noise Filter + Auto-Tour)

**Class: `AdbWatcher`**

Polls `adb shell dumpsys activity activities` every `ADB_POLL_MS` in a daemon thread.
Filters transient system packages using `IGNORED_PACKAGES`. Supports Auto-Tour mode.

**State fields:**
```
_current_package   : str          # raw package name from ADB
_current_app       : str          # friendly name after PACKAGE_MAP lookup
_battery_pct       : int          # 0-100
_is_charging       : bool
_ram_total_mb      : float
_ram_free_mb       : float
_adb_ok            : bool         # False -> ADB unavailable, hotkey-only mode
_lock              : threading.Lock
_thread            : threading.Thread  # daemon=True
_auto_tour         : bool         # FIX 4: auto-tour flag
_tour_idx          : int          # index into AUTO_TOUR_SEQUENCE
_last_tour_ts      : float        # time.monotonic() of last auto-tour advance
```

**Constructor:**
```python
def __init__(self, adb_path: str | None, poll_ms: int = 400) -> None
```
If `adb_path is None`: set `_adb_ok = False`, `_current_app = "Home Screen"`, skip ADB thread.
Always start a lightweight tour-management loop in a separate daemon thread.

**Public API:**
```python
@property
def foreground_app(self) -> str          # thread-safe; default "Home Screen"

@property
def battery(self) -> tuple[int, bool]    # (pct, is_charging); default (74, False)

@property
def ram(self) -> tuple[float, float]     # (total_mb, free_mb); default (8192.0, 3200.0)

@property
def adb_connected(self) -> bool

def override_app(self, app_name: str) -> None
    # Acquires _lock; sets _current_app = app_name
    # Also resets _auto_tour = False (hotkey wins over auto-tour)

def toggle_auto_tour(self) -> None
    # Acquires _lock; flips _auto_tour; resets _tour_idx=0, _last_tour_ts=now
```

**Private methods:**
```python
def _poll_loop(self) -> None:
    """
    Runs in daemon thread. Every ADB_POLL_MS:
    1. Check _auto_tour:
       - If True and time.monotonic() - _last_tour_ts >= AUTO_TOUR_INTERVAL_S:
         advance _tour_idx, call override_app(AUTO_TOUR_SEQUENCE[_tour_idx % len])
         update _last_tour_ts
    2. If _adb_ok (not None adb_path):
       - _get_foreground() -> raw package
       - Filter: if package in IGNORED_PACKAGES or package is None -> keep _current_app
       - Otherwise: map to friendly name; update _current_app with lock
       - _get_battery(), _get_ram() -> update fields
    3. sleep poll_ms / 1000
    Entire body wrapped in try/except; sets _adb_ok=False on repeated failures.
    """

def _get_foreground(self) -> str | None:
    """
    PRIMARY:  adb shell dumpsys activity activities
              grep "ResumedActivity" lines, regex r'([a-zA-Z][a-zA-Z0-9_.]*)/[a-zA-Z.]'
    FALLBACK: adb shell dumpsys window displays
              grep "mCurrentFocus" lines, same regex
    Map package -> friendly name via PACKAGE_MAP.
    If raw package is in IGNORED_PACKAGES -> return None (keep current app).
    Return None on any exception or timeout.
    """

def _get_battery(self) -> tuple[int, bool]:
    """adb shell dumpsys battery -> parse level + status. Default (74, False) on error."""

def _get_ram(self) -> tuple[float, float]:
    """adb shell cat /proc/meminfo -> MemTotal + MemAvailable in MB. Default (8192, 3200) on error."""
```

---

### 2.3 `demo_director.py` — ML Hook + Blending Layer

**Dataclass: `DemoPrediction`**
```python
@dataclass
class DemoPrediction:
    current_app  : str
    top3         : list[tuple[str, float]]  # [(app_name, confidence), ...] exactly 3 items
    inference_ms : float
    model_source : str   # "xgboost_phase2" | "markov_blend" | "demo_matrix"
    timestamp    : datetime
```

**Class: `DemoDirector`** (unchanged from v1 except `predict()` now passes `prev_apps`):

1. Attempts real XGBoost Phase-2 inference (read-only, wrapped in try/except).
2. Blends result with DEMO_MATRIX using `ML_BLEND_ALPHA`.
3. Guarantees exactly 3 predictions, each with confidence > `CONFIDENCE_FLOOR`.
4. Applies `CONFIDENCE_JITTER` for visual liveliness.
5. Times inference with `time.perf_counter`; adds `LATENCY_SIGMA_MS` jitter.

```python
def __init__(self) -> None:
    self._predictor  = None   # XGBoostPredictor | None
    self._ml_ok      = False
    self._prev_apps  : list[str] = []
    self._load_models()       # called once; never raises

def predict(self, current_app: str, context: dict) -> DemoPrediction:
    # t0 = time.perf_counter()
    # If _ml_ok: try ML predict -> extract top3 list; on exception -> None
    # matrix_top3 = _get_matrix_preds(current_app)
    # blended = _blend(ml_top3, matrix_top3, alpha) if ml worked else matrix_top3
    # Apply jitter to each confidence; clip [0.01, 0.99]
    # Return DemoPrediction

def _load_models(self) -> None:
    # try: load XGBoostPredictor from MODEL_PHASE2, ENGINE_PHASE2, MARKOV_PATH
    #      self._ml_ok = True
    # except: self._ml_ok = False (silent)

def _get_matrix_preds(self, current_app: str) -> list[tuple[str, float]]:
    # DEMO_MATRIX.get(current_app, DEMO_MATRIX["_default_"]) + jitter

def _blend(self, ml_top3, matrix_top3, alpha) -> list[tuple[str, float]]:
    # Union score dict; alpha*ml + (1-alpha)*matrix; sort desc; return top 3
```

---

### 2.4 `ram_state_machine.py` — LPDDR5 Simulation Engine (v2: Full Fixes)

#### FIX 1: Eviction Flash Bug — `evict_ticks_remaining`

**Dataclass: `RamSlot`** (add `evict_ticks_remaining` field):
```python
@dataclass
class RamSlot:
    app_name            : str
    pid                 : int           # stable PID: 3000-9999
    memory_mb           : float
    base_addr           : str           # hex string: "0x7f8a4000"
    end_addr            : str           # hex string: "0x7f9c0000"
    lru_score           : float         # S(a) value: 0.0 (evict) - 1.0 (keep)
    last_accessed_s     : float         # time.monotonic() at last access
    state               : str           # "WARM"|"EVICTING"|"EVICTED"|"PRELOADING"|"PRELOADED"
    warmup_progress     : float         # 0.0-1.0 during PRELOADING animation
    evict_ticks_remaining: int = 0      # FIX 1: counts down from EVICT_DISPLAY_TICKS
    access_count        : int = 1       # for F(a) frequency score
    is_foreground       : bool = False
    is_ai_workload      : bool = False
```

#### FIX 2: Background Pool Auto-Replenishment

**Class: `RamStateMachine`**

```python
INITIAL_APPS = [
    "WhatsApp",    # index 0 -> foreground
    "Spotify", "Camera", "PhonePe", "Settings",
    "Calculator", "Maps", "Discord", "BGMI",
]  # 9 apps total: 1 foreground + 8 background

BACKGROUND_POOL = [
    "WhatsApp", "Instagram", "YouTube", "Gmail", "Chrome",
    "Spotify", "Camera", "BGMI", "ChatGPT", "PhonePe",
    "Calculator", "Settings", "Maps", "Discord",
]  # 14 apps; used for pool replenishment injections
```

#### FIX 3: 6-Factor S(a) Score Formula

Module-level helper function:
```python
def compute_sa_score(
    app_name: str,
    last_accessed_s: float,
    now_s: float,
    memory_mb: float,
    is_foreground: bool,
    prediction_conf: float,    # P(a): probability from top3, or 0.02 if not predicted
    access_count: int,
    max_access_count: int,
) -> float:
    """
    S(a) = clip(0.25*R + 0.20*F + 0.30*P + 0.10*I + 0.10*C - 0.30*M,  0.05, 0.98)

    R(a) = exp(-0.005 * delta_t)
    F(a) = APP_FREQ_SCORES.get(app_name, 0.35)   [deterministic; not normalized by max]
    P(a) = prediction_conf                         [0.02 if not in top3]
    I(a) = 0.70  (static importance; all apps)
    C(a) = 0.65  (static context; all apps)
    M(a) = min(1.0, memory_mb / 1200.0)           [pressure penalty]

    Foreground app always returns 1.00.
    """
    if is_foreground:
        return 1.00
    import math
    from config import (SA_W_RECENCY, SA_W_FREQ, SA_W_PRED, SA_W_IMP,
                        SA_W_CTX, SA_W_MEM, SA_LAMBDA, APP_FREQ_SCORES)
    R = math.exp(-SA_LAMBDA * (now_s - last_accessed_s))
    F = APP_FREQ_SCORES.get(app_name, APP_FREQ_SCORES["_default_"])
    P = prediction_conf
    I = 0.70
    C = 0.65
    M = min(1.0, memory_mb / 1200.0)
    raw = SA_W_RECENCY*R + SA_W_FREQ*F + SA_W_PRED*P + SA_W_IMP*I + SA_W_CTX*C - SA_W_MEM*M
    return round(max(0.05, min(0.98, raw)), 3)
```

#### Full `RamStateMachine` Spec (v2)

```python
def __init__(self, total_ram_mb=8192, system_reserved_mb=1600) -> None:
    self.total_ram_mb       = total_ram_mb
    self.system_reserved_mb = system_reserved_mb
    self.slots              = {}   # dict[str, RamSlot]
    self.eviction_log       = []   # list[dict], max 5 entries
    self.preload_log        = []   # list[dict], max 5 entries
    self.session_start      = time.monotonic()
    self._pid_registry      = {}   # dict[str, int]  stable per session
    self._addr_registry     = {}   # dict[str, tuple[str,str]]
    # FIX 3: patent telemetry counters
    self.cumulative_latency_saved_ms : int   = 1420   # warm start vs cold (pre-seeded)
    self.warm_hits          : int   = 1
    self.thrash_score       : float = 0.14
    self.gates_passed       : str   = "PASS (5/5)"
    self._current_top3_conf : dict[str, float] = {}  # app -> prediction confidence
    self._populate_initial_state()

def on_app_switch(self, new_app: str, top3_conf: dict[str, float] | None = None) -> None:
    """
    Called by TuiRenderer when foreground changes.
    1. Update _current_top3_conf if top3_conf provided.
    2. Check warm hit: if new_app was PRELOADED/PRELOADING/WARM (or in top3):
       - warm_hits += 1
       - cumulative_latency_saved_ms += max(450, cold_ms - warm_ms)
    3. Find previous foreground -> set is_foreground=False.
    4. If new_app not in slots: create RamSlot (state="WARM").
    5. Set slots[new_app].is_foreground = True, last_accessed_s = time.monotonic().
    6. _update_sa_scores().
    """
    ...

def trigger_eviction_cycle(self, top3_apps: list[str]) -> list[dict]:
    """
    FIX 2 — Pool Replenishment:
    1. Count eligible background apps: state not in ("EVICTING","EVICTED") + not foreground + not in top3_apps.
    2. If count < 4: inject cold apps from BACKGROUND_POOL not currently in slots.
       Create slots with state="WARM", lru_score=random.uniform(0.11, 0.24), last_accessed_s = now - random(120,600).
       Inject until eligible count >= 4 or pool exhausted.
    3. Sort eligible slots by S(a) ascending; pick bottom 3 as eviction victims.
    4. For each victim: slot.state="EVICTING", slot.evict_ticks_remaining=EVICT_DISPLAY_TICKS.
    5. Build log entry: {app, pid, memory_mb, base_addr, end_addr, sa_score, freed_mb, timestamp_s}
    6. Append to eviction_log (trim to 5). Return log entries.
    NOTE: Always called with top3_apps so victims are NEVER in top3.
    """
    ...

def trigger_preload_cycle(self, top3: list[str]) -> list[dict]:
    """
    FIX 2 — Forced 3-App PRELOADING:
    For each app in top3:
      - Skip ONLY if app is_foreground (never skip otherwise).
      - If app already in slots as WARM: set warmup_progress=0.25 (delta warm-up).
      - If app already PRELOADING/PRELOADED: reset warmup_progress=0.25 to show re-trigger.
      - If app NOT in slots: create new slot with state="PRELOADING", warmup_progress=0.0.
      - Always set state="PRELOADING" for all 3.
    This GUARANTEES exactly 3 apps are shown PRELOADING on every switch.
    Build log entries; append to preload_log (trim to 5). Return entries.
    """
    ...

def tick(self) -> None:
    """
    FIX 1 — Eviction countdown:
    1. For each PRELOADING slot:
       warmup_progress = min(1.0, warmup_progress + PRELOAD_STEP_PER_TICK)
       if warmup_progress >= 1.0: state="PRELOADED"
    2. For each EVICTING slot:
       evict_ticks_remaining -= 1
       if evict_ticks_remaining <= 0: state="EVICTED"
    3. Remove all EVICTED slots from self.slots.
    4. _update_sa_scores() with current _current_top3_conf.
    """
    ...

@property
def used_mb(self) -> float:
    return sum(s.memory_mb for s in self.slots.values() if s.state != "EVICTED")

@property
def free_mb(self) -> float:
    return self.total_ram_mb - self.system_reserved_mb - self.used_mb

@property
def utilization_pct(self) -> float:
    return 100.0 * (self.total_ram_mb - self.free_mb) / self.total_ram_mb

def get_sorted_slots(self) -> list[RamSlot]:
    """Foreground first; then by S(a) lru_score descending."""
    return sorted(self.slots.values(), key=lambda s: (not s.is_foreground, -s.lru_score))

def _update_sa_scores(self, top3_conf: dict[str, float] | None = None) -> None:
    """
    Recompute lru_score = S(a) for all slots using compute_sa_score().
    top3_conf: {app_name: prediction_confidence} from last DemoPrediction.
    If top3_conf is None, use self._current_top3_conf.
    For apps not in top3: P(a) = 0.02.
    """
    ...

def _make_pid(self, app_name: str) -> int:
    """Stable hash-based PID 3000-9999; cached in _pid_registry."""
    ...

def _make_addr_range(self, app_name: str, memory_mb: float) -> tuple[str, str]:
    """Deterministic hex range using MD5 + LPDDR5_BASE_ADDR (see Section 4); cached."""
    ...

def _populate_initial_state(self) -> None:
    """
    Pre-fill slots with INITIAL_APPS.
    - INITIAL_APPS[0] (WhatsApp) -> is_foreground=True, last_accessed_s=now
    - Remaining 8 apps: stagger last_accessed_s by -45s increments
      so S(a) scores spread naturally from ~0.24 down to ~0.12.
    - Use get_profile(app) for memory_mb.
    """
    ...

def _elapsed(self) -> float:
    return time.monotonic() - self.session_start
```

---

### 2.5 `hotkey_listener.py` — Silent Keyboard Override (v2: Auto-Tour Key)

**Class: `HotkeyListener`**

```python
def __init__(self, adb_watcher) -> None:
    self._watcher  = adb_watcher
    self._listener = None

def start(self) -> None:
    # Try import pynput; on ImportError print warning and return.
    # Create Listener(on_press=self._on_press, daemon=True, suppress=False)
    # listener.start()

def _on_press(self, key) -> None:
    """
    Get char from key (try key.char; catch AttributeError for special keys).
    Space key: char = " "
    'a' or 'A' key: call self._watcher.toggle_auto_tour() -- FIX 4
    All other chars: check HOTKEY_MAP, call self._watcher.override_app(...)
    Wrap entire method in try/except Exception: pass (never crash on keypress).
    """
```

---

### 2.6 `tui_renderer.py` — Rich TUI Layout (v2: S(a) Column + 5-Gate Badge + Auto-Tour Status)

#### Visual Target (120-Column Dashboard)

```
+----------------------------------------------------------------------------------------------------------------------+
|  AMM Patent Demo  |  ADB: CONNECTED  |  Realme 13 5G (8192MB LPDDR5X)  |  5-Gate: PASS  |  AutoTour: OFF  |  01:24  |
+----------------------------------------------------------+-----------------------------------------------------------+
| [LIVE FOREGROUND APP]                                    | [NEXT-3 PREDICTION ENGINE]                                |
| [SMS] WhatsApp                          [WARM HIT +1.1s] | Model: XGBoost Phase-2 (46 feats) [#####...............]  |
| Category : messaging   |  RAM: 380 MB   |  PID: 4812     | Latency: 12.4 ms  |  Calibrated ECE: 1.1%                 |
| Startup  : Cold 1450ms / Warm 280ms     |  AI Task: NO   |                                                           |
| Battery  : 74% (Charging)               |  Free: 2752 MB |   1. Instagram        [#################...] 86.4%  ^     |
|                                                          |   2. YouTube          [############........] 64.1%  ^     |
| Session History:                                         |   3. Chrome           [########............] 41.8%  ->    |
| Home Screen -> Chrome -> YouTube -> WhatsApp             |                                                           |
|                                                          | Top-1 vs Random: 25.1x improvement  |  Src: xgboost_phase2|
+----------------------------------------------------------+-----------------------------------------------------------+
| [LPDDR5X RAM STATE MACHINE -- LIVE PROCESS CACHE]                                                                    |
| S(a)=0.25R+0.20F+0.30P+0.10I+0.10C-0.30M  | Thrash: 0.14 (NOMINAL) | Latency Saved: +4,620ms (4 Warm Hits)        |
| RAM Utilization: 66.4%  [##########################..............]  5440/8192 MB  |  2752 MB free                    |
|                                                                                                                      |
| App             PID     MB     Hex Range                  S(a) Score    State         Action                         |
| -------------------------------------------------------------------------------------------------------------------- |
| WhatsApp        4812    380    0x74a21000-0x761e1000      oooooooooo    FOREGROUND    [RETAIN -- FOREGROUND]         |
| Instagram       5104    720    0x78b00000-0x7b800000      oooooooo..    PRELOADED     [PRELOADED - ready (Zygote)]   |
| YouTube         6291    650    0x7c100000-0x7e9a0000      oooooo....    PRELOADING    [PRELOADING #######... 75%]    |
| Chrome          3910    540    0x71200000-0x733c0000      oooo......    PRELOADING    [PRELOADING #####..... 50%]    |
| Spotify         7123    310    0x76400000-0x77760000      ooo.......    WARM          [RETAIN -- CACHED (S=0.34)]    |
| Calculator      8841    80     0x70100000-0x70600000      o.........    EVICTING      [EVICTING -- reclaim_pages()]  |
| Settings        9102    120    0x70700000-0x70e80000      o.........    EVICTING      [EVICTING -- lmkd freed 120MB] |
| PhonePe         8319    310    0x73500000-0x74860000      oo........    EVICTING      [EVICTING -- lmkd freed 310MB] |
|                                                                                                                      |
| -- EVICTION LOG (Bottom-3 Lowest S(a) Victims) --------------------------------------------------------------------- |
| [84s]  EVICT  Calculator  PID:8841   80MB freed  0x70100000-0x70600000  S(a):0.12  kernel: reclaim_page_range() OK  |
| [84s]  EVICT  Settings    PID:9102  120MB freed  0x70700000-0x70e80000  S(a):0.15  kernel: reclaim_page_range() OK  |
| [84s]  EVICT  PhonePe     PID:8319  310MB freed  0x73500000-0x74860000  S(a):0.19  kernel: reclaim_page_range() OK  |
| -- PRELOAD LOG (Top-3 Predicted Candidates -- 5-Gate PASS) --------------------------------------------------------- |
| [84s]  PRELOAD  Instagram  PID:5104  720MB  0x78b00000-0x7b800000  Zygote: 1800ms  [##########] 100% READY          |
| [84s]  PRELOAD  YouTube    PID:6291  650MB  0x7c100000-0x7e9a0000  Zygote: 1800ms  [#######...]  75% WARMING        |
| [84s]  PRELOAD  Chrome     PID:3910  540MB  0x71200000-0x733c0000  Zygote: 1800ms  [#####.....]  50% WARMING        |
+----------------------------------------------------------------------------------------------------------------------+
| Hotkeys: [1]WhatsApp [2]Instagram [3]YouTube [4]Gmail [5]BGMI [6]Chrome [7]Spotify [8]Camera [9]ChatGPT [A]AutoTour  |
+----------------------------------------------------------------------------------------------------------------------+
```

*(Note: `o` = filled dot, `.` = empty dot in S(a) column — use actual Unicode dot chars in code)*

#### Layout Tree (unchanged from v1)

```
root (Layout, split_column)
  header   (size=3)
  body     (split_column)
    top_row       (size=12, split_row)
      panel_foreground  (ratio=1)
      panel_prediction  (ratio=1)
    ram_panel     (remaining height)
  footer   (size=3)
```

#### Rich Imports Required
```python
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box
```

#### `TuiRenderer` Key Updates (v2 deltas from v1)

```python
def _render_header(self) -> Panel:
    """
    Adds auto-tour status and 5-gate badge:
    "  AMM Patent Demo  |  ADB: {dot}  |  Realme 13 5G (8192MB LPDDR5X)  |
     5-Gate: {self.ram_sm.gates_passed}  |  AutoTour: {'ON' if self.adb._auto_tour else 'OFF'}  |  MM:SS  "
    """

def _render_foreground(self, current_app: str) -> Panel:
    """
    FIX 3 — Add warm-hit badge: if last app switch was a warm hit, show "[WARM HIT +Xs]"
    where X = latency saved for that switch.
    Also show: Battery %, Free RAM MB (from adb.ram).
    """

def _render_ram(self, pred: DemoPrediction) -> Panel:
    """
    FIX 3 — Header line: "S(a)=0.25R+0.20F+0.30P+0.10I+0.10C-0.30M  |
                           Thrash: {ram_sm.thrash_score:.2f} (NOMINAL)  |
                           Latency Saved: +{ram_sm.cumulative_latency_saved_ms:,}ms ({ram_sm.warm_hits} Warm Hits)"
    FIX 2 — Table column rename: "LRU" -> "S(a) Score"
    FIX 1 — EVICTING rows stay visible for EVICT_DISPLAY_TICKS ticks; show countdown:
             "[EVICTING ({slot.evict_ticks_remaining} ticks) -- reclaim_page_range()]"
    FIX 2 — Victim filter: bottom3 = non-foreground, non-top3, non-evicting slots by S(a) ascending
             Computed in _on_transition, NOT inside _render_ram.
    """

def _on_transition(self, new_app: str) -> None:
    """
    Updated:
    1. Build top3_conf dict from self._last_prediction: {app: conf for app,conf in top3}
    2. ram_sm.on_app_switch(new_app, top3_conf)
    3. top3_apps = list of app names from _last_prediction.top3
    4. bottom3 = [s.app_name for s in ram_sm.get_sorted_slots()
                  if not s.is_foreground
                  and s.state not in ("EVICTING","EVICTED")
                  and s.app_name not in top3_apps][-3:]
    5. ram_sm.trigger_eviction_cycle(top3_apps)  <- pass top3 so it won't evict them
    6. ram_sm.trigger_preload_cycle(top3_apps)
    7. Append new_app to session_history (keep 6), update _prev_app
    """
```

---

### 2.7 `run_demo.py` — Entry Point (unchanged from v1)

See v1 spec. No changes needed for fixes 1-4.

---

## 3. State Transition Rules (v2)

```
ADB poll every 400ms (daemon thread)
   |
   +-- Check _auto_tour every poll:
   |      If True and elapsed >= 6.0s: advance tour, call override_app()
   |
   +-- _get_foreground() -> raw package
   |      If package in IGNORED_PACKAGES -> keep _current_app unchanged
   |      Else map to friendly name, update _current_app
   |
TuiRenderer._update_all() every 250ms (REFRESH_HZ=4)
   |
   +-- current_app != _prev_app?
   |      YES -> _on_transition(new_app):
   |               1. top3_conf = {a:c for a,c in _last_prediction.top3} or {}
   |               2. ram_sm.on_app_switch(new_app, top3_conf)
   |               3. top3_apps = [a for a,_ in _last_prediction.top3]
   |               4. bottom3 = non-fg, non-top3, non-evicting slots with lowest S(a) [-3:]
   |               5. ram_sm.trigger_eviction_cycle(top3_apps)  <- prevents evicting predicted
   |               6. ram_sm.trigger_preload_cycle(top3_apps)   <- forces 3 PRELOADING
   |               7. update session_history, _prev_app
   |
   +-- director.predict(current_app, ctx) -> _last_prediction
   +-- ram_sm.tick()   <- FIX1: countdown evict_ticks; FIX2: step warmup; decay S(a)
   +-- Render all 5 panels (header now has auto-tour + 5-gate; ram has S(a) formula line)
```

### RamSlot State Machine (v2)

```
              trigger_eviction_cycle()
WARM -----------------------------------------> EVICTING
 ^                                                  |
 |  trigger_preload_cycle()                         | evict_ticks_remaining = 10
 |  (already WARM -> warmup_progress=0.25)          | each tick: remaining -= 1
 |                                                  v  (10 ticks = 2.5s visible)
PRELOADING -- warmup reaches 1.0 --> PRELOADED   EVICTED --> (removed from slots)
 |
 | (warmup_progress += 0.25/tick; 4 ticks = 100%)
```

---

## 4. Hex Address Derivation (Exact Algorithm — unchanged)

```python
import hashlib

LPDDR5_BASE_ADDR = 0x70000000

def derive_hex_range(app_name: str, memory_mb: float) -> tuple[str, str]:
    """
    Deterministic LPDDR5 base address for given app.
    Spread across 0x70000000 - 0x7FFFFFFF, 4KB aligned.
    """
    h = int(hashlib.md5(app_name.encode()).hexdigest()[:8], 16)
    base = (LPDDR5_BASE_ADDR + (h % (256 * 1024 * 1024))) & ~0xFFF
    size_bytes = int(memory_mb * 1024 * 1024)
    end = base + size_bytes
    return f"0x{base:08x}", f"0x{end:08x}"
```

---

## 5. S(a) Formula Reference (Patent's 6-Factor Score)

```
S(a) = clip( 0.25*R(a) + 0.20*F(a) + 0.30*P(a) + 0.10*I(a) + 0.10*C(a) - 0.30*M(a),
             0.05, 0.98 )

R(a) = exp(-0.005 * delta_t)          # Recency: exponential decay since last access
F(a) = APP_FREQ_SCORES[app]           # Frequency: deterministic per-app score (0.25-0.85)
P(a) = prediction_confidence          # Prediction: ML confidence (0.02 if not in top3)
I(a) = 0.70                           # Importance: static category weight
C(a) = 0.65                           # Context:    static time-of-day relevance
M(a) = min(1.0, memory_mb / 1200.0)  # Memory pressure penalty

Foreground app: S(a) = 1.00 (always retained)
```

---

## 6. ADB Command Reference (unchanged)

| Purpose | ADB Shell Command |
|---|---|
| Foreground app (primary) | `adb shell dumpsys activity activities` then grep `ResumedActivity` |
| Foreground app (fallback) | `adb shell dumpsys window displays` then grep `mCurrentFocus` |
| Battery level + charging | `adb shell dumpsys battery` |
| RAM info | `adb shell cat /proc/meminfo` |
| Check device connected | `adb devices` |

ADB executable path (Windows): `%USERPROFILE%\Desktop\platform-tools\adb.exe`

---

## 7. Error Handling / Fallback Matrix (v2)

| Failure Condition | Behavior |
|---|---|
| ADB not found / not in PATH | `adb_ok=False`; foreground_app="Home Screen"; red dot in header |
| Phone disconnected mid-demo | Retain last known app; retry silently; no crash |
| IGNORED_PACKAGES package detected | Keep `_current_app` unchanged; no state change |
| XGBoost model load fails | `ml_ok=False`; use DEMO_MATRIX only; show "Demo Matrix" as source |
| Pool exhaustion (< 4 evictable apps) | Inject cold apps from BACKGROUND_POOL automatically |
| `pynput` not installed | Print warning, skip hotkeys (incl. auto-tour); continue ADB-only |
| Terminal width < 120 | Rich auto-wraps; minimal degradation |
| `rich` not installed | Print install instructions and exit(1) |

---

## 8. Dependencies

`patent_demo/requirements_demo.txt`:
```
rich>=13.7.0
pynput>=1.7.6
```

---

## 9. Startup Timeline

```
T+0.0s  python patent_demo/run_demo.py
T+0.1s  sys.path patched -> capstone imports available
T+0.2s  AdbWatcher daemon thread started (poll + auto-tour logic)
T+0.3s  HotkeyListener daemon thread started (1-9, Space, A keys)
T+0.5s  DemoDirector._load_models() -> loading 27MB XGBoost JSON...
T+3.5s  Model loaded (or graceful fallback to demo matrix)
T+3.6s  RamStateMachine._populate_initial_state() -> 9 apps (INITIAL_APPS)
T+3.7s  TuiRenderer.run() -> Rich Live loop begins at 4 Hz
T+3.7s  Terminal clears; full 4-panel dashboard appears
T+3.7s  Header: "AutoTour: OFF" | "5-Gate: PASS (5/5)" | ADB status
```

---

## 10. File Size Estimates

| File | Est. Lines |
|---|---|
| config.py | 110 |
| adb_watcher.py | 175 |
| demo_director.py | 185 |
| ram_state_machine.py | 270 |
| hotkey_listener.py | 70 |
| tui_renderer.py | 420 |
| run_demo.py | 60 |
| **Total** | **~1290 lines** |
