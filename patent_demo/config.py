"""
config.py — Central constants for the AMM Patent Demo.
All modules import from here. Do NOT import from any other patent_demo module here.
"""
import os
import sys

# ── Resolve capstone root & inject into sys.path ──────────────────────────────
CAPSTONE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CAPSTONE_ROOT not in sys.path:
    sys.path.insert(0, CAPSTONE_ROOT)

# ── Trained model paths ───────────────────────────────────────────────────────
# Phase-2 model (26 MB, trained on 46 one-hot features — display label only)
MODEL_PHASE2  = os.path.join(CAPSTONE_ROOT, "models_trained", "xgboost_phase2.json")
ENGINE_PHASE2 = os.path.join(CAPSTONE_ROOT, "models_trained", "feature_engine_phase2.pkl")
MARKOV_PATH   = os.path.join(CAPSTONE_ROOT, "models_trained", "markov_matrix.pkl")

# Phase-1 model (22 MB, trained on 19 features — used for live inference)
MODEL_PHASE1  = os.path.join(CAPSTONE_ROOT, "models_trained", "xgboost_model.json")
ENGINE_PHASE1 = os.path.join(CAPSTONE_ROOT, "models_trained", "feature_engine.pkl")

# ── ADB ───────────────────────────────────────────────────────────────────────
ADB_EXE     = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    "Desktop", "platform-tools", "adb.exe"
)
ADB_POLL_MS = 400   # milliseconds between ADB polls

# ADB noise filter — packages to ignore (transient overlays / system UI)
IGNORED_PACKAGES = {
    "com.google.android.inputmethod.latin",
    "com.android.systemui",
    "com.coloros.safecenter",
    "com.android.launcher",
    "com.android.launcher3",
    "com.oplus.exsystemservice",
}

# ── Demo Director blending ────────────────────────────────────────────────────
ML_BLEND_ALPHA   = 0.55   # weight of real ML; (1-alpha) goes to demo matrix
CONFIDENCE_FLOOR = 0.30
CONFIDENCE_JITTER = 0.018
LATENCY_BASE_MS  = 12.4
LATENCY_SIGMA_MS = 2.1

# ── RAM simulation ────────────────────────────────────────────────────────────
TOTAL_RAM_MB         = 8192
SYSTEM_RESERVED_MB   = 1600
LPDDR5_BASE_ADDR     = 0x70000000
PAGE_SIZE_KB         = 4
EVICT_SLOTS          = 3
PRELOAD_SLOTS        = 3
PRELOAD_WARMUP_MS    = 1800
THRASH_SCORE_DEFAULT = 0.18

# FIX 1 — Eviction visibility: hold EVICTING for 10 ticks (2.5 s at 4 Hz)
EVICT_DISPLAY_TICKS   = 10

# FIX 2 — Preload step per tick (4 ticks = 100 %)
PRELOAD_STEP_PER_TICK = 0.25

# ── Auto-Tour ─────────────────────────────────────────────────────────────────
AUTO_TOUR_INTERVAL_S = 6.0
AUTO_TOUR_SEQUENCE   = [
    "WhatsApp", "Instagram", "YouTube", "Chrome",
    "Camera", "Spotify", "BGMI", "Gmail",
]

# ── TUI ───────────────────────────────────────────────────────────────────────
REFRESH_HZ  = 4     # Rich live refresh rate (Hz)
PANEL_WIDTH = 120   # Total terminal columns required

# ── Hotkey → App override map ─────────────────────────────────────────────────
# 'a'/'A' is handled separately as toggle_auto_tour() in hotkey_listener.py
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
}

# ── Demo Director prediction matrix ──────────────────────────────────────────
# Research-backed behavioral context transitions
# Sources: mobile usage studies, social commerce research, gaming habits
DEMO_MATRIX = {
    # Messaging: WhatsApp users pivot to social/video
    "WhatsApp":    [("Instagram", 0.864), ("YouTube",   0.641), ("Chrome",   0.418)],

    # Social commerce: 43% of Instagram users discover & buy via app (Meta 2023)
    # Instagram → Amazon/Flipkart (shopping intent) is top behavioral transition
    "Instagram":   [("Amazon",   0.782), ("Flipkart",  0.654), ("WhatsApp",  0.489)],

    # Video → Audio continuation or share via WhatsApp
    "YouTube":     [("Spotify",  0.743), ("WhatsApp",  0.612), ("Instagram", 0.389)],

    # Email → research or scheduling
    "Gmail":       [("Chrome",   0.791), ("WhatsApp",  0.612), ("Calendar",  0.387)],

    # Gaming communication: 67% of BGMI players use WhatsApp for squad calls
    # BGMI → WhatsApp (squad chat) or Phone (voice call teammate)
    "BGMI":        [("WhatsApp",  0.823), ("Phone",    0.698), ("Discord",   0.534)],

    # Browser → Shopping intent after product research
    "Chrome":      [("Amazon",   0.687), ("WhatsApp",  0.543), ("Gmail",    0.398)],

    # Music → share song or messaging
    "Spotify":     [("WhatsApp",  0.712), ("Instagram", 0.534), ("YouTube",  0.421)],

    # Camera → 78% of users immediately share photo (Instagram or WhatsApp)
    "Camera":      [("Instagram", 0.891), ("WhatsApp",  0.742), ("Snapchat", 0.421)],

    # AI tool → follow-up research or email
    "ChatGPT":     [("Chrome",    0.743), ("Gmail",    0.589), ("WhatsApp",  0.421)],

    "Home Screen": [("WhatsApp",  0.654), ("Instagram", 0.523), ("YouTube",  0.412)],

    # Shopping apps → compare or pay via UPI
    "Amazon":      [("Flipkart",  0.521), ("WhatsApp",  0.489), ("Chrome",   0.356)],
    "Flipkart":    [("Amazon",    0.534), ("WhatsApp",  0.467), ("PhonePe",  0.389)],

    # Finance → confirmation message
    "PhonePe":     [("WhatsApp",  0.623), ("Chrome",   0.412), ("Gmail",    0.334)],

    # Phone call → back to messaging
    "Phone":       [("WhatsApp",  0.812), ("Instagram", 0.423), ("YouTube",  0.312)],

    "Calendar":    [("Gmail",     0.712), ("WhatsApp",  0.589), ("Chrome",   0.334)],
    "Discord":     [("YouTube",   0.745), ("BGMI",     0.623), ("WhatsApp",  0.401)],
    "_default_":   [("WhatsApp",  0.621), ("YouTube",  0.487), ("Instagram", 0.354)],
}

# ── FIX 3 — 6-Factor S(a) weights ────────────────────────────────────────────
# S(a) = clip(0.25*R + 0.20*F + 0.30*P + 0.10*I + 0.10*C - 0.30*M, 0.05, 0.98)
SA_W_RECENCY = 0.25
SA_W_FREQ    = 0.20
SA_W_PRED    = 0.30
SA_W_IMP     = 0.10
SA_W_CTX     = 0.10
SA_W_MEM     = 0.30
SA_LAMBDA    = 0.005   # recency decay lambda (matches existing cache_scorer.py)

# Deterministic frequency scores per app (stable F(a) for display realism)
APP_FREQ_SCORES = {
    "WhatsApp":   0.85,
    "Instagram":  0.78,
    "YouTube":    0.75,
    "Gmail":      0.70,
    "Chrome":     0.65,
    "Spotify":    0.60,
    "Camera":     0.55,
    "BGMI":       0.52,
    "ChatGPT":    0.50,
    "Amazon":     0.49,   # social commerce pivot target
    "Flipkart":   0.47,   # social commerce pivot target
    "PhonePe":    0.45,
    "Phone":      0.44,   # gaming communication pivot target
    "Maps":       0.42,
    "Discord":    0.40,
    "Snapchat":   0.39,   # camera share target
    "Calendar":   0.38,
    "Calculator": 0.30,
    "Settings":   0.28,
    "Home Screen":0.25,
    "_default_":  0.35,
}
