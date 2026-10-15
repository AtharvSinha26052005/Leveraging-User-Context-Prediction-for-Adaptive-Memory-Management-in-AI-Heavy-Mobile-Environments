"""
App profiles — 20 realistic application definitions with resource footprints.
Each profile represents a typical mobile app with its memory, launch time, and category.
"""
from src.models.context import AppProfile

APP_PROFILES: dict[str, AppProfile] = {
    # ── Social Media ──
    "Instagram": AppProfile("Instagram", "social", 180, 80, 2200, 150, 0.6, is_ai_workload=True, ai_cache_mb=120),
    "Twitter": AppProfile("Twitter", "social", 120, 50, 1800, 100, 0.5),
    "TikTok": AppProfile("TikTok", "social", 200, 100, 2500, 180, 0.5, is_ai_workload=True, ai_cache_mb=150),

    # ── Messaging ──
    "WhatsApp": AppProfile("WhatsApp", "messaging", 100, 60, 1200, 80, 0.9),
    "Telegram": AppProfile("Telegram", "messaging", 90, 40, 1100, 70, 0.8),

    # ── Productivity ──
    "Gmail": AppProfile("Gmail", "productivity", 110, 50, 1500, 90, 0.85),
    "Docs": AppProfile("Docs", "productivity", 100, 40, 1400, 100, 0.7),
    "Calendar": AppProfile("Calendar", "productivity", 60, 20, 800, 60, 0.75),

    # ── Entertainment ──
    "YouTube": AppProfile("YouTube", "entertainment", 160, 70, 2000, 130, 0.5, is_ai_workload=True, ai_cache_mb=80),
    "Netflix": AppProfile("Netflix", "entertainment", 180, 80, 2200, 140, 0.4),
    "Spotify": AppProfile("Spotify", "entertainment", 100, 50, 1300, 90, 0.5),

    # ── Navigation ──
    "Maps": AppProfile("Maps", "navigation", 140, 60, 1800, 120, 0.7, is_ai_workload=True, ai_cache_mb=60),

    # ── Camera / AI ──
    "Camera": AppProfile("Camera", "camera", 150, 50, 3000, 200, 0.8, is_ai_workload=True, ai_cache_mb=400),
    "ChatGPT": AppProfile("ChatGPT", "ai_tools", 200, 60, 4000, 180, 0.6, is_ai_workload=True, ai_cache_mb=800),
    "Translate": AppProfile("Translate", "ai_tools", 120, 40, 2500, 120, 0.6, is_ai_workload=True, ai_cache_mb=300),

    # ── Gaming ──
    "CandyCrush": AppProfile("CandyCrush", "gaming", 180, 100, 3000, 200, 0.3),
    "BGMI": AppProfile("BGMI", "gaming", 400, 200, 5000, 300, 0.2),

    # ── Finance ──
    "PhonePe": AppProfile("PhonePe", "finance", 80, 30, 1200, 80, 0.85),
    "Paytm": AppProfile("Paytm", "finance", 90, 35, 1300, 90, 0.8),

    # ── System ──
    "Settings": AppProfile("Settings", "system", 50, 10, 600, 50, 0.95),
}

# Convenience lists
ALL_APP_NAMES = list(APP_PROFILES.keys())
CATEGORIES = list(set(p.category for p in APP_PROFILES.values()))

def get_profile(app_name: str) -> AppProfile:
    """Get app profile by name, with a sensible default fallback."""
    if app_name in APP_PROFILES:
        return APP_PROFILES[app_name]
    return AppProfile(app_name, "unknown", 100, 30, 1500, 100, 0.5)

def get_apps_by_category(category: str) -> list[str]:
    """Get all app names in a category."""
    return [name for name, p in APP_PROFILES.items() if p.category == category]
