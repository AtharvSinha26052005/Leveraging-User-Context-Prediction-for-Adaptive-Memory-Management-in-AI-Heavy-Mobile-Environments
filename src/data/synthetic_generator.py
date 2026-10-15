"""
Synthetic data generator — produces realistic app-switch events for 5 user personas.
Each persona has distinct usage patterns, temporal rhythms, and app preferences.
"""
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd

from src.data.app_profiles import APP_PROFILES, get_profile


# ═══════════════════════════════════════════════════════════════
# PERSONA DEFINITIONS
# ═══════════════════════════════════════════════════════════════

PERSONAS: Dict[str, dict] = {
    "student": {
        "description": "College student — heavy social media at night, educational apps during day",
        "app_weights": {
            "Instagram": 0.15, "TikTok": 0.12, "WhatsApp": 0.14, "YouTube": 0.12,
            "Telegram": 0.06, "Spotify": 0.06, "Docs": 0.05, "Gmail": 0.04,
            "Twitter": 0.04, "Camera": 0.04, "ChatGPT": 0.05, "CandyCrush": 0.04,
            "PhonePe": 0.03, "Maps": 0.02, "Calendar": 0.02, "Settings": 0.01,
            "Netflix": 0.01,
        },
        "hourly_activity": {  # relative activity level per hour (0.0 – 1.0)
            8: 0.3, 9: 0.4, 10: 0.5, 11: 0.5, 12: 0.7, 13: 0.6, 14: 0.5,
            15: 0.4, 16: 0.5, 17: 0.6, 18: 0.7, 19: 0.8, 20: 0.9, 21: 1.0,
            22: 0.9, 23: 0.7, 0: 0.4, 1: 0.2,
        },
        "transitions": {  # common app sequences
            "Instagram": ["WhatsApp", "Camera", "TikTok"],
            "WhatsApp": ["Instagram", "YouTube", "Telegram"],
            "YouTube": ["Instagram", "WhatsApp", "Spotify"],
            "ChatGPT": ["Docs", "Gmail", "WhatsApp"],
        },
    },
    "professional": {
        "description": "Working professional — productivity during work hours, light entertainment evenings",
        "app_weights": {
            "Gmail": 0.16, "WhatsApp": 0.14, "Calendar": 0.08, "Docs": 0.10,
            "Maps": 0.06, "PhonePe": 0.05, "Twitter": 0.04, "Instagram": 0.05,
            "YouTube": 0.05, "Telegram": 0.04, "Spotify": 0.04, "Camera": 0.03,
            "ChatGPT": 0.05, "Netflix": 0.03, "Settings": 0.02, "Paytm": 0.03,
            "Translate": 0.03,
        },
        "hourly_activity": {
            7: 0.4, 8: 0.7, 9: 0.9, 10: 1.0, 11: 0.9, 12: 0.6, 13: 0.7,
            14: 0.9, 15: 0.8, 16: 0.7, 17: 0.6, 18: 0.5, 19: 0.4, 20: 0.5,
            21: 0.4, 22: 0.3,
        },
        "transitions": {
            "Gmail": ["Calendar", "Docs", "WhatsApp"],
            "Calendar": ["Gmail", "Maps", "Docs"],
            "Maps": ["WhatsApp", "Gmail", "PhonePe"],
        },
    },
    "casual": {
        "description": "Casual user — entertainment + messaging, moderate usage",
        "app_weights": {
            "WhatsApp": 0.18, "YouTube": 0.14, "Instagram": 0.12, "Spotify": 0.08,
            "Netflix": 0.06, "TikTok": 0.06, "Twitter": 0.04, "Camera": 0.04,
            "PhonePe": 0.05, "Gmail": 0.04, "Maps": 0.03, "CandyCrush": 0.06,
            "Telegram": 0.03, "Settings": 0.02, "Paytm": 0.03, "Calendar": 0.02,
        },
        "hourly_activity": {
            9: 0.3, 10: 0.4, 11: 0.4, 12: 0.5, 13: 0.5, 14: 0.4, 15: 0.4,
            16: 0.5, 17: 0.6, 18: 0.7, 19: 0.8, 20: 0.9, 21: 1.0, 22: 0.8,
            23: 0.5,
        },
        "transitions": {
            "WhatsApp": ["YouTube", "Instagram", "PhonePe"],
            "YouTube": ["WhatsApp", "Instagram", "Netflix"],
            "Instagram": ["WhatsApp", "Camera", "YouTube"],
        },
    },
    "power_user": {
        "description": "Power user — heavy gaming + AI tools, high device usage",
        "app_weights": {
            "BGMI": 0.12, "ChatGPT": 0.10, "YouTube": 0.08, "WhatsApp": 0.10,
            "Instagram": 0.06, "Camera": 0.06, "Twitter": 0.04, "Translate": 0.05,
            "Gmail": 0.04, "Telegram": 0.05, "Spotify": 0.04, "TikTok": 0.04,
            "Maps": 0.04, "Docs": 0.03, "CandyCrush": 0.05, "Netflix": 0.04,
            "PhonePe": 0.03, "Settings": 0.03,
        },
        "hourly_activity": {
            8: 0.3, 9: 0.4, 10: 0.5, 11: 0.6, 12: 0.6, 13: 0.5, 14: 0.6,
            15: 0.7, 16: 0.8, 17: 0.8, 18: 0.9, 19: 1.0, 20: 1.0, 21: 0.9,
            22: 0.8, 23: 0.6, 0: 0.4, 1: 0.2,
        },
        "transitions": {
            "BGMI": ["WhatsApp", "YouTube", "ChatGPT"],
            "ChatGPT": ["Docs", "Translate", "WhatsApp"],
            "Camera": ["Instagram", "WhatsApp", "ChatGPT"],
        },
    },
    "developer": {
        "description": "Developer — productivity-heavy, uses AI tools, moderate social",
        "app_weights": {
            "ChatGPT": 0.14, "Gmail": 0.10, "Docs": 0.10, "WhatsApp": 0.10,
            "Telegram": 0.08, "Twitter": 0.06, "YouTube": 0.06, "Spotify": 0.05,
            "Maps": 0.04, "Instagram": 0.04, "Camera": 0.03, "Settings": 0.04,
            "Translate": 0.04, "PhonePe": 0.03, "Calendar": 0.04, "Netflix": 0.03,
            "Paytm": 0.02,
        },
        "hourly_activity": {
            8: 0.4, 9: 0.7, 10: 0.9, 11: 1.0, 12: 0.5, 13: 0.6, 14: 0.9,
            15: 1.0, 16: 0.9, 17: 0.7, 18: 0.5, 19: 0.4, 20: 0.5, 21: 0.6,
            22: 0.7, 23: 0.5, 0: 0.3,
        },
        "transitions": {
            "ChatGPT": ["Docs", "Gmail", "Translate"],
            "Gmail": ["Calendar", "ChatGPT", "Docs"],
            "Docs": ["ChatGPT", "Gmail", "WhatsApp"],
        },
    },
}


class SyntheticDataGenerator:
    """Generates realistic app-switch event sequences for simulation."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.profiles = APP_PROFILES

    def generate_persona_events(
        self,
        persona_name: str,
        n_events: int = 10000,
        ram_total_mb: float = 6144,
        start_date: datetime = datetime(2026, 1, 1, 8, 0, 0),
    ) -> pd.DataFrame:
        """Generate app-switch events for a single persona."""
        persona = PERSONAS[persona_name]
        apps = list(persona["app_weights"].keys())
        weights = list(persona["app_weights"].values())
        transitions = persona.get("transitions", {})
        hourly = persona["hourly_activity"]

        events = []
        current_time = start_date
        prev_apps = ["", "", ""]
        current_app = self.rng.choices(apps, weights=weights, k=1)[0]
        battery = 0.85 + self.rng.uniform(-0.1, 0.1)
        session_depth = 0

        for i in range(n_events):
            # ── Advance time ──
            hour = current_time.hour
            activity = hourly.get(hour, 0.1)
            # Higher activity → shorter gaps between switches
            gap_seconds = max(5, self.rng.expovariate(activity * 0.02))
            gap_seconds = min(gap_seconds, 1800)  # cap at 30 min
            current_time += timedelta(seconds=gap_seconds)
            hour = current_time.hour
            dow = current_time.weekday()

            # ── Battery simulation ──
            battery -= gap_seconds * 0.00002  # slow drain
            is_charging = False
            if battery < 0.15:
                is_charging = True
                battery = min(1.0, battery + 0.01)
            if self.rng.random() < 0.05:
                is_charging = True
                battery = min(1.0, battery + 0.05)
            battery = max(0.05, min(1.0, battery))

            # ── RAM simulation ──
            ram_base_free = ram_total_mb * 0.35  # ~35% free on average
            ram_noise = self.rng.gauss(0, ram_total_mb * 0.08)
            ram_free = max(200, min(ram_total_mb * 0.6, ram_base_free + ram_noise))

            # ── Network ──
            network = self.rng.choices(
                ["wifi", "cellular", "none"], weights=[0.6, 0.35, 0.05], k=1
            )[0]

            # ── AI task ──
            ai_apps = {"Camera", "ChatGPT", "Translate", "Instagram", "TikTok", "YouTube"}
            ai_active = current_app in ai_apps and self.rng.random() < 0.6

            # ── Choose next app ──
            # Blend: 40% transition-based, 40% persona-weighted, 20% hourly-influenced
            next_app = self._choose_next_app(
                current_app, apps, weights, transitions, hour, dow, persona_name
            )

            # ── Build event ──
            profile = get_profile(current_app)
            events.append({
                "event_id": i,
                "session_id": f"{persona_name}_{start_date.strftime('%Y%m%d')}",
                "persona": persona_name,
                "timestamp": current_time.isoformat(),
                "current_app": current_app,
                "next_app": next_app,
                "prev_app_1": prev_apps[0],
                "prev_app_2": prev_apps[1],
                "prev_app_3": prev_apps[2],
                "hour": hour,
                "day_of_week": dow,
                "battery": round(battery, 3),
                "is_charging": is_charging,
                "ram_free_mb": round(ram_free, 1),
                "ram_total_mb": ram_total_mb,
                "network_type": network,
                "ai_task_active": ai_active,
                "app_category": profile.category,
                "session_depth": session_depth,
                "time_since_last_switch": round(gap_seconds, 1),
            })

            # ── Update state ──
            prev_apps = [current_app] + prev_apps[:2]
            current_app = next_app
            session_depth += 1

            # Reset session after a big gap (>30 min)
            if gap_seconds > 1800:
                session_depth = 0

        return pd.DataFrame(events)

    def _choose_next_app(
        self, current: str, apps: list, weights: list,
        transitions: dict, hour: int, dow: int, persona: str
    ) -> str:
        """Blended next-app selection: transitions + persona weights + temporal patterns."""
        roll = self.rng.random()

        # 40% chance: use transition patterns (sequential dependency)
        if roll < 0.40 and current in transitions:
            candidates = transitions[current]
            # Pick from transition candidates with some noise
            return self.rng.choice(candidates)

        # 20% chance: temporal influence (different apps at different times)
        if roll < 0.60:
            return self._temporal_choice(apps, weights, hour, dow)

        # 40% chance: persona base distribution
        next_app = self.rng.choices(apps, weights=weights, k=1)[0]
        # Avoid staying on the same app
        attempts = 0
        while next_app == current and attempts < 5:
            next_app = self.rng.choices(apps, weights=weights, k=1)[0]
            attempts += 1
        return next_app

    def _temporal_choice(self, apps: list, weights: list, hour: int, dow: int) -> str:
        """Adjust app probabilities based on time of day."""
        adjusted = list(weights)
        for i, app in enumerate(apps):
            profile = get_profile(app)
            # Morning commute boost for Maps, Gmail
            if 7 <= hour <= 9 and profile.category in ("navigation", "productivity"):
                adjusted[i] *= 2.0
            # Work hours boost for productivity
            elif 9 <= hour <= 17 and profile.category == "productivity":
                adjusted[i] *= 1.8
            # Evening boost for entertainment + social
            elif 19 <= hour <= 23 and profile.category in ("entertainment", "social"):
                adjusted[i] *= 2.0
            # Night boost for social + messaging
            elif hour >= 22 or hour <= 1:
                if profile.category in ("social", "messaging"):
                    adjusted[i] *= 1.5
            # Weekend boost for gaming + entertainment
            if dow >= 5 and profile.category in ("gaming", "entertainment"):
                adjusted[i] *= 1.5

        total = sum(adjusted)
        adjusted = [w / total for w in adjusted]
        return self.rng.choices(apps, weights=adjusted, k=1)[0]

    def generate_all(
        self,
        events_per_persona: int = 10000,
        ram_total_mb: float = 6144,
    ) -> pd.DataFrame:
        """Generate events for ALL 5 personas and combine."""
        all_dfs = []
        for persona in PERSONAS:
            print(f"  Generating {events_per_persona} events for '{persona}'...")
            df = self.generate_persona_events(persona, events_per_persona, ram_total_mb)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"  Total events generated: {len(combined)}")
        return combined
