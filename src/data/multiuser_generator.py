"""
Multi-User Multi-Day Data Generator — Phase 2
Generates research-grade app-switch events for 10 distinct user profiles
spanning 7-14 simulated days each, with:
  - Circadian rhythms (per-hour activity curves)
  - Morning/evening habitual routines (70-80% repeatability)
  - Weekend vs weekday behavioral shifts
  - Realistic battery drain-charge cycles
  - Dynamic RAM pressure based on app memory footprints
  - Concept drift after day 5 (gradual preference shift)
  - Inter-session gaps following exponential distribution

Grounded in published statistics:
  - Avg Indian smartphone usage: 6.5 hrs/day (Statista 2024)
  - Avg app switches: 65-70/day (IAMAI Report)
  - Peak usage: 8-10 PM IST (Google Digital Wellbeing 2023)
"""
import json
import math
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# Import app profiles for memory footprints and categories
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.app_profiles import get_profile, APP_PROFILES


# ═══════════════════════════════════════════════════════════════
# EXTENDED APP PROFILES — apps used by profiles but not in
# original 20 apps (add fallback profiles for them)
# ═══════════════════════════════════════════════════════════════
EXTENDED_APPS = {
    "Chrome": {"category": "browser", "memory_mb": 150, "cold_launch_ms": 1800},
    "Slack": {"category": "productivity", "memory_mb": 130, "cold_launch_ms": 2000},
    "LinkedIn": {"category": "productivity", "memory_mb": 120, "cold_launch_ms": 1700},
    "Discord": {"category": "social", "memory_mb": 140, "cold_launch_ms": 2200},
    "Pinterest": {"category": "social", "memory_mb": 130, "cold_launch_ms": 1900},
    "Snapchat": {"category": "social", "memory_mb": 160, "cold_launch_ms": 2100},
    "GitHub": {"category": "productivity", "memory_mb": 100, "cold_launch_ms": 1500},
    "Gallery": {"category": "utility", "memory_mb": 80, "cold_launch_ms": 800},
    "Flipkart": {"category": "shopping", "memory_mb": 140, "cold_launch_ms": 2000},
}


def get_app_category(app_name: str) -> str:
    """Get category for any app including extended ones."""
    if app_name in APP_PROFILES:
        return APP_PROFILES[app_name].category
    if app_name in EXTENDED_APPS:
        return EXTENDED_APPS[app_name]["category"]
    return "other"


def get_app_memory(app_name: str) -> float:
    """Get memory footprint in MB."""
    if app_name in APP_PROFILES:
        return APP_PROFILES[app_name].total_memory_mb
    if app_name in EXTENDED_APPS:
        return EXTENDED_APPS[app_name]["memory_mb"]
    return 100.0


# ═══════════════════════════════════════════════════════════════
# CIRCADIAN ACTIVITY CURVES
# Based on aggregated Indian smartphone usage patterns
# Values are relative activity levels 0.0-1.0 per hour
# ═══════════════════════════════════════════════════════════════
CIRCADIAN_CURVES = {
    "student": {
        "weekday": {
            0: 0.3, 1: 0.1, 2: 0.05, 3: 0.02, 4: 0.01, 5: 0.02,
            6: 0.1, 7: 0.3, 8: 0.5, 9: 0.4, 10: 0.4, 11: 0.5,
            12: 0.7, 13: 0.6, 14: 0.4, 15: 0.4, 16: 0.5, 17: 0.6,
            18: 0.7, 19: 0.8, 20: 0.9, 21: 1.0, 22: 0.9, 23: 0.6,
        },
        "weekend": {
            0: 0.4, 1: 0.2, 2: 0.1, 3: 0.02, 4: 0.01, 5: 0.01,
            6: 0.05, 7: 0.1, 8: 0.2, 9: 0.4, 10: 0.6, 11: 0.7,
            12: 0.7, 13: 0.7, 14: 0.6, 15: 0.6, 16: 0.7, 17: 0.7,
            18: 0.8, 19: 0.9, 20: 1.0, 21: 1.0, 22: 0.9, 23: 0.7,
        },
    },
    "professional": {
        "weekday": {
            0: 0.1, 1: 0.05, 2: 0.02, 3: 0.01, 4: 0.01, 5: 0.1,
            6: 0.4, 7: 0.6, 8: 0.8, 9: 1.0, 10: 0.9, 11: 0.8,
            12: 0.6, 13: 0.7, 14: 0.9, 15: 0.8, 16: 0.7, 17: 0.5,
            18: 0.4, 19: 0.5, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.1,
        },
        "weekend": {
            0: 0.1, 1: 0.05, 2: 0.02, 3: 0.01, 4: 0.01, 5: 0.02,
            6: 0.1, 7: 0.2, 8: 0.4, 9: 0.5, 10: 0.5, 11: 0.5,
            12: 0.6, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.6, 17: 0.7,
            18: 0.8, 19: 0.9, 20: 0.9, 21: 0.8, 22: 0.5, 23: 0.2,
        },
    },
    "power_user": {
        "weekday": {
            0: 0.4, 1: 0.2, 2: 0.1, 3: 0.05, 4: 0.02, 5: 0.02,
            6: 0.05, 7: 0.1, 8: 0.2, 9: 0.4, 10: 0.5, 11: 0.6,
            12: 0.6, 13: 0.5, 14: 0.6, 15: 0.7, 16: 0.8, 17: 0.8,
            18: 0.9, 19: 1.0, 20: 1.0, 21: 0.9, 22: 0.8, 23: 0.6,
        },
        "weekend": {
            0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05, 4: 0.02, 5: 0.02,
            6: 0.05, 7: 0.1, 8: 0.2, 9: 0.3, 10: 0.5, 11: 0.7,
            12: 0.7, 13: 0.7, 14: 0.8, 15: 0.8, 16: 0.9, 17: 0.9,
            18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.9, 23: 0.7,
        },
    },
    "casual": {
        "weekday": {
            0: 0.1, 1: 0.05, 2: 0.02, 3: 0.01, 4: 0.01, 5: 0.05,
            6: 0.2, 7: 0.3, 8: 0.3, 9: 0.3, 10: 0.3, 11: 0.4,
            12: 0.5, 13: 0.5, 14: 0.4, 15: 0.4, 16: 0.5, 17: 0.6,
            18: 0.7, 19: 0.8, 20: 0.9, 21: 0.8, 22: 0.5, 23: 0.2,
        },
        "weekend": {
            0: 0.1, 1: 0.05, 2: 0.02, 3: 0.01, 4: 0.01, 5: 0.02,
            6: 0.1, 7: 0.2, 8: 0.3, 9: 0.4, 10: 0.5, 11: 0.5,
            12: 0.6, 13: 0.6, 14: 0.5, 15: 0.5, 16: 0.6, 17: 0.7,
            18: 0.8, 19: 0.9, 20: 0.9, 21: 0.8, 22: 0.5, 23: 0.2,
        },
    },
    "developer": {
        "weekday": {
            0: 0.3, 1: 0.1, 2: 0.05, 3: 0.02, 4: 0.01, 5: 0.02,
            6: 0.05, 7: 0.1, 8: 0.4, 9: 0.7, 10: 0.9, 11: 1.0,
            12: 0.5, 13: 0.6, 14: 0.9, 15: 1.0, 16: 0.9, 17: 0.7,
            18: 0.5, 19: 0.4, 20: 0.5, 21: 0.6, 22: 0.7, 23: 0.5,
        },
        "weekend": {
            0: 0.3, 1: 0.1, 2: 0.05, 3: 0.02, 4: 0.01, 5: 0.01,
            6: 0.05, 7: 0.1, 8: 0.2, 9: 0.3, 10: 0.5, 11: 0.6,
            12: 0.6, 13: 0.6, 14: 0.5, 15: 0.5, 16: 0.6, 17: 0.7,
            18: 0.8, 19: 0.8, 20: 0.7, 21: 0.6, 22: 0.5, 23: 0.4,
        },
    },
}


class MultiUserGenerator:
    """
    Generates multi-day, multi-person app-switch event data
    with realistic temporal patterns, battery cycles, and RAM dynamics.
    """

    def __init__(self, profiles_path: str, seed: int = 42):
        with open(profiles_path, "r") as f:
            data = json.load(f)
        self.user_profiles = data["users"]
        self.rng = random.Random(seed)
        self.start_date = datetime(2026, 9, 1, 0, 0, 0)  # Monday Sept 1, 2026

    def generate_all(self) -> pd.DataFrame:
        """Generate events for all users and combine."""
        all_dfs = []
        for profile in self.user_profiles:
            uid = profile["user_id"]
            days = profile["sim_days"]
            print(f"  Generating {days} days for {uid} ({profile['persona']}, "
                  f"{profile['device_model']}, {profile['device_ram_gb']}GB RAM)...")
            df = self._generate_user(profile)
            all_dfs.append(df)
            print(f"    -> {len(df)} events, {df['is_switch'].sum()} switches, "
                  f"{df['current_app'].nunique()} unique apps")

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n  Total: {len(combined)} events across {len(self.user_profiles)} users")
        print(f"  Total switches: {combined['is_switch'].sum()}")
        print(f"  Date range: {combined['timestamp'].min()[:10]} -> "
              f"{combined['timestamp'].max()[:10]}")
        return combined

    def _generate_user(self, profile: dict) -> pd.DataFrame:
        """Generate multi-day events for a single user."""
        user_id = profile["user_id"]
        persona = profile["persona"]
        device_model = profile["device_model"]
        ram_total_mb = profile["device_ram_gb"] * 1024
        sim_days = profile["sim_days"]
        apps = list(profile["app_weights"].keys())
        base_weights = list(profile["app_weights"].values())
        transitions = profile.get("transitions", {})
        morning_routine = profile.get("morning_routine", [])
        evening_routine = profile.get("evening_routine", [])
        wake_wd = profile.get("wake_hour_weekday", 7)
        wake_we = profile.get("wake_hour_weekend", 9)
        sleep_hour = profile.get("sleep_hour", 23)
        weekend_boost = profile.get("weekend_boost", {})
        weekend_suppress = profile.get("weekend_suppress", {})

        events = []
        global_event_id = 0
        prev_apps = ["", "", ""]
        session_depth = 0

        for day_num in range(sim_days):
            day_date = self.start_date + timedelta(days=day_num)
            is_weekend = day_date.weekday() >= 5
            wake_hour = wake_we if is_weekend else wake_wd

            # Battery starts high after overnight charging
            battery = 0.90 + self.rng.uniform(-0.05, 0.08)
            battery = min(1.0, battery)
            was_charging_overnight = True

            # RAM resets each day (simulating device standby cleanup)
            ram_free = ram_total_mb * (0.40 + self.rng.uniform(-0.05, 0.10))
            recent_apps_memory = {}  # track RAM consumed by recently opened apps

            # Get circadian curve
            curve_type = "weekend" if is_weekend else "weekday"
            circadian = CIRCADIAN_CURVES.get(persona, CIRCADIAN_CURVES["casual"])
            hourly_curve = circadian.get(curve_type, circadian["weekday"])

            # Concept drift: after day 5, gradually shift preferences
            drift_factor = max(0, (day_num - 5) / sim_days) * 0.15
            drifted_weights = self._apply_drift(apps, base_weights, drift_factor)

            # Weekend modifier
            if is_weekend:
                drifted_weights = self._apply_weekend_modifier(
                    apps, drifted_weights, weekend_boost, weekend_suppress
                )

            # Morning routine
            current_time = day_date.replace(hour=wake_hour, minute=self.rng.randint(0, 30))
            current_app = None

            # Execute morning routine (70-80% chance of following it)
            if morning_routine and self.rng.random() < 0.75:
                for routine_app in morning_routine:
                    if routine_app not in apps:
                        continue
                    gap = self.rng.uniform(30, 180)  # 30s to 3min between routine apps
                    current_time += timedelta(seconds=gap)

                    prev_app = current_app
                    current_app = routine_app
                    is_switch = prev_app is not None and prev_app != current_app

                    # Battery drain
                    battery -= gap * 0.00003
                    battery = max(0.05, battery)

                    # RAM update
                    app_mem = get_app_memory(current_app)
                    recent_apps_memory[current_app] = app_mem
                    total_used = sum(recent_apps_memory.values())
                    ram_free = max(200, ram_total_mb - total_used -
                                  ram_total_mb * 0.30)  # OS overhead ~30%

                    event = self._build_event(
                        global_event_id, user_id, persona, device_model,
                        ram_total_mb, day_num + 1, current_time, current_app,
                        prev_apps, battery, False, ram_free, ram_total_mb,
                        is_switch, session_depth, gap
                    )
                    events.append(event)
                    global_event_id += 1

                    if is_switch:
                        prev_apps = [prev_app or ""] + prev_apps[:2]
                        session_depth += 1

            # Main day simulation
            if current_app is None:
                current_app = self.rng.choices(apps, weights=drifted_weights, k=1)[0]

            # Simulate until sleep hour
            if sleep_hour < wake_hour:
                # Sleeps past midnight (e.g., sleep at 1 AM)
                end_time = day_date.replace(hour=23, minute=59)
                end_time_next = day_date + timedelta(days=1)
                end_time_next = end_time_next.replace(hour=sleep_hour, minute=0)
            else:
                end_time = day_date.replace(hour=sleep_hour, minute=0)
                end_time_next = None

            is_charging = False
            last_charge_check = battery

            while current_time < end_time or (end_time_next and current_time < end_time_next):
                if current_time >= end_time and end_time_next:
                    # Past midnight, check against next-day sleep time
                    if current_time >= end_time_next:
                        break

                hour = current_time.hour
                activity = hourly_curve.get(hour, 0.1)

                # Gap between events (exponential, modulated by activity level)
                # Higher activity → shorter gaps
                lambda_param = activity * 0.025 + 0.005
                gap = max(5, self.rng.expovariate(lambda_param))
                gap = min(gap, 2400)  # cap at 40 min

                # Very low activity hours → much longer gaps
                if activity < 0.1:
                    gap = max(gap, 600)  # at least 10 min gaps during sleep-like hours

                current_time += timedelta(seconds=gap)
                hour = current_time.hour
                dow = current_time.weekday()

                # Battery simulation (non-linear drain)
                app_category = get_app_category(current_app)
                drain_rate = 0.00002  # base drain per second
                if app_category in ("gaming",):
                    drain_rate = 0.00006  # 3x drain for gaming
                elif app_category in ("entertainment",):
                    drain_rate = 0.00004  # 2x for video streaming
                elif app_category in ("camera",):
                    drain_rate = 0.00005  # camera + AI processing

                battery -= gap * drain_rate

                # Charging logic
                if battery < 0.15 and not is_charging:
                    is_charging = True
                if is_charging:
                    battery += gap * 0.00008  # charging rate
                    if battery >= 0.90:
                        is_charging = False
                battery = max(0.03, min(1.0, battery))

                # Network
                network = self.rng.choices(
                    ["wifi", "cellular", "none"],
                    weights=[0.55, 0.40, 0.05], k=1
                )[0]

                # AI task
                ai_apps = {"Camera", "ChatGPT", "Translate", "Instagram", "TikTok",
                           "YouTube", "Snapchat"}
                ai_active = current_app in ai_apps and self.rng.random() < 0.5

                # Choose next app (blended strategy)
                next_app = self._choose_next_app(
                    current_app, apps, drifted_weights, transitions,
                    hour, dow, evening_routine, is_weekend
                )

                is_switch = next_app != current_app

                # RAM dynamics
                app_mem = get_app_memory(next_app)
                recent_apps_memory[next_app] = app_mem

                # Evict old apps from RAM tracking if too many (simulate OS eviction)
                if len(recent_apps_memory) > 8:
                    # Remove the oldest entry (crude LRU)
                    oldest_key = list(recent_apps_memory.keys())[0]
                    del recent_apps_memory[oldest_key]

                total_used = sum(recent_apps_memory.values())
                ram_free = max(150, ram_total_mb - total_used -
                              ram_total_mb * 0.30)
                ram_noise = self.rng.gauss(0, ram_total_mb * 0.03)
                ram_free = max(150, min(ram_total_mb * 0.55, ram_free + ram_noise))

                event = self._build_event(
                    global_event_id, user_id, persona, device_model,
                    ram_total_mb, day_num + 1, current_time, current_app,
                    prev_apps, battery, is_charging, ram_free, ram_total_mb,
                    is_switch, session_depth, gap,
                    network=network, ai_active=ai_active, next_app=next_app
                )
                events.append(event)
                global_event_id += 1

                if is_switch:
                    prev_apps = [current_app] + prev_apps[:2]
                    session_depth += 1

                current_app = next_app

                # Session reset after long gap
                if gap > 1800:
                    session_depth = 0

        # Post-process: fill next_app for events that don't have it
        for i in range(len(events) - 1):
            if not events[i].get("next_app"):
                events[i]["next_app"] = events[i + 1]["current_app"]
        if events:
            if not events[-1].get("next_app"):
                events[-1]["next_app"] = events[-1]["current_app"]

        return pd.DataFrame(events)

    def _build_event(
        self, event_id, user_id, persona, device_model, ram_total_mb,
        day_number, timestamp, current_app, prev_apps, battery,
        is_charging, ram_free, ram_total, is_switch, session_depth,
        gap_seconds, network="wifi", ai_active=False, next_app=None
    ) -> dict:
        """Build a single event dictionary."""
        category = get_app_category(current_app)
        return {
            "event_id": event_id,
            "user_id": user_id,
            "persona": persona,
            "device_model": device_model,
            "device_ram_gb": ram_total_mb / 1024,
            "day_number": day_number,
            "session_id": f"{user_id}_day{day_number}",
            "timestamp": timestamp.isoformat(),
            "current_app": current_app,
            "next_app": next_app or "",
            "prev_app_1": prev_apps[0] if len(prev_apps) > 0 else "",
            "prev_app_2": prev_apps[1] if len(prev_apps) > 1 else "",
            "prev_app_3": prev_apps[2] if len(prev_apps) > 2 else "",
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "battery": round(battery, 3),
            "is_charging": is_charging,
            "ram_free_mb": round(ram_free, 1),
            "ram_total_mb": ram_total,
            "network_type": network,
            "ai_task_active": ai_active,
            "app_category": category,
            "session_depth": session_depth,
            "time_since_last_switch": round(gap_seconds, 1),
            "is_switch": int(is_switch),
        }

    def _choose_next_app(
        self, current: str, apps: list, weights: list,
        transitions: dict, hour: int, dow: int,
        evening_routine: list, is_weekend: bool
    ) -> str:
        """Blended next-app selection with temporal awareness."""
        roll = self.rng.random()

        # 35% chance: transition patterns (sequential dependency)
        if roll < 0.35 and current in transitions:
            candidates = transitions[current]
            valid = [c for c in candidates if c in apps]
            if valid:
                return self.rng.choice(valid)

        # 15% chance: evening routine (if it's evening hours 19-23)
        if roll < 0.50 and 19 <= hour <= 23 and evening_routine:
            valid = [r for r in evening_routine if r in apps and r != current]
            if valid:
                return self.rng.choice(valid)

        # 20% chance: temporal influence
        if roll < 0.70:
            return self._temporal_choice(apps, weights, hour, dow, is_weekend)

        # 30% chance: persona base distribution (avoid same app)
        next_app = self.rng.choices(apps, weights=weights, k=1)[0]
        attempts = 0
        while next_app == current and attempts < 5:
            next_app = self.rng.choices(apps, weights=weights, k=1)[0]
            attempts += 1
        return next_app

    def _temporal_choice(
        self, apps: list, weights: list, hour: int, dow: int,
        is_weekend: bool
    ) -> str:
        """Adjust app probabilities based on time of day and day of week."""
        adjusted = list(weights)
        for i, app in enumerate(apps):
            cat = get_app_category(app)

            # Morning commute (6-9): Maps, Gmail, productivity
            if 6 <= hour <= 9 and cat in ("navigation", "productivity"):
                adjusted[i] *= 2.0
            # Work hours (9-17): productivity, browser
            elif 9 <= hour <= 17 and cat in ("productivity", "browser"):
                adjusted[i] *= 1.8
            # Lunch (12-13): social, entertainment spike
            elif 12 <= hour <= 13 and cat in ("social", "entertainment"):
                adjusted[i] *= 1.5
            # Evening (18-22): entertainment, social, gaming
            elif 18 <= hour <= 22 and cat in ("entertainment", "social", "gaming"):
                adjusted[i] *= 2.0
            # Late night (22+): messaging, social
            elif hour >= 22 or hour <= 1:
                if cat in ("messaging", "social"):
                    adjusted[i] *= 1.5

            # Weekend gaming/entertainment boost
            if is_weekend and cat in ("gaming", "entertainment"):
                adjusted[i] *= 1.4

        total = sum(adjusted)
        if total > 0:
            adjusted = [w / total for w in adjusted]
        return self.rng.choices(apps, weights=adjusted, k=1)[0]

    def _apply_drift(
        self, apps: list, weights: list, drift_factor: float
    ) -> list:
        """Apply concept drift: gradually shift preferences over days."""
        if drift_factor <= 0:
            return list(weights)

        drifted = list(weights)
        n = len(apps)
        for i in range(n):
            # Random perturbation scaled by drift_factor
            perturbation = self.rng.gauss(0, drift_factor)
            drifted[i] = max(0.005, drifted[i] + perturbation)

        total = sum(drifted)
        return [w / total for w in drifted]

    def _apply_weekend_modifier(
        self, apps: list, weights: list,
        boost: dict, suppress: dict
    ) -> list:
        """Apply weekend-specific boosts and suppressions."""
        modified = list(weights)
        for i, app in enumerate(apps):
            cat = get_app_category(app)
            if cat in boost:
                modified[i] *= boost[cat]
            if cat in suppress:
                modified[i] *= suppress[cat]

        total = sum(modified)
        if total > 0:
            modified = [w / total for w in modified]
        return modified
