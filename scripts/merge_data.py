"""
merge_data.py — Merge all data sources into unified train/test split.

Sources merged:
  1. data/synthetic_events.csv      (50,000 events, 5 personas)
  2. data/multiuser/multiuser_events.csv  (131,179 events, 10 users)
  3. data/self_collected/*.csv       (friends/self data if available)

Output:
  data/merged/train_events.csv  (70%)
  data/merged/test_events.csv   (30%)
  data/merged/merge_report.json (stats)

Usage: python scripts/merge_data.py
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Canonical schema — all sources normalised to these columns
CANONICAL_COLS = [
    "user_id", "persona", "session_id", "timestamp",
    "current_app", "next_app",
    "prev_app_1", "prev_app_2", "prev_app_3",
    "hour", "day_of_week", "battery", "is_charging",
    "ram_free_mb", "ram_total_mb", "network_type",
    "ai_task_active", "app_category",
    "session_depth", "time_since_last_switch",
]

APP_CATEGORY_MAP = {
    "Chrome": "browser", "Slack": "productivity", "LinkedIn": "productivity",
    "Discord": "social", "Pinterest": "social", "Snapchat": "social",
    "GitHub": "productivity", "Gallery": "utility", "Flipkart": "shopping",
}


def _normalise_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise original synthetic_events.csv to canonical schema."""
    df = df.copy()
    # synthetic has 'persona' but no 'user_id' — create one
    if "user_id" not in df.columns:
        df["user_id"] = "synth_" + df["persona"].astype(str)
    return df


def _normalise_multiuser(df: pd.DataFrame) -> pd.DataFrame:
    """Multiuser already has all columns — just return as-is."""
    return df.copy()


def _normalise_self_collected(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalise friend/self collected CSVs to canonical schema."""
    df = df.copy()
    if "user_id" not in df.columns:
        df["user_id"] = source_name
    if "persona" not in df.columns:
        df["persona"] = "real_user"
    if "prev_app_1" not in df.columns:
        df["prev_app_1"] = ""
    if "prev_app_2" not in df.columns:
        df["prev_app_2"] = ""
    if "prev_app_3" not in df.columns:
        df["prev_app_3"] = ""
    if "session_depth" not in df.columns:
        df["session_depth"] = 0
    if "time_since_last_switch" not in df.columns:
        df["time_since_last_switch"] = 30.0
    if "network_type" not in df.columns:
        df["network_type"] = "wifi"
    if "ai_task_active" not in df.columns:
        df["ai_task_active"] = False
    if "app_category" not in df.columns:
        df["app_category"] = df["current_app"].map(APP_CATEGORY_MAP).fillna("other")
    if "next_app" not in df.columns:
        df["next_app"] = df["current_app"].shift(-1).fillna(df["current_app"])
    return df


def merge_all() -> pd.DataFrame:
    frames = []
    report = {"sources": []}

    # ── Source 1: Synthetic ──
    synth_path = "data/synthetic_events.csv"
    if os.path.isfile(synth_path):
        df_s = pd.read_csv(synth_path)
        df_s = _normalise_synthetic(df_s)
        frames.append(df_s)
        report["sources"].append({"name": "synthetic", "events": len(df_s)})
        print(f"  [1] Synthetic:   {len(df_s):>8,} events")
    else:
        print(f"  [1] Synthetic:   NOT FOUND (run python scripts/run_all.py first)")

    # ── Source 2: Multiuser ──
    multi_path = "data/multiuser/multiuser_events.csv"
    if os.path.isfile(multi_path):
        df_m = pd.read_csv(multi_path)
        df_m = _normalise_multiuser(df_m)
        frames.append(df_m)
        report["sources"].append({"name": "multiuser", "events": len(df_m)})
        print(f"  [2] Multiuser:   {len(df_m):>8,} events ({df_m['user_id'].nunique()} users)")
    else:
        print(f"  [2] Multiuser:   NOT FOUND (run scripts/generate_multiuser_data.py first)")

    # ── Source 3: Self-collected (phone + laptop CSVs) ──
    collected_paths = (
        glob.glob("data/self_collected/*.csv") +
        glob.glob("data/self_collected/phone_events*.csv") +
        glob.glob("data/self_collected/laptop_events*.csv")
    )
    collected_paths = list(set(collected_paths))
    if collected_paths:
        for path in sorted(collected_paths):
            try:
                df_c = pd.read_csv(path)
                name = os.path.splitext(os.path.basename(path))[0]
                df_c = _normalise_self_collected(df_c, name)
                frames.append(df_c)
                report["sources"].append({"name": name, "events": len(df_c)})
                print(f"  [3] {name:<20} {len(df_c):>6,} events")
            except Exception as e:
                print(f"  [!] Skipped {path}: {e}")
    else:
        print(f"  [3] Self-collected: none found (optional)")

    if not frames:
        raise RuntimeError("No data sources found.")

    # ── Combine ──
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # ── Fill missing columns ──
    for col in CANONICAL_COLS:
        if col not in combined.columns:
            combined[col] = "" if combined.dtypes.get(col, "O") == object else 0

    combined["current_app"] = combined["current_app"].fillna("").astype(str)
    combined["next_app"] = combined["next_app"].fillna("").astype(str)
    combined["prev_app_1"] = combined["prev_app_1"].fillna("").astype(str)
    combined["prev_app_2"] = combined["prev_app_2"].fillna("").astype(str)
    combined["prev_app_3"] = combined["prev_app_3"].fillna("").astype(str)
    combined["battery"] = combined["battery"].clip(0.0, 1.0)
    combined["ram_free_mb"] = combined["ram_free_mb"].clip(0)
    combined["is_charging"] = combined["is_charging"].fillna(False).astype(bool)
    combined["ai_task_active"] = combined["ai_task_active"].fillna(False).astype(bool)

    # Drop rows where current_app is empty
    combined = combined[combined["current_app"] != ""]

    # ── Temporal train/test split (80/20) ──
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], format="ISO8601")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(combined) * 0.80)
    train = combined.iloc[:split_idx].copy()
    test = combined.iloc[split_idx:].copy()

    # ── Save ──
    os.makedirs("data/merged", exist_ok=True)
    train.to_csv("data/merged/train_events.csv", index=False)
    test.to_csv("data/merged/test_events.csv", index=False)

    report.update({
        "total_events": len(combined),
        "train_events": len(train),
        "test_events": len(test),
        "unique_apps": combined["current_app"].nunique(),
        "unique_users": combined["user_id"].nunique() if "user_id" in combined else 1,
        "date_range": f"{str(combined['timestamp'].min())[:10]} to {str(combined['timestamp'].max())[:10]}",
    })
    with open("data/merged/merge_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Merged:  {len(combined):>8,} events total")
    print(f"  Train:   {len(train):>8,} events (80%)")
    print(f"  Test:    {len(test):>8,} events (20%)")
    print(f"  Apps:    {combined['current_app'].nunique()}")
    print(f"  Saved -> data/merged/")

    return combined


if __name__ == "__main__":
    print("=" * 60)
    print("  DATA MERGE — Combining All Sources")
    print("=" * 60)
    merge_all()
