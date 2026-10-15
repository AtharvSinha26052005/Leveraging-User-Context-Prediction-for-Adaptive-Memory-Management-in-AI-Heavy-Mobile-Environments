"""
generate_multiuser_data.py — CLI script to generate multi-person multi-day dataset.

Usage: python scripts/generate_multiuser_data.py
Output: data/multiuser/multiuser_events.csv
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.multiuser_generator import MultiUserGenerator


def main():
    print("=" * 70)
    print("  MULTI-USER MULTI-DAY DATA GENERATOR — Phase 2")
    print("  10 users × 7-14 days × realistic temporal patterns")
    print("=" * 70)

    start = time.time()

    profiles_path = os.path.join("data", "multiuser", "user_profiles.json")
    if not os.path.isfile(profiles_path):
        print(f"ERROR: Profiles file not found: {profiles_path}")
        sys.exit(1)

    gen = MultiUserGenerator(profiles_path, seed=42)
    df = gen.generate_all()

    # Save
    output_path = os.path.join("data", "multiuser", "multiuser_events.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start

    # Summary stats
    print(f"\n{'=' * 70}")
    print(f"  GENERATION COMPLETE")
    print(f"  Total events:     {len(df):,}")
    print(f"  Total switches:   {df['is_switch'].sum():,}")
    print(f"  Unique users:     {df['user_id'].nunique()}")
    print(f"  Unique apps:      {df['current_app'].nunique()}")
    print(f"  Date range:       {df['timestamp'].min()[:10]} -> {df['timestamp'].max()[:10]}")
    print(f"  Time taken:       {elapsed:.1f}s")
    print(f"  Saved to:         {output_path}")
    print(f"{'=' * 70}")

    # Per-user breakdown
    print(f"\n  Per-User Breakdown:")
    print(f"  {'User':<18} {'Persona':<14} {'Device':<20} {'RAM':>5} {'Days':>5} {'Events':>8} {'Switches':>9} {'Apps':>5}")
    print(f"  {'-'*94}")
    for uid in df["user_id"].unique():
        udf = df[df["user_id"] == uid]
        row = udf.iloc[0]
        print(f"  {uid:<18} {row['persona']:<14} {row['device_model']:<20} "
              f"{row['device_ram_gb']:>4.0f}G {udf['day_number'].nunique():>5} "
              f"{len(udf):>8,} {udf['is_switch'].sum():>9,} "
              f"{udf['current_app'].nunique():>5}")


if __name__ == "__main__":
    main()
