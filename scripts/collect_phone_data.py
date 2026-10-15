"""
Phone Data Collector — Collects foreground app data from Android via ADB.

Prerequisites:
  1. Enable Developer Options on phone (tap Build Number 7 times)
  2. Enable USB Debugging in Developer Options
  3. Connect phone via USB
  4. Install ADB: https://developer.android.com/tools/releases/platform-tools
  5. Run: adb devices  (should show your phone)

Usage: python scripts/collect_phone_data.py --duration 30
       (collects for 30 minutes, Ctrl+C to stop early)
"""
import subprocess
import time
import csv
import os
import sys
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# App package → friendly name mapping
PACKAGE_MAP = {
    "com.whatsapp": "WhatsApp",
    "com.instagram.android": "Instagram",
    "com.facebook.katana": "Facebook",
    "com.google.android.youtube": "YouTube",
    "com.twitter.android": "Twitter",
    "com.snapchat.android": "Snapchat",
    "com.spotify.music": "Spotify",
    "com.google.android.apps.maps": "Google Maps",
    "com.google.android.gm": "Gmail",
    "com.google.android.apps.docs": "Google Docs",
    "com.google.android.calendar": "Google Calendar",
    "com.android.chrome": "Chrome",
    "org.telegram.messenger": "Telegram",
    "com.discord": "Discord",
    "com.amazon.mShop.android.shopping": "Amazon",
    "com.flipkart.android": "Flipkart",
    "in.amazon.mShop.android.shopping": "Amazon",
    "com.phonepe.app": "PhonePe",
    "net.one97.paytm": "Paytm",
    "com.google.android.apps.nbu.paisa.user": "Google Pay",
    "com.zhiliaoapp.musically": "TikTok",
    "com.linkedin.android": "LinkedIn",
    "com.microsoft.teams": "MS Teams",
    "com.google.android.apps.meetings": "Google Meet",
    "us.zoom.videomeetings": "Zoom",
    "com.netflix.mediaclient": "Netflix",
    "in.startv.hotstar": "Hotstar",
    "com.reddit.frontpage": "Reddit",
    "com.android.vending": "Play Store",
    "com.android.settings": "Settings",
    "com.android.dialer": "Phone",
    "com.android.mms": "Messages",
    "com.android.camera": "Camera",
    "com.coloros.launcher": "Home Screen",
    "com.realme.launcher": "Home Screen",
    "com.android.launcher3": "Home Screen",
    "com.oplus.battery": "Battery",
    "com.heytap.browser": "Realme Browser",
    "com.coloros.filemanager": "File Manager",
    "com.battlegroundsmobilegame": "BGMI",
    "com.garena.game.codm": "COD Mobile",
    "com.supercell.clashofclans": "Clash of Clans",
}

# Category mapping
CATEGORY_MAP = {
    "WhatsApp": "messaging", "Instagram": "social", "Facebook": "social",
    "YouTube": "entertainment", "Twitter": "social", "Snapchat": "social",
    "Spotify": "entertainment", "Google Maps": "navigation", "Gmail": "productivity",
    "Google Docs": "productivity", "Google Calendar": "productivity",
    "Chrome": "browser", "Telegram": "messaging", "Discord": "social",
    "Amazon": "shopping", "Flipkart": "shopping", "PhonePe": "finance",
    "Paytm": "finance", "Google Pay": "finance", "TikTok": "entertainment",
    "LinkedIn": "productivity", "MS Teams": "productivity",
    "Google Meet": "productivity", "Zoom": "productivity",
    "Netflix": "entertainment", "Hotstar": "entertainment", "Reddit": "social",
    "Play Store": "utility", "Settings": "system", "Phone": "system",
    "Messages": "messaging", "Camera": "utility", "Home Screen": "system",
    "BGMI": "gaming", "COD Mobile": "gaming", "Clash of Clans": "gaming",
}


def get_foreground_app():
    """Get the current foreground app using ADB."""
    try:
        # Method 1: dumpsys activity (most reliable)
        adb_path = os.path.join(os.environ["USERPROFILE"], "Desktop", "platform-tools", "adb.exe")
        result = subprocess.run(
            [adb_path, "shell", "dumpsys", "activity", "activities"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout

        # Look for mResumedActivity, topResumedActivity, or ResumedActivity
        for line in output.split("\n"):
            if "ResumedActivity" in line or "mFocusedActivity" in line:
                # Extract package name: com.example.app/.ActivityName
                match = re.search(r'([a-zA-Z][a-zA-Z0-9_.]*)/[a-zA-Z.]', line)
                if match:
                    return match.group(1)

        # Method 2: Fallback - dumpsys window
        result = subprocess.run(
            [adb_path, "shell", "dumpsys", "window", "displays"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "mCurrentFocus" in line or "mFocusedApp" in line:
                match = re.search(r'([a-zA-Z][a-zA-Z0-9_.]*)/[a-zA-Z.]', line)
                if match:
                    return match.group(1)

        return None
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print("ERROR: 'adb' not found. Install Android Platform Tools.")
        print("Download: https://developer.android.com/tools/releases/platform-tools")
        sys.exit(1)


def get_battery_info():
    """Get battery level and charging state."""
    try:
        adb_path = os.path.join(os.environ["USERPROFILE"], "Desktop", "platform-tools", "adb.exe")
        result = subprocess.run(
            [adb_path, "shell", "dumpsys", "battery"],
            capture_output=True, text=True, timeout=5
        )
        level = 50
        charging = False
        for line in result.stdout.split("\n"):
            if "level:" in line:
                level = int(re.search(r'\d+', line).group())
            if "status:" in line:
                charging = "2" in line or "5" in line  # 2=charging, 5=full
        return level, charging
    except Exception:
        return 50, False


def get_memory_info():
    """Get RAM usage info."""
    try:
        adb_path = os.path.join(os.environ["USERPROFILE"], "Desktop", "platform-tools", "adb.exe")
        result = subprocess.run(
            [adb_path, "shell", "cat", "/proc/meminfo"],
            capture_output=True, text=True, timeout=5
        )
        total_kb = 0
        free_kb = 0
        for line in result.stdout.split("\n"):
            if line.startswith("MemTotal:"):
                total_kb = int(re.search(r'\d+', line).group())
            elif line.startswith("MemAvailable:"):
                free_kb = int(re.search(r'\d+', line).group())
        return total_kb / 1024, free_kb / 1024  # Return in MB
    except Exception:
        return 8192, 4096


def package_to_name(package):
    """Convert package name to friendly app name."""
    if package in PACKAGE_MAP:
        return PACKAGE_MAP[package]
    # Extract last part of package name
    parts = package.split(".")
    return parts[-1].capitalize() if parts else package


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect phone app usage via ADB")
    parser.add_argument("--duration", type=int, default=30, help="Duration in minutes (default: 30)")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval in seconds (default: 5)")
    parser.add_argument("--output", type=str, default="data/self_collected/phone_events.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # Check ADB connection
    print("Checking ADB connection...")
    adb_path = os.path.join(os.environ["USERPROFILE"], "Desktop", "platform-tools", "adb.exe")
    result = subprocess.run([adb_path, "devices"], capture_output=True, text=True)
    devices = [l for l in result.stdout.strip().split("\n")[1:] if l.strip() and "device" in l]
    if not devices:
        print("ERROR: No device found. Make sure:")
        print("  1. USB Debugging is enabled")
        print("  2. Phone is connected via USB")
        print("  3. You accepted the USB debugging prompt on phone")
        sys.exit(1)
    print(f"Connected to: {devices[0].split()[0]}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Start collecting
    start_time = time.time()
    end_time = start_time + args.duration * 60
    events = []
    prev_app = None
    event_count = 0
    switch_count = 0
    last_save_time = start_time
    fieldnames = [
        "timestamp", "current_app", "previous_app", "next_app", "package",
        "category", "hour", "day_of_week", "battery", "is_charging",
        "ram_total_mb", "ram_free_mb", "is_switch",
    ]

    print(f"\n{'='*60}")
    print(f"  PHONE DATA COLLECTOR")
    print(f"  Duration: {args.duration} min | Interval: {args.interval}s")
    print(f"  Output: {args.output}")
    print(f"  Use your phone normally. Ctrl+C to stop early.")
    print(f"{'='*60}\n")

    try:
        while time.time() < end_time:
            package = get_foreground_app()
            if package is None:
                time.sleep(args.interval)
                continue

            app_name = package_to_name(package)
            category = CATEGORY_MAP.get(app_name, "other")
            now = datetime.now()
            battery, charging = get_battery_info()
            ram_total, ram_free = get_memory_info()

            is_switch = (app_name != prev_app) and (prev_app is not None)
            if is_switch:
                switch_count += 1

            event = {
                "timestamp": now.isoformat(),
                "current_app": app_name,
                "previous_app": prev_app or app_name,
                "next_app": "",  # filled in post-processing
                "package": package,
                "category": category,
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "battery": battery / 100.0,
                "is_charging": int(charging),
                "ram_total_mb": round(ram_total, 0),
                "ram_free_mb": round(ram_free, 0),
                "is_switch": int(is_switch),
            }
            events.append(event)
            event_count += 1

            if is_switch:
                elapsed = int(time.time() - start_time)
                print(f"  [{elapsed//60:02d}:{elapsed%60:02d}] Switch #{switch_count}: "
                      f"{prev_app} -> {app_name} "
                      f"(battery={battery}%, RAM free={ram_free:.0f}MB)")

            prev_app = app_name

            # Incremental save every 60 seconds (so data is never lost)
            if time.time() - last_save_time > 60 and events:
                _save_events(events, args.output, fieldnames)
                last_save_time = time.time()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # Post-process: fill in next_app
    for i in range(len(events) - 1):
        events[i]["next_app"] = events[i + 1]["current_app"]
    if events:
        events[-1]["next_app"] = events[-1]["current_app"]

    # Final save
    _save_events(events, args.output, fieldnames)

    if events:
        unique_apps = set(e["current_app"] for e in events)
        print(f"\n{'='*60}")
        print(f"  Collection Complete!")
        print(f"  Total polls:    {event_count}")
        print(f"  App switches:   {switch_count}")
        print(f"  Duration:       {(time.time() - start_time) / 60:.1f} minutes")
        print(f"  Unique apps:    {len(unique_apps)} ({', '.join(sorted(unique_apps))})")
        print(f"  Saved to:       {args.output}")
        print(f"{'='*60}")
    else:
        print("No events collected!")


def _save_events(events, output_path, fieldnames):
    """Save events to CSV with post-processed next_app."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


if __name__ == "__main__":
    main()

