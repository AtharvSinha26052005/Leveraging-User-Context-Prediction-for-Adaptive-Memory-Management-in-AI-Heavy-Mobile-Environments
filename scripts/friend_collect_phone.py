"""
Phone Data Collector — Collects foreground app data from Android via ADB.
PORTABLE VERSION: Send this single file to friends.

Prerequisites:
  1. Enable Developer Options on phone (tap Build Number 7 times in Settings > About Phone)
  2. Enable USB Debugging in Developer Options
  3. Connect phone via USB cable
  4. Install ADB (Android Platform Tools):
     Download: https://developer.android.com/tools/releases/platform-tools
     Extract the zip anywhere (e.g., Desktop/platform-tools)
  5. Run: adb devices  (should list your phone)

Usage:
  python friend_collect_phone.py                    # collect for 6 hours (default)
  python friend_collect_phone.py --duration 360     # collect for 6 hours
  python friend_collect_phone.py --duration 120     # collect for 2 hours
  python friend_collect_phone.py --duration 30      # collect for 30 mins (quick test)
  Press Ctrl+C to stop early at any time.

Output: phone_events.csv (in same folder as this script)
"""
import subprocess
import time
import csv
import os
import sys
import re
import shutil
from datetime import datetime


# ──────────────────────────────────────────────────────────────
# ADB DETECTION: Tries multiple locations automatically
# ──────────────────────────────────────────────────────────────
def find_adb():
    """Auto-detect ADB executable path."""
    # 1. Check if adb is in system PATH
    adb_in_path = shutil.which("adb")
    if adb_in_path:
        return adb_in_path

    # 2. Common locations on Windows
    home = os.environ.get("USERPROFILE", "")
    common_locations = [
        os.path.join(home, "Desktop", "platform-tools", "adb.exe"),
        os.path.join(home, "Downloads", "platform-tools", "adb.exe"),
        os.path.join(home, "platform-tools", "adb.exe"),
        os.path.join(home, "AppData", "Local", "Android", "Sdk", "platform-tools", "adb.exe"),
        r"C:\platform-tools\adb.exe",
        r"C:\Android\platform-tools\adb.exe",
        # Same folder as this script
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "adb.exe"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "platform-tools", "adb.exe"),
    ]

    for loc in common_locations:
        if os.path.isfile(loc):
            return loc

    return None


ADB_PATH = find_adb()
if ADB_PATH is None:
    print("=" * 60)
    print("  ERROR: ADB not found!")
    print()
    print("  Please do ONE of these:")
    print("  1. Add adb to your system PATH")
    print("  2. Place the platform-tools folder on your Desktop")
    print("  3. Place adb.exe in the same folder as this script")
    print()
    print("  Download ADB from:")
    print("  https://developer.android.com/tools/releases/platform-tools")
    print("=" * 60)
    sys.exit(1)

print(f"  Using ADB: {ADB_PATH}")


# ──────────────────────────────────────────────────────────────
# APP PACKAGE → FRIENDLY NAME MAPPING
# ──────────────────────────────────────────────────────────────
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
    "com.samsung.android.launcher": "Home Screen",
    "com.miui.home": "Home Screen",
    "com.sec.android.app.launcher": "Home Screen",
    "com.google.android.apps.nexuslauncher": "Home Screen",
    # Xiaomi / Redmi / MIUI apps
    "com.android.launcher3": "Home Screen",
    "com.bbk.launcher2": "Home Screen",
    "com.miui.securitycenter": "Security",
    "com.miui.gallery": "Gallery",
    "com.miui.player": "Mi Music",
    "com.miui.video": "Mi Video",
    "com.miui.notes": "Mi Notes",
    "com.miui.cleaner": "Cleaner",
    "com.mi.android.globalFileexplorer": "File Manager",
    "com.xiaomi.mipicks": "GetApps",
    "com.xiaomi.market": "Mi Store",
    "com.miui.weather2": "Weather",
    "com.miui.calculator": "Calculator",
    "com.miui.compass": "Compass",
    "com.miui.screenrecorder": "Screen Recorder",
    "com.miui.voiceassist": "Mi AI",
    "com.miui.yellowpage": "Yellow Pages",
    "com.xiaomi.account": "Mi Account",
    "com.miui.cloudservice": "Mi Cloud",
    "com.mi.globalbrowser": "Mi Browser",
    "com.miui.tachyon": "Tachyon Search",
    "com.miui.systemAdSolution": "MIUI Ads",
    "com.miui.mishare.connectivity": "Mi Share",
    "com.android.contacts": "Contacts",
    "com.android.phone": "Phone",
    "com.miui.personalassistant": "Personal Assistant",
    "com.android.messaging": "Messages",
    "com.xiaomi.smarthome": "Mi Home",
    "com.milink.service": "MiLink",
    "com.google.android.googlequicksearchbox": "Google Search",
    "com.google.android.apps.photos": "Google Photos",
    "com.google.android.keep": "Google Keep",
    "com.microsoft.launcher": "Microsoft Launcher",
}

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
    # Xiaomi / MIUI
    "Security": "system", "Gallery": "utility", "Mi Music": "entertainment",
    "Mi Video": "entertainment", "Mi Notes": "productivity", "Cleaner": "utility",
    "File Manager": "utility", "GetApps": "utility", "Mi Store": "shopping",
    "Weather": "utility", "Calculator": "utility", "Compass": "utility",
    "Screen Recorder": "utility", "Mi AI": "system", "Mi Account": "system",
    "Mi Cloud": "system", "Mi Browser": "browser", "Tachyon Search": "system",
    "Mi Share": "utility", "Contacts": "system", "Personal Assistant": "system",
    "Mi Home": "utility", "Google Search": "browser", "Google Photos": "utility",
    "Google Keep": "productivity",
}


def get_foreground_app():
    """Get the current foreground app using ADB."""
    try:
        result = subprocess.run(
            [ADB_PATH, "shell", "dumpsys", "activity", "activities"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "ResumedActivity" in line or "mFocusedActivity" in line:
                match = re.search(r'([a-zA-Z][a-zA-Z0-9_.]*)/[a-zA-Z.]', line)
                if match:
                    return match.group(1)

        # Fallback
        result = subprocess.run(
            [ADB_PATH, "shell", "dumpsys", "window", "displays"],
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
        print("ERROR: ADB binary not accessible.")
        sys.exit(1)


def get_battery_info():
    """Get battery level and charging state."""
    try:
        result = subprocess.run(
            [ADB_PATH, "shell", "dumpsys", "battery"],
            capture_output=True, text=True, timeout=5
        )
        level = 50
        charging = False
        for line in result.stdout.split("\n"):
            # Match only '  level: 55' NOT 'Capacity level: -1'
            # Use strict pattern: line must start with optional spaces then 'level:'
            stripped = line.strip()
            if re.match(r'^level:\s*\d+$', stripped):
                m = re.search(r'(\d+)', stripped)
                if m:
                    level = int(m.group(1))
            if "status:" in line and not "Charging state" in line:
                charging = "2" in line or "5" in line
        return level, charging
    except Exception:
        return 50, False


def get_memory_info():
    """Get RAM usage info."""
    try:
        result = subprocess.run(
            [ADB_PATH, "shell", "cat", "/proc/meminfo"],
            capture_output=True, text=True, timeout=5
        )
        total_kb = 0
        free_kb = 0
        for line in result.stdout.split("\n"):
            if line.startswith("MemTotal:"):
                total_kb = int(re.search(r'\d+', line).group())
            elif line.startswith("MemAvailable:"):
                free_kb = int(re.search(r'\d+', line).group())
        return total_kb / 1024, free_kb / 1024
    except Exception:
        return 8192, 4096


def package_to_name(package):
    """Convert package name to friendly app name."""
    if package in PACKAGE_MAP:
        return PACKAGE_MAP[package]
    parts = package.split(".")
    return parts[-1].capitalize() if parts else package


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect phone app usage via ADB")
    parser.add_argument("--duration", type=int, default=360, help="Duration in minutes (default: 360 = 6 hours)")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval in seconds (default: 5)")
    parser.add_argument("--name", type=str, default="friend", help="Your name (for filename, e.g. --name rahul)")
    args = parser.parse_args()

    # Output file named after the person
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"phone_events_{args.name}.csv")

    # Check ADB connection
    print("Checking ADB connection...")
    result = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True)
    devices = [l for l in result.stdout.strip().split("\n")[1:] if l.strip() and "device" in l]
    if not devices:
        print("=" * 60)
        print("  ERROR: No phone detected!")
        print()
        print("  Make sure:")
        print("  1. USB Debugging is ON (Settings > Developer Options)")
        print("  2. Phone is connected via USB cable")
        print("  3. You tapped 'Allow' on the USB debugging popup on phone")
        print("  4. Try: unplug and replug the cable")
        print("=" * 60)
        sys.exit(1)
    print(f"  Connected to: {devices[0].split()[0]}")

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
    print(f"  Collector: {args.name}")
    print(f"  Duration:  {args.duration} min | Interval: {args.interval}s")
    print(f"  Output:    {output}")
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
                "next_app": "",
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

            # Auto-save every 60 seconds
            if time.time() - last_save_time > 60 and events:
                _save_events(events, output, fieldnames)
                last_save_time = time.time()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # Post-process: fill in next_app
    for i in range(len(events) - 1):
        events[i]["next_app"] = events[i + 1]["current_app"]
    if events:
        events[-1]["next_app"] = events[-1]["current_app"]

    _save_events(events, output, fieldnames)

    if events:
        unique_apps = set(e["current_app"] for e in events)
        print(f"\n{'='*60}")
        print(f"  Collection Complete!")
        print(f"  Total polls:    {event_count}")
        print(f"  App switches:   {switch_count}")
        print(f"  Duration:       {(time.time() - start_time) / 60:.1f} minutes")
        print(f"  Unique apps:    {len(unique_apps)} ({', '.join(sorted(unique_apps))})")
        print(f"  Saved to:       {output}")
        print(f"{'='*60}")
        print(f"\n  >>> Send '{os.path.basename(output)}' back to Sinha! <<<")
    else:
        print("No events collected!")


def _save_events(events, output_path, fieldnames):
    """Save events to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


if __name__ == "__main__":
    main()
