"""
Laptop Data Collector — Monitors foreground Windows application.
PORTABLE VERSION: Send this file + get_foreground.ps1 to friends.

No extra software needed! Just Python + Windows.

Usage:
  python friend_collect_laptop.py                    # collect for 6 hours (default)
  python friend_collect_laptop.py --duration 360     # collect for 6 hours
  python friend_collect_laptop.py --duration 120     # collect for 2 hours
  python friend_collect_laptop.py --duration 30      # collect for 30 mins (quick test)
  Press Ctrl+C to stop early at any time.

Output: laptop_events_<name>.csv (in same folder as this script)
"""
import time
import csv
import os
import sys
import subprocess
import ctypes
from datetime import datetime

kernel32 = ctypes.windll.kernel32


def get_foreground_window_info():
    """Get foreground window info using PowerShell helper script."""
    try:
        ps1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_foreground.ps1")
        if not os.path.isfile(ps1_path):
            print(f"ERROR: get_foreground.ps1 not found at: {ps1_path}")
            print("  Make sure get_foreground.ps1 is in the SAME folder as this script!")
            sys.exit(1)

        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1_path],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout.strip()
        if "|" in output:
            parts = output.split("|", 1)
            process_name = parts[0].strip() + ".exe"
            window_title = parts[1].strip() if len(parts) > 1 else ""
            if process_name and process_name != ".exe":
                return window_title, process_name, 0

        # Fallback
        result2 = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
             "Sort-Object CPU -Descending | Select-Object -First 1 -ExpandProperty ProcessName"],
            capture_output=True, text=True, timeout=5
        )
        proc = result2.stdout.strip()
        if proc:
            return proc, proc + ".exe", 0

        return None, None, None
    except Exception:
        return None, None, None


# Process name → App name mapping
PROCESS_MAP = {
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "firefox.exe": "Firefox",
    "brave.exe": "Brave",
    "Code.exe": "VS Code",
    "code.exe": "VS Code",
    "AntigravityIDE.exe": "Antigravity IDE",
    "antigravity.exe": "Antigravity IDE",
    "WhatsApp.exe": "WhatsApp Desktop",
    "Telegram.exe": "Telegram Desktop",
    "Discord.exe": "Discord",
    "Spotify.exe": "Spotify",
    "slack.exe": "Slack",
    "WINWORD.EXE": "MS Word",
    "EXCEL.EXE": "MS Excel",
    "POWERPNT.EXE": "MS PowerPoint",
    "OUTLOOK.EXE": "MS Outlook",
    "Teams.exe": "MS Teams",
    "ms-teams.exe": "MS Teams",
    "explorer.exe": "File Explorer",
    "WindowsTerminal.exe": "Terminal",
    "cmd.exe": "Command Prompt",
    "powershell.exe": "PowerShell",
    "notepad.exe": "Notepad",
    "Notion.exe": "Notion",
    "vlc.exe": "VLC",
    "Postman.exe": "Postman",
    "python.exe": "Python",
    "idea64.exe": "IntelliJ IDEA",
    "pycharm64.exe": "PyCharm",
    "studio64.exe": "Android Studio",
    "GitHubDesktop.exe": "GitHub Desktop",
    "Calculator.exe": "Calculator",
    "SnippingTool.exe": "Snipping Tool",
    "Taskmgr.exe": "Task Manager",
    "ApplicationFrameHost.exe": "Windows App",
    "SystemSettings.exe": "Settings",
    "streamlit.exe": "Streamlit",
}

CATEGORY_MAP = {
    "Chrome": "browser", "Edge": "browser", "Firefox": "browser", "Brave": "browser",
    "VS Code": "productivity", "Antigravity IDE": "productivity",
    "WhatsApp Desktop": "messaging", "Telegram Desktop": "messaging",
    "Discord": "social", "Spotify": "entertainment", "Slack": "productivity",
    "MS Word": "productivity", "MS Excel": "productivity",
    "MS PowerPoint": "productivity", "MS Outlook": "productivity",
    "MS Teams": "productivity", "File Explorer": "utility",
    "Terminal": "productivity", "Command Prompt": "productivity",
    "PowerShell": "productivity", "Notepad": "productivity",
    "Notion": "productivity", "VLC": "entertainment",
    "Postman": "productivity", "Python": "productivity",
    "IntelliJ IDEA": "productivity", "PyCharm": "productivity",
    "Android Studio": "productivity", "GitHub Desktop": "productivity",
    "Calculator": "utility", "Snipping Tool": "utility",
    "Task Manager": "system", "Windows App": "utility",
    "Settings": "system", "Streamlit": "productivity",
}


def process_to_app_name(process_name, window_title):
    """Convert process name to friendly app name."""
    if process_name in PROCESS_MAP:
        return PROCESS_MAP[process_name]
    title_lower = window_title.lower() if window_title else ""
    if "chrome" in title_lower:
        return "Chrome"
    if "whatsapp" in title_lower:
        return "WhatsApp Desktop"
    if "antigravity" in title_lower:
        return "Antigravity IDE"
    name = process_name.replace(".exe", "").replace(".EXE", "")
    return name if name else "Unknown"


def get_system_metrics():
    """Get basic system metrics without psutil."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))

        total_mb = mem.ullTotalPhys / (1024 * 1024)
        free_mb = mem.ullAvailPhys / (1024 * 1024)
        usage_pct = mem.dwMemoryLoad

        return total_mb, free_mb, usage_pct
    except Exception:
        return 16384, 8192, 50


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect laptop foreground app data")
    parser.add_argument("--duration", type=int, default=360, help="Duration in minutes (default: 360 = 6 hours)")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval in seconds (default: 5)")
    parser.add_argument("--name", type=str, default="friend", help="Your name (for filename, e.g. --name rahul)")
    args = parser.parse_args()

    # Output file named after the person
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"laptop_events_{args.name}.csv")

    end_time = time.time() + args.duration * 60
    events = []
    prev_app = None
    event_count = 0
    switch_count = 0
    start_time = time.time()

    print("=" * 60)
    print("  LAPTOP DATA COLLECTOR")
    print(f"  Collector: {args.name}")
    print(f"  Duration:  {args.duration} min | Interval: {args.interval}s")
    print(f"  Output:    {output}")
    print("  Use your laptop normally. Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        while time.time() < end_time:
            window_title, process_name, pid = get_foreground_window_info()

            if not process_name or process_name == "Unknown":
                time.sleep(args.interval)
                continue

            app_name = process_to_app_name(process_name, window_title)
            category = CATEGORY_MAP.get(app_name, "other")
            now = datetime.now()
            ram_total, ram_free, ram_pct = get_system_metrics()

            is_switch = (app_name != prev_app) and (prev_app is not None)
            if is_switch:
                switch_count += 1

            event = {
                "timestamp": now.isoformat(),
                "current_app": app_name,
                "previous_app": prev_app or app_name,
                "next_app": "",
                "category": category,
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "battery": 1.0,
                "is_charging": 1,
                "ram_total_mb": round(ram_total, 0),
                "ram_free_mb": round(ram_free, 0),
                "ram_usage_pct": ram_pct,
                "is_switch": int(is_switch),
                "window_title": window_title[:100] if window_title else "",
                "process": process_name,
            }
            events.append(event)
            event_count += 1

            if is_switch:
                elapsed = int(time.time() - start_time)
                print(f"  [{elapsed//60:02d}:{elapsed%60:02d}] Switch #{switch_count}: "
                      f"{prev_app} -> {app_name} "
                      f"(RAM: {ram_pct}% used, free={ram_free:.0f}MB)")

            prev_app = app_name
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # Post-process: fill in next_app
    for i in range(len(events) - 1):
        if events[i]["is_switch"] or events[i]["current_app"] != events[i + 1]["current_app"]:
            events[i]["next_app"] = events[i + 1]["current_app"]
        else:
            events[i]["next_app"] = events[i]["current_app"]
    if events:
        events[-1]["next_app"] = events[-1]["current_app"]

    # Save to CSV
    if events:
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=events[0].keys())
            writer.writeheader()
            writer.writerows(events)

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


if __name__ == "__main__":
    main()
