#!/usr/bin/env python3
"""
run_demo.py — Entry point for the AMM Patent Video CLI Dashboard.

Usage:
    python patent_demo/run_demo.py              # Live mode (phone via USB)
    python patent_demo/run_demo.py --no-adb     # Demo mode (hotkeys + Auto-Tour only)

Install dependencies first:
    pip install rich pynput
"""
import argparse
import os
import sys

# ── Inject both capstone root AND patent_demo dir into sys.path ──────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CAPSTONE = os.path.dirname(_THIS_DIR)
if _CAPSTONE not in sys.path:
    sys.path.insert(0, _CAPSTONE)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _check_deps() -> None:
    """Fail early with a helpful message if rich is missing."""
    missing = []
    try:
        import rich  # noqa: F401
    except ImportError:
        missing.append("rich>=13.7.0")
    if missing:
        print(
            f"\n[Patent Demo] Missing required packages: {', '.join(missing)}\n"
            f"  Install with:  pip install rich pynput\n"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMM Patent Demo — Live CLI Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Hotkeys during demo:\n"
            "  1-9      → force foreground app\n"
            "  Space    → Home Screen\n"
            "  A        → toggle Auto-Tour (cycles 8 apps every 6 s)\n"
            "  Ctrl+C   → exit\n"
        ),
    )
    parser.add_argument(
        "--no-adb",
        action="store_true",
        help="Hotkey + Auto-Tour only — no phone required",
    )
    args = parser.parse_args()
    _check_deps()

    # ── Imports after path is set up ─────────────────────────────────────────
    from config            import ADB_EXE, ADB_POLL_MS
    from adb_watcher       import AdbWatcher
    from hotkey_listener   import HotkeyListener
    from demo_director     import DemoDirector
    from ram_state_machine import RamStateMachine
    from tui_renderer      import TuiRenderer

    print()
    print("[Patent Demo] Booting AMM Adaptive Memory Management Demo...")

    # 1. ADB Watcher
    adb_path = ADB_EXE if not args.no_adb else None
    watcher  = AdbWatcher(adb_path, ADB_POLL_MS)
    mode_str = "ADB LIVE" if not args.no_adb else "DEMO MODE (no-adb)"
    print(f"[Patent Demo] ADB Watcher started — mode: {mode_str}")

    # 2. Hotkey Listener (1–9, Space, A/Auto-Tour)
    hotkeys = HotkeyListener(watcher)
    hotkeys.start()
    print("[Patent Demo] Hotkey listener started  (1-9, Space, A=AutoTour)")

    # 3. Demo Director (loads Phase-2 XGBoost — ~3 s)
    print("[Patent Demo] Loading ML models (~3 s)...")
    director = DemoDirector()
    ml_status = (
        "XGBoost Phase-2 loaded (alpha-blend mode)"
        if director._ml_ok
        else "Demo Matrix mode (ML unavailable — prediction matrix active)"
    )
    print(f"[Patent Demo] {ml_status}")

    # 4. RAM State Machine (9 initial apps) — Issue 4: wire adb_watcher for real phone sync
    ram_sm = RamStateMachine(adb_watcher=watcher)
    print(f"[Patent Demo] RAM State Machine ready — {len(ram_sm.slots)} initial app slots")

    # 5. TUI Renderer — blocks until Ctrl+C
    renderer = TuiRenderer(watcher, director, ram_sm)
    print("[Patent Demo] Launching fullscreen dashboard... (Ctrl+C to exit)\n")

    try:
        renderer.run()
    except KeyboardInterrupt:
        pass

    print("\n[Patent Demo] Stopped. Thank you.")


if __name__ == "__main__":
    main()
