"""
hotkey_listener.py — Silent keyboard override using pynput.

Binds:
  "1"-"9"  → override foreground app via HOTKEY_MAP
  " "      → Home Screen
  "a"/"A"  → toggle Auto-Tour on AdbWatcher
  (all other keys ignored silently)

Does NOT suppress keypresses (suppress=False) to avoid breaking the terminal.
Runs in a pynput daemon thread — completely silent on screen.
"""
import sys
import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
_CAPSTONE = os.path.dirname(_HERE)
if _CAPSTONE not in sys.path:
    sys.path.insert(0, _CAPSTONE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import HOTKEY_MAP


class HotkeyListener:
    """Listens for keypresses and routes them to AdbWatcher overrides."""

    def __init__(self, adb_watcher) -> None:
        self._watcher  = adb_watcher
        self._listener = None

    def start(self) -> None:
        """Launch pynput Listener in a daemon thread. Warns and skips if pynput missing."""
        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                daemon=True,
                suppress=False,
            )
            self._listener.start()
        except ImportError:
            print(
                "[HotkeyListener] pynput not installed; hotkeys disabled.\n"
                "  Install with: pip install pynput"
            )
        except Exception as exc:
            print(f"[HotkeyListener] Could not start listener: {exc}")

    def _on_press(self, key) -> None:
        """Dispatch keypress to the appropriate watcher action."""
        try:
            # Attempt to read as a character key
            char = key.char
        except AttributeError:
            # Special key — check for Space
            try:
                from pynput.keyboard import Key
                if key == Key.space:
                    char = " "
                else:
                    return
            except Exception:
                return

        try:
            if char in ("a", "A"):
                # FIX 4: Auto-Tour toggle
                self._watcher.toggle_auto_tour()
            elif char in HOTKEY_MAP:
                self._watcher.override_app(HOTKEY_MAP[char])
        except Exception:
            pass   # never crash on a keypress
