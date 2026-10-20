import sys
sys.path.insert(0, r'c:\Users\sinha\Desktop\capstone')
sys.path.insert(0, r'c:\Users\sinha\Desktop\capstone\patent_demo')

print('=== IMPORT SMOKE TEST ===')
from patent_demo.adb_watcher import AdbWatcher, REVERSE_PACKAGE_MAP
print(f'adb_watcher OK   REVERSE_PACKAGE_MAP has {len(REVERSE_PACKAGE_MAP)} entries')
print(f'  WhatsApp -> {REVERSE_PACKAGE_MAP.get("WhatsApp")}')
print(f'  Camera   -> {REVERSE_PACKAGE_MAP.get("Camera")}')

from patent_demo.ram_state_machine import RamStateMachine
print()

# Test edge case A: 0 eligible
r2 = RamStateMachine()
for name, slot in r2.slots.items():
    if not slot.is_foreground:
        slot.state = 'PRELOADING'
r2.trigger_eviction_cycle(['Spotify','Camera','PhonePe'])
print('Edge Case A (0 eligible):')
print(f'  eviction_log last entry status: {r2.eviction_log[-1].get("status", "N/A")}')

# Test Issue 4: adb_watcher parameter
from patent_demo.tui_renderer import TuiRenderer
from patent_demo.demo_director import DemoDirector
w = AdbWatcher(adb_path=None)
rm = RamStateMachine(adb_watcher=w)
d  = DemoDirector()
renderer = TuiRenderer(w, d, rm)
print()
print('TuiRenderer init OK')

# Test Issue 2: evicted slots stay
r3 = RamStateMachine()
r3.trigger_preload_cycle(['Instagram','YouTube','Chrome'])
r3.trigger_eviction_cycle(['Instagram','YouTube','Chrome'])
for _ in range(12):
    r3.tick()
evicted = [s for s in r3.get_sorted_slots() if s.state == 'EVICTED']
print(f'Issue 2 EVICTED still in table: {len(evicted)} (expected 3)')
assert len(evicted) == 3, 'FAIL: EVICTED slots were deleted prematurely'

# Test Issue 5B: partial eviction (2 eligible)
r4 = RamStateMachine()
apps = list(r4.slots.keys())
for i, (k, v) in enumerate(r4.slots.items()):
    if not v.is_foreground and i > 2:
        v.state = 'PRELOADING'
r4.trigger_eviction_cycle(['BGMI'])
print(f'Issue 5 eviction_log entries: {len(r4.eviction_log)}')

print()
print('=== ALL SMOKE TESTS PASSED ===')
