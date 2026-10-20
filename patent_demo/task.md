# Patent Demo — Atomic Task Checklist (v2)
# Adaptive Memory Management: Live Patent Video CLI Dashboard
# Cross-Model Handoff Document — Use implementation_plan.md v2 as reference

---

## Handoff State Tracker (v2 — Revised Model Assignments)

| Task ID | Description | Model | Status | Depends On |
|---|---|---|---|---|
| T-01 | Create patent_demo/ `__init__.py` + `requirements_demo.txt` | FLASH | [ ] | — |
| T-02 | Write `config.py` (all constants + EVICT_DISPLAY_TICKS + AUTO_TOUR + S(a) weights) | FLASH | [ ] | T-01 |
| T-03 | Write `adb_watcher.py` (AdbWatcher + IGNORED_PACKAGES + Auto-Tour logic) | FLASH | [ ] | T-02 |
| T-04 | Write `hotkey_listener.py` (1-9, Space, 'a'/'A' Auto-Tour toggle) | FLASH | [ ] | T-03 |
| T-05 | Write `demo_director.py` (DemoPrediction + DemoDirector ML blend) | FLASH | [ ] | T-02 |
| T-06 | Write `ram_state_machine.py` (RamSlot + RamStateMachine + 6-Factor S(a) + pool) | SONNET | [ ] | T-02 |
| T-07 | Write `tui_renderer.py` (TuiRenderer + all 5 _render_* methods + S(a) header) | SONNET | [ ] | T-03, T-05, T-06 |
| T-08 | Write `run_demo.py` (entry point + dep check + boot sequence) | FLASH | [ ] | T-03–T-07 |
| T-09 | Install runtime dependencies (`rich`, `pynput`) | FLASH | [ ] | T-01 |
| T-10 | Smoke test `--no-adb` mode: full TUI renders without phone | FLASH | [ ] | T-08, T-09 |
| T-11 | Smoke test ADB live mode: foreground detection + IGNORED_PACKAGES filter | SONNET | [ ] | T-10 |
| T-12 | Verify ML model loads + predictions flow into TUI + blending visible | SONNET | [ ] | T-10 |
| T-13 | Verify hotkeys 1-9, Space; verify 'A' toggles Auto-Tour | FLASH | [ ] | T-10 |
| T-14 | Verify eviction 10-tick countdown visible; 3 PRELOADING always shown | FLASH | [ ] | T-10 |
| T-15 | Final end-to-end record-ready check (5-min stress test) | SONNET | [ ] | T-11–T-14 |

---

## PHASE 1 — Project Skeleton

### [T-01] [MODEL: FLASH] Create patent_demo/ skeleton files

- [ ] Create `patent_demo/__init__.py` with contents: `# Patent Demo Package`
- [ ] Create `patent_demo/requirements_demo.txt`:
  ```
  rich>=13.7.0
  pynput>=1.7.6
  ```

**Verification:** Both files exist in `patent_demo/`.

---

## PHASE 2 — Core Modules

### [T-02] [MODEL: FLASH] Write `patent_demo/config.py`

All values from `implementation_plan.md` Section 2.1. Implement exactly:

**Paths & ADB:**
- [ ] `CAPSTONE_ROOT` via `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
- [ ] `sys.path.insert(0, CAPSTONE_ROOT)` at module level
- [ ] `MODEL_PHASE2`, `ENGINE_PHASE2`, `MARKOV_PATH` joined with CAPSTONE_ROOT
- [ ] `ADB_EXE` from `%USERPROFILE%\Desktop\platform-tools\adb.exe`
- [ ] `ADB_POLL_MS = 400`

**FIX 4 — IGNORED_PACKAGES:**
- [ ] `IGNORED_PACKAGES = {"com.google.android.inputmethod.latin", "com.android.systemui", "com.coloros.safecenter", "com.android.launcher", "com.android.launcher3", "com.oplus.exsystemservice"}`

**Demo Director:**
- [ ] `ML_BLEND_ALPHA = 0.55`
- [ ] `CONFIDENCE_FLOOR = 0.30`
- [ ] `CONFIDENCE_JITTER = 0.018`
- [ ] `LATENCY_BASE_MS = 12.4`
- [ ] `LATENCY_SIGMA_MS = 2.1`

**RAM simulation:**
- [ ] `TOTAL_RAM_MB = 8192`, `SYSTEM_RESERVED_MB = 1600`
- [ ] `LPDDR5_BASE_ADDR = 0x70000000`
- [ ] `PAGE_SIZE_KB = 4`
- [ ] `EVICT_SLOTS = 3`, `PRELOAD_SLOTS = 3`
- [ ] `PRELOAD_WARMUP_MS = 1800`
- [ ] `THRASH_SCORE_DEFAULT = 0.18`

**FIX 1 — Eviction visibility:**
- [ ] `EVICT_DISPLAY_TICKS = 10`

**FIX 2 — Preload animation:**
- [ ] `PRELOAD_STEP_PER_TICK = 0.25`

**FIX 4 — Auto-Tour:**
- [ ] `AUTO_TOUR_INTERVAL_S = 6.0`
- [ ] `AUTO_TOUR_SEQUENCE = ["WhatsApp", "Instagram", "YouTube", "Chrome", "Camera", "Spotify", "BGMI", "Gmail"]`

**TUI:**
- [ ] `REFRESH_HZ = 4`
- [ ] `PANEL_WIDTH = 120`

**Hotkey map:**
- [ ] `HOTKEY_MAP` dict: keys "1"-"9" + " " mapped to 9 app names + "Home Screen"
  (NOTE: "a"/"A" NOT in HOTKEY_MAP — handled as toggle_auto_tour() in hotkey_listener.py)

**DEMO_MATRIX:**
- [ ] 10 app keys + "_default_" key, each with exactly 3 (app, confidence) tuples:
  - WhatsApp -> [Instagram 0.864, YouTube 0.641, Chrome 0.418]
  - Instagram -> [WhatsApp 0.812, YouTube 0.573, Camera 0.389]
  - YouTube -> [WhatsApp 0.734, Instagram 0.521, Spotify 0.341]
  - Gmail -> [Chrome 0.791, WhatsApp 0.612, Calendar 0.387]
  - BGMI -> [YouTube 0.823, Discord 0.654, WhatsApp 0.412]
  - Chrome -> [YouTube 0.687, WhatsApp 0.543, Gmail 0.398]
  - Spotify -> [WhatsApp 0.712, YouTube 0.601, Instagram 0.334]
  - Camera -> [Instagram 0.891, WhatsApp 0.672, YouTube 0.421]
  - ChatGPT -> [Chrome 0.743, Gmail 0.589, WhatsApp 0.421]
  - Home Screen -> [WhatsApp 0.654, Instagram 0.523, YouTube 0.412]
  - _default_ -> [WhatsApp 0.621, YouTube 0.487, Instagram 0.354]

**FIX 3 — S(a) weights:**
- [ ] `SA_W_RECENCY=0.25`, `SA_W_FREQ=0.20`, `SA_W_PRED=0.30`, `SA_W_IMP=0.10`, `SA_W_CTX=0.10`, `SA_W_MEM=0.30`
- [ ] `SA_LAMBDA = 0.005`
- [ ] `APP_FREQ_SCORES` dict with 15 app entries (WhatsApp:0.85 down to Home Screen:0.25) + `_default_:0.35`

**Verification:** `python -c "from patent_demo.config import EVICT_DISPLAY_TICKS, AUTO_TOUR_SEQUENCE; print(EVICT_DISPLAY_TICKS, len(AUTO_TOUR_SEQUENCE))"` prints `10 8`

---

### [T-03] [MODEL: FLASH] Write `patent_demo/adb_watcher.py`

See `implementation_plan.md` Section 2.2 for full spec.

**Imports & package map:**
- [ ] `import subprocess, threading, re, time, os, sys`
- [ ] Import PACKAGE_MAP with fallback:
  ```python
  try:
      from scripts.collect_phone_data import PACKAGE_MAP
  except Exception:
      PACKAGE_MAP = {}
  ```
- [ ] Import from config: `ADB_POLL_MS, IGNORED_PACKAGES, AUTO_TOUR_INTERVAL_S, AUTO_TOUR_SEQUENCE`

**`AdbWatcher.__init__(self, adb_path, poll_ms)`:**
- [ ] If `adb_path is None`: `_adb_ok=False`, `_current_app="Home Screen"`, `_battery_pct=74`, `_is_charging=False`, `_ram_total_mb=8192.0`, `_ram_free_mb=3200.0`
- [ ] `_lock = threading.Lock()`
- [ ] `_auto_tour = False`, `_tour_idx = 0`, `_last_tour_ts = time.monotonic()`
- [ ] Always start daemon thread calling `self._poll_loop` (handles both ADB and auto-tour)

**Properties:**
- [ ] `foreground_app`: `with self._lock: return self._current_app`
- [ ] `battery`: returns `(self._battery_pct, self._is_charging)`
- [ ] `ram`: returns `(self._ram_total_mb, self._ram_free_mb)`
- [ ] `adb_connected`: returns `self._adb_ok`

**`override_app(self, app_name)`:**
- [ ] `with self._lock: self._current_app = app_name; self._auto_tour = False`

**FIX 4 — `toggle_auto_tour(self)`:**
- [ ] `with self._lock: self._auto_tour = not self._auto_tour; self._tour_idx = 0; self._last_tour_ts = time.monotonic()`

**`_poll_loop(self)` — entire body in `try/except Exception`:**
- [ ] Step 1: Check auto-tour first (works even without ADB):
  ```python
  with self._lock:
      is_touring = self._auto_tour
      last_ts    = self._last_tour_ts
      idx        = self._tour_idx
  if is_touring and (time.monotonic() - last_ts) >= AUTO_TOUR_INTERVAL_S:
      next_app = AUTO_TOUR_SEQUENCE[idx % len(AUTO_TOUR_SEQUENCE)]
      with self._lock:
          self._current_app   = next_app
          self._tour_idx      = idx + 1
          self._last_tour_ts  = time.monotonic()
  ```
- [ ] Step 2: If `self._adb_ok` (not None adb_path): call `_get_foreground()`
  - If returned package is None or in IGNORED_PACKAGES: keep `_current_app` unchanged
  - Else: map to friendly name via PACKAGE_MAP; update `_current_app` with lock
- [ ] Step 3: `_get_battery()` -> update fields; `_get_ram()` -> update fields
- [ ] Set `_adb_ok = True` on success block; `False` on exception
- [ ] `time.sleep(self._poll_ms / 1000)`

**FIX 4 — `_get_foreground()` with IGNORED_PACKAGES filter:**
- [ ] PRIMARY: `adb shell dumpsys activity activities`, grep `ResumedActivity`, regex `r'([a-zA-Z][a-zA-Z0-9_.]*)/[a-zA-Z.]'`
- [ ] FALLBACK: `adb shell dumpsys window displays`, grep `mCurrentFocus`, same regex
- [ ] If extracted package in `IGNORED_PACKAGES`: return `None`
- [ ] Map via PACKAGE_MAP; fallback: `parts[-1].capitalize()`
- [ ] Return None on any exception/timeout

**`_get_battery()` and `_get_ram()`:**
- [ ] Identical to v1 spec (from collect_phone_data.py logic)
- [ ] Default `(74, False)` and `(8192.0, 3200.0)` on error

**Verification:** `python -c "from patent_demo.adb_watcher import AdbWatcher; w=AdbWatcher(None); print(w.foreground_app, w.adb_connected)"` prints `Home Screen False`

---

### [T-04] [MODEL: FLASH] Write `patent_demo/hotkey_listener.py`

See `implementation_plan.md` Section 2.5.

- [ ] Import `HOTKEY_MAP` from config
- [ ] `HotkeyListener.__init__(self, adb_watcher)`: store `_watcher`, `_listener=None`
- [ ] `start(self)`: try import pynput; on ImportError warn + return; else start Listener daemon
- [ ] `_on_press(self, key)` — entire body in `try/except Exception: pass`:
  - Get char: try `key.char`; on AttributeError check `keyboard.Key.space` -> `char = " "`
  - **FIX 4:** If char in `("a", "A")`: call `self._watcher.toggle_auto_tour()` and return
  - Else if char in HOTKEY_MAP: call `self._watcher.override_app(HOTKEY_MAP[char])`

**Verification:** Module imports without error; all 12 bound keys handled.

---

### [T-05] [MODEL: FLASH] Write `patent_demo/demo_director.py`

See `implementation_plan.md` Section 2.3.

- [ ] `@dataclass class DemoPrediction` with 5 fields (current_app, top3, inference_ms, model_source, timestamp)
- [ ] `DemoDirector.__init__`: `_predictor=None`, `_ml_ok=False`, `_load_models()`
- [ ] `_load_models()`: try load XGBoostPredictor from MODEL_PHASE2/ENGINE_PHASE2/MARKOV_PATH; on exception `_ml_ok=False`
- [ ] `predict(self, current_app, context) -> DemoPrediction`:
  - t0 = perf_counter
  - If ml_ok: try ML predict -> ml_top3; except -> None
  - matrix_top3 = _get_matrix_preds(current_app)
  - blended = _blend(ml_top3, matrix_top3, ML_BLEND_ALPHA) if ml_ok else matrix_top3
  - Apply jitter; clip [0.01, 0.99]; inference_ms = wall time + latency jitter
  - Return DemoPrediction (NEVER raise)
- [ ] `_get_matrix_preds(current_app)`: lookup DEMO_MATRIX + jitter
- [ ] `_blend(ml_top3, matrix_top3, alpha)`: union dict, weighted sum, sort desc, top 3

**Verification:** `python -c "from patent_demo.demo_director import DemoDirector; d=DemoDirector(); p=d.predict('WhatsApp',{}); print(len(p.top3), p.model_source)"` prints `3 demo_matrix` (or `3 xgboost_phase2` if model loads).

---

### [T-06] [MODEL: SONNET] Write `patent_demo/ram_state_machine.py`

See `implementation_plan.md` Section 2.4 for full spec. This is the most complex module.

**Module-level functions:**
- [ ] `compute_sa_score(app_name, last_accessed_s, now_s, memory_mb, is_foreground, prediction_conf, access_count, max_access_count) -> float`:
  - Full 6-Factor formula: `S(a) = clip(0.25*R + 0.20*F + 0.30*P + 0.10*I + 0.10*C - 0.30*M, 0.05, 0.98)`
  - If is_foreground: return 1.00
  - R = exp(-SA_LAMBDA * dt)
  - F = APP_FREQ_SCORES.get(app_name, 0.35)
  - P = prediction_conf (0.02 if not in top3)
  - I = 0.70, C = 0.65
  - M = min(1.0, memory_mb / 1200.0)
- [ ] `derive_hex_range(app_name, memory_mb) -> tuple[str, str]`: MD5-based (see plan Section 4)

**`@dataclass RamSlot`** with all 12 fields:
- [ ] `evict_ticks_remaining: int = 0` — **FIX 1** field
- [ ] `access_count: int = 1` — for F(a) normalization

**Class constants:**
- [ ] `INITIAL_APPS = ["WhatsApp", "Spotify", "Camera", "PhonePe", "Settings", "Calculator", "Maps", "Discord", "BGMI"]`
  (9 apps: WhatsApp=foreground + 8 background)
- [ ] `BACKGROUND_POOL = ["WhatsApp", "Instagram", "YouTube", "Gmail", "Chrome", "Spotify", "Camera", "BGMI", "ChatGPT", "PhonePe", "Calculator", "Settings", "Maps", "Discord"]`
  (14 apps for FIX 2 replenishment)

**`__init__`:**
- [ ] All dict/list fields as in plan: `slots`, `eviction_log`, `preload_log`, registries
- [ ] **FIX 3 counters:** `cumulative_latency_saved_ms=1420`, `warm_hits=1`, `thrash_score=0.14`, `gates_passed="PASS (5/5)"`
- [ ] `_current_top3_conf: dict = {}`
- [ ] Call `_populate_initial_state()`

**`on_app_switch(self, new_app, top3_conf=None)`:**
- [ ] Update `_current_top3_conf` if provided
- [ ] **FIX 3 warm-hit detection:** check if new_app was PRELOADED/PRELOADING/WARM (or in top3_conf):
  - If yes: `warm_hits += 1`; `cumulative_latency_saved_ms += max(450, cold_ms - warm_ms)` from get_profile()
- [ ] Find previous foreground slot -> set `is_foreground=False`
- [ ] Create slot if new_app not in slots (state="WARM")
- [ ] Set `slots[new_app].is_foreground=True`, `last_accessed_s=now`, `access_count+=1`
- [ ] Call `_update_sa_scores(top3_conf)`

**`trigger_eviction_cycle(self, top3_apps)`:**
- [ ] **FIX 2 — Pool replenishment:**
  ```
  eligible = [s for s in slots.values()
              if not s.is_foreground
              and s.state not in ("EVICTING","EVICTED")
              and s.app_name not in top3_apps]
  while len(eligible) < 4:
      inject cold app from BACKGROUND_POOL not in slots and not in top3_apps
      with lru_score=random.uniform(0.11, 0.24), last_accessed_s=now-random.uniform(120,600)
      state="WARM"
      re-compute eligible list; break if pool exhausted
  ```
- [ ] Sort eligible by `lru_score` ascending; pick `[:3]` as victims
- [ ] **FIX 1:** For each victim: `slot.state="EVICTING"`, `slot.evict_ticks_remaining=EVICT_DISPLAY_TICKS`
- [ ] Build log entry: `{app, pid, memory_mb, base_addr, end_addr, sa_score=slot.lru_score, freed_mb, timestamp_s}`
- [ ] Append to `eviction_log`, trim to last 5; return entries

**`trigger_preload_cycle(self, top3)`:**
- [ ] **FIX 2 — Forced 3-app PRELOADING:**
  ```
  for app in top3:
      if slots.get(app) and slots[app].is_foreground: continue  # only skip foreground
      if app in slots:
          slots[app].state = "PRELOADING"
          slots[app].warmup_progress = 0.25  # warm delta if already cached
      else:
          create new RamSlot(state="PRELOADING", warmup_progress=0.0)
  ```
  This guarantees exactly 3 PRELOADING slots visible.
- [ ] Build log entries; append to `preload_log`, trim to 5; return entries

**`tick(self)`:**
- [ ] **FIX 2 warmup:** For PRELOADING: `warmup_progress = min(1.0, warmup_progress + PRELOAD_STEP_PER_TICK)` -> if >= 1.0: state="PRELOADED"
- [ ] **FIX 1 countdown:** For EVICTING: `evict_ticks_remaining -= 1` -> if <= 0: state="EVICTED"
- [ ] Remove EVICTED slots: `self.slots = {k:v for k,v in self.slots.items() if v.state != "EVICTED"}`
- [ ] Call `_update_sa_scores()`

**Properties:** `used_mb`, `free_mb`, `utilization_pct`, `get_sorted_slots()`

**`_update_sa_scores(self, top3_conf=None)`:**
- [ ] Use `self._current_top3_conf` if top3_conf is None
- [ ] For each slot: `p = top3_conf.get(slot.app_name, 0.02)`
- [ ] `slot.lru_score = compute_sa_score(slot.app_name, slot.last_accessed_s, now, slot.memory_mb, slot.is_foreground, p, slot.access_count, max_access)`

**`_make_pid(self, app_name)`:** hash % 7000 + 3000; cached in `_pid_registry`

**`_make_addr_range(self, app_name, memory_mb)`:** call `derive_hex_range()`; cached in `_addr_registry`

**`_populate_initial_state(self)`:**
- [ ] WhatsApp: foreground=True, last_accessed_s=now, stagger=-0s
- [ ] Remaining 8 apps: last_accessed_s = now - (i+1)*45 (so S(a) spreads ~0.24 down to ~0.12)
- [ ] Use `get_profile(app).total_memory_mb` for memory_mb

**Verification:**
```
python -c "
from patent_demo.ram_state_machine import RamStateMachine
r = RamStateMachine()
print('slots:', len(r.slots))
r.tick()
print('pct:', round(r.utilization_pct, 1))
r.trigger_eviction_cycle(['Instagram', 'YouTube', 'Chrome'])
r.trigger_preload_cycle(['Instagram', 'YouTube', 'Chrome'])
r.tick()
ev = [s for s in r.slots.values() if s.state=='EVICTING']
pr = [s for s in r.slots.values() if s.state in ('PRELOADING','PRELOADED')]
print('evicting:', len(ev), 'preloading:', len(pr))
"
```
Expected output: `slots: 9`, some utilization %, `evicting: 3 preloading: 3`

---

### [T-07] [MODEL: SONNET] Write `patent_demo/tui_renderer.py`

See `implementation_plan.md` Section 2.6 for full spec and visual target wireframe.

**`build_layout() -> Layout`:**
- [ ] Exact layout tree: root -> header(size=3), body, footer(size=3)
- [ ] body -> top_row(size=12), ram_panel
- [ ] top_row -> panel_foreground(ratio=1), panel_prediction(ratio=1)

**`CATEGORY_LABELS` dict (module level, ASCII-safe):**
```python
CATEGORY_LABELS = {
    "messaging": "[SMS]", "social": "[SOC]", "productivity": "[PRD]",
    "entertainment": "[ENT]", "gaming": "[GAM]", "ai_tools": "[BOT]",
    "navigation": "[NAV]", "camera": "[CAM]", "finance": "[FIN]",
    "system": "[SYS]", "unknown": "[APP]",
}
```

**`TuiRenderer.__init__`:**
- [ ] `self.adb`, `self.director`, `self.ram_sm`
- [ ] `self.console = Console(width=PANEL_WIDTH)`
- [ ] `self.layout = build_layout()`
- [ ] `self._start = time.monotonic()`, `self._prev_app = ""`, `self._session_history = []`, `self._last_prediction = None`
- [ ] `self._last_warm_hit_text = ""`  (for warm-hit badge in foreground panel)

**`run(self)`:**
- [ ] `with Live(..., screen=True, refresh_per_second=REFRESH_HZ)` wrapping `while True` + `time.sleep`
- [ ] Inner `_update_all()` wrapped in `try/except Exception: pass`

**`_update_all(self)`:**
- [ ] Get current_app; detect transition; predict; tick; render all

**`_on_transition(self, new_app)` — FIX 2 + 3 combined:**
- [ ] `top3_conf = {a:c for a,c in self._last_prediction.top3} if self._last_prediction else {}`
- [ ] `ram_sm.on_app_switch(new_app, top3_conf)`
- [ ] `top3_apps = list(top3_conf.keys())`
- [ ] **FIX 2 victim filter:**
  ```python
  bottom3 = [s.app_name for s in ram_sm.get_sorted_slots()
             if not s.is_foreground
             and s.state not in ("EVICTING", "EVICTED")
             and s.app_name not in top3_apps][-3:]
  ```
- [ ] `ram_sm.trigger_eviction_cycle(top3_apps)` ← pass top3_apps (pool replenishment uses it)
- [ ] `ram_sm.trigger_preload_cycle(top3_apps)` ← always 3 PRELOADING
- [ ] Update session_history (keep 6), update `_prev_app`
- [ ] Update `_last_warm_hit_text` based on `ram_sm.warm_hits` delta

**`_render_header(self)` — FIX 3 + 4:**
- [ ] `adb_dot = "[bold green]CONNECTED[/]" if self.adb.adb_connected else "[bold red]DEMO MODE[/]"`
- [ ] `tour_status = "ON " if self.adb._auto_tour else "OFF"`
- [ ] `gate_text = self.ram_sm.gates_passed`
- [ ] One-line Text: `"  AMM Patent Demo  |  ADB: {dot}  |  Realme 13 5G (8192MB LPDDR5X)  |  5-Gate: {gate_text}  |  AutoTour: {tour_status}  |  {MM:SS}  "`
- [ ] `Panel(text, style="bold on black", box=box.HORIZONTALS, height=3)`

**`_render_foreground(self, current_app)` — FIX 3:**
- [ ] Look up `get_profile(current_app)`, get slot from `ram_sm.slots`
- [ ] Show warm-hit badge: if `self._last_warm_hit_text`: append "[bold green]{_last_warm_hit_text}[/]" on first line
- [ ] Show Battery %, is_charging, free RAM from `self.adb.ram`
- [ ] All fields per plan wireframe (category label, RAM MB, PID, cold/warm ms, AI badge, session history, ADB status)
- [ ] `Panel(..., title="[bold cyan]LIVE FOREGROUND APP[/]", box=box.ROUNDED, border_style="cyan")`

**`_render_prediction(self, pred)`:**
- [ ] Model line, inference bar (20-char `#`/`.`), ECE line
- [ ] Top-3 rows with confidence bars, trend arrows (`^`, `->`, `v`)
- [ ] Bottom: improvement ratio `pred.top3[0][1] / (1/29)` and model source
- [ ] `Panel(..., title="[bold magenta]NEXT-3 PREDICTION ENGINE[/]", box=box.ROUNDED, border_style="magenta")`

**`_render_ram(self, pred)` — ALL 4 FIXES:**
- [ ] **FIX 3 — Formula header line:**
  `"S(a)=0.25R+0.20F+0.30P+0.10I+0.10C-0.30M  |  Thrash: {thrash_score:.2f} (NOMINAL)  |  Latency Saved: +{latency:,}ms ({warm_hits} Warm Hits)"`
- [ ] Utilization bar (40 chars wide)
- [ ] Rich Table columns (7):
  - App (cyan, width=14), PID (dim, width=6), MB (width=5), Hex Range (dim, width=25)
  - **FIX 3:** Column header renamed to "S(a) Score" (yellow, width=12)
  - State (bold, width=12), Action (width=36)
  - `box=box.SIMPLE, show_header=True, padding=(0,1)`
- [ ] For each slot in `ram_sm.get_sorted_slots()`:
  - LRU dots: `"o"*int(slot.lru_score*10) + "."*(10-int(slot.lru_score*10))`
  - **FIX 1 — EVICTING action:** `f"[EVICTING ({slot.evict_ticks_remaining} ticks) -- reclaim_page_range()]"`
  - **FIX 2 — PRELOADING action:** `f"[PRELOADING {'#'*int(slot.warmup_progress*10)+'.'*(10-int(slot.warmup_progress*10))} {int(slot.warmup_progress*100)}%]"`
  - FOREGROUND row_style: `"bold white on dark_blue"`
  - PRELOADED row_style: `"bold green"`
  - PRELOADING row_style: `"yellow"`
  - EVICTING row_style: `"bold red"` (no strikethrough — strikethrough may not render on Windows cmd)
  - WARM row_style: `""` (default)
  - **FIX 2 — WARM action:** check if app in top3_apps: `f"[PRELOAD <- predicted #{rank+1}]"` else `f"[RETAIN -- CACHED (S={slot.lru_score:.2f})]"`
- [ ] Eviction log section: last 5 entries, dim; show sa_score in each line
- [ ] Preload log section: last 5 entries, green; show warmup progress
- [ ] `Panel(..., title="[bold blue]LPDDR5X RAM STATE MACHINE -- LIVE PROCESS CACHE[/]", box=box.HEAVY, border_style="blue")`

**`_render_footer(self)` — FIX 4:**
- [ ] Include `[A] AutoTour` in hotkey reference
- [ ] `"[dim]Hotkeys: [1]WhatsApp [2]Instagram [3]YouTube [4]Gmail [5]BGMI [6]Chrome [7]Spotify [8]Camera [9]ChatGPT [A]AutoTour [Space]Home  |  [Ctrl+C] Exit[/]"`

**`_build_ctx(self, current_app) -> dict`:**
- [ ] Construct 19-feature dict from `self.adb.battery`, `self.adb.ram`, `datetime.now()`, `get_profile()`

**Verification:**
`python -c "from patent_demo.tui_renderer import build_layout; l=build_layout(); print([k for k in l.children])"` shows layout children.

---

## PHASE 3 — Entry Point

### [T-08] [MODEL: FLASH] Write `patent_demo/run_demo.py`

See `implementation_plan.md` Section 2.7.

- [ ] Shebang + docstring with `--no-adb` usage
- [ ] sys.path inserts: capstone root AND patent_demo dir
- [ ] `_check_deps()`: try `import rich`; on fail print `pip install rich pynput` and exit(1)
- [ ] argparse `--no-adb`
- [ ] Boot prints: Booting, Loading ML models, ML status
- [ ] Relative imports: `from config import ...`, `from adb_watcher import ...`, etc.
- [ ] Boot order: AdbWatcher -> HotkeyListener -> DemoDirector -> RamStateMachine -> TuiRenderer
- [ ] `try: renderer.run() except KeyboardInterrupt: print("[Patent Demo] Stopped.")`
- [ ] `if __name__ == "__main__": main()`

---

## PHASE 4 — Install & Verification

### [T-09] [MODEL: FLASH] Install Runtime Dependencies

```shell
pip install "rich>=13.7.0" "pynput>=1.7.6"
```

- [ ] `python -c "import rich; print(rich.__version__)"` >= 13.7.0
- [ ] `python -c "import pynput; print('ok')"` prints ok

---

### [T-10] [MODEL: FLASH] Smoke Test — No-ADB Mode

```shell
cd c:\Users\sinha\Desktop\capstone
python patent_demo/run_demo.py --no-adb
```

- [ ] No import errors; boot messages appear
- [ ] Full Rich TUI renders within 5 seconds (all 5 panels visible)
- [ ] Header shows `ADB: DEMO MODE` (red), `5-Gate: PASS (5/5)`, `AutoTour: OFF`
- [ ] Panel 1 shows "Home Screen" as foreground app
- [ ] Panel 2 shows 3 predictions with confidence bars (no blank entries)
- [ ] Panel 3 header shows S(a) formula line with Latency Saved counter
- [ ] Panel 3 table shows "S(a) Score" column (not "LRU")
- [ ] Panel 3 shows at least 8 app slots from INITIAL_APPS
- [ ] Footer shows `[A] AutoTour` in hotkey list

---

### [T-11] [MODEL: SONNET] Smoke Test — ADB Live Mode (requires phone)

```shell
python patent_demo/run_demo.py
```

- [ ] Header shows green `ADB: CONNECTED`
- [ ] Panel 1 updates to match actual foreground app on phone within 0.8s
- [ ] Opening WhatsApp, YouTube, etc. on phone triggers visible transitions in all 3 panels
- [ ] Transient overlays (keyboard, system UI) do NOT cause false app switches (IGNORED_PACKAGES working)
- [ ] Each real transition triggers exactly 3 EVICTING + 3 PRELOADING rows

---

### [T-12] [MODEL: SONNET] Verify ML Model Flow

- [ ] `DemoDirector._ml_ok` is True after boot
- [ ] Panel 2 shows "xgboost_phase2" as model source
- [ ] Different apps produce meaningfully different top-3 predictions (not identical to DEMO_MATRIX)
- [ ] Confidence values jitter slightly each render tick (CONFIDENCE_JITTER working)
- [ ] Inference latency displayed in 10-20ms range (LATENCY_SIGMA_MS working)

---

### [T-13] [MODEL: FLASH] Verify Hotkeys + Auto-Tour

- [ ] Keys 1-9: correct app override (check each against HOTKEY_MAP in config.py)
- [ ] Space: sets foreground to "Home Screen"
- [ ] `A` key: toggles Auto-Tour ON (header shows `AutoTour: ON`)
- [ ] In Auto-Tour mode: app advances through AUTO_TOUR_SEQUENCE every 6 seconds
- [ ] Pressing `A` again: toggles Auto-Tour OFF; manual hotkeys work again
- [ ] Pressing any 1-9 key during Auto-Tour: turns off Auto-Tour AND overrides app
- [ ] Rapid key presses do not crash TUI

---

### [T-14] [MODEL: FLASH] Verify Eviction Countdown + Forced 3-App Preload

**Eviction visibility (FIX 1):**
- [ ] After any app switch: 3 rows appear with state `EVICTING` in Panel 3
- [ ] EVICTING rows remain visible for at least 2.5 seconds (10 ticks x 250ms)
- [ ] EVICTING action column shows countdown: `[EVICTING (10 ticks) ...]`, then (9), (8), etc.
- [ ] After 10 ticks: EVICTING rows disappear from table
- [ ] Eviction log persists the last 5 eviction events even after slots are removed

**Forced 3-app preload (FIX 2):**
- [ ] After EVERY app switch: exactly 3 rows show PRELOADING state (not 0, not 1)
- [ ] PRELOADING warmup_progress starts at 0% (new app) or 25% (warm delta)
- [ ] Progress bar in Action column increments: `#####.....  50%` -> `#######...  75%` -> `########## 100%`
- [ ] After 4 ticks (1 second): all 3 PRELOADING -> PRELOADED state
- [ ] Pool replenishment: after 5+ consecutive app switches, still 3 EVICTING on every switch

**FIX 3 — Telemetry:**
- [ ] `Latency Saved` counter in Panel 3 header increases after each warm hit
- [ ] Warm Hits counter increments when a PRELOADED app becomes foreground
- [ ] S(a) scores differ meaningfully between apps (not all the same value)

---

### [T-15] [MODEL: SONNET] Final Record-Ready Check

- [ ] Terminal is >= 120 columns
- [ ] All 5 panels fill screen without wrapping or overlap
- [ ] Header timer counts up MM:SS correctly
- [ ] Auto-Tour mode runs unattended for 5+ minutes without crash
- [ ] S(a) scores are realistic: FOREGROUND=1.00, fresh apps ~0.75-0.85, stale ~0.10-0.25
- [ ] Hex addresses look realistic: `0x7xxxxx00` format, non-overlapping
- [ ] Both `--no-adb` and live ADB modes produce record-quality output
- [ ] Ctrl+C exits cleanly, no traceback printed

**Record command:**
```shell
python patent_demo/run_demo.py          # with phone
python patent_demo/run_demo.py --no-adb # without phone
```

---

## Quick Reference: Import Chain (v2)

```
run_demo.py
  |-- config.py  (EVICT_DISPLAY_TICKS, PRELOAD_STEP_PER_TICK, IGNORED_PACKAGES,
  |               AUTO_TOUR_SEQUENCE, SA_W_*, APP_FREQ_SCORES, DEMO_MATRIX, HOTKEY_MAP)
  |-- adb_watcher.py  (AdbWatcher + IGNORED_PACKAGES filter + toggle_auto_tour)
  |   |-- scripts/collect_phone_data.py  (PACKAGE_MAP - read only; fallback to {})
  |-- hotkey_listener.py  (1-9, Space -> override_app; a/A -> toggle_auto_tour)
  |-- demo_director.py  (DemoDirector + DemoPrediction)
  |   |-- src/prediction/xgboost_predictor.py  (read only)
  |-- ram_state_machine.py  (RamStateMachine + RamSlot + compute_sa_score)
  |   |-- src/data/app_profiles.py  (APP_PROFILES, get_profile - read only)
  |-- tui_renderer.py  (TuiRenderer + build_layout; all 5 _render_* methods)
      |-- src/data/app_profiles.py  (get_profile - read only)
      |-- rich  (external dep)
```

---

## Zero-Destructive-Change Checklist (unchanged)

**NONE of these files may be edited:**
- [ ] `src/prediction/xgboost_predictor.py`
- [ ] `src/prediction/feature_engine.py`
- [ ] `src/prediction/markov_predictor.py`
- [ ] `src/memory/cache_scorer.py`
- [ ] `src/models/prediction.py`, `src/models/memory.py`, `src/models/context.py`
- [ ] `src/data/app_profiles.py`
- [ ] `scripts/collect_phone_data.py`, `scripts/live_demo.py`
- [ ] `dashboard/`, `models_trained/` (read only)

**ALL new code lives exclusively inside `patent_demo/`.**
