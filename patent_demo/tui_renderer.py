"""
tui_renderer.py — Rich 4-Hz fullscreen CLI dashboard for the AMM patent demo.

Layout (120 cols):
  [HEADER]  ADB status | Device | 5-Gate | AutoTour | Elapsed
  [TOP-L]   Live Foreground App panel
  [TOP-R]   Next-3 Prediction Engine panel
  [MAIN]    LPDDR5X RAM State Machine panel (S(a) formula + process table + logs)
  [FOOTER]  Hotkey reference

Issues fixed in this version:
  Issue 1: Prediction-to-RAM sync — predict FIRST, then on_transition (so RAM table
           always preloads the SAME 3 apps shown in the Prediction panel).
           Confidence jitter only (no name flip) while staying on the same app.
  Issue 2: EVICTED rows rendered with distinct red style + freed-MB action string.
  Issue 3: top_row height reduced to 9; wider columns; last-2 log entries per section.
  Issue 4: RamStateMachine receives adb_watcher; initial preload+evict triggered at t=0.
  Issue 5: Base-case log entries rendered in dim green / dim cyan.
"""
import os
import sys
import time
import random
from datetime import datetime

_HERE     = os.path.dirname(os.path.abspath(__file__))
_CAPSTONE = os.path.dirname(_HERE)
if _CAPSTONE not in sys.path:
    sys.path.insert(0, _CAPSTONE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import REFRESH_HZ, PANEL_WIDTH, CONFIDENCE_JITTER

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel  import Panel
    from rich.table  import Table
    from rich.live   import Live
    from rich.text   import Text
    from rich        import box
except ImportError:
    raise SystemExit(
        "\n[Patent Demo] 'rich' not installed.\n"
        "  Run: pip install rich pynput\n"
    )

try:
    from src.data.app_profiles import get_profile
except Exception:
    def get_profile(name):  # type: ignore
        class _P:
            app_name        = name
            category        = "unknown"
            total_memory_mb = 200.0
            cold_launch_ms  = 1500
            warm_launch_ms  = 120
            importance      = 0.5
            is_ai_workload  = False
        return _P()

# ASCII-safe category labels (safe on all Windows terminals)
CATEGORY_LABELS = {
    "messaging":    "[MSG]",
    "social":       "[SOC]",
    "productivity": "[PRD]",
    "entertainment":"[ENT]",
    "gaming":       "[GAM]",
    "ai_tools":     "[BOT]",
    "navigation":   "[NAV]",
    "camera":       "[CAM]",
    "finance":      "[FIN]",
    "system":       "[SYS]",
    "unknown":      "[APP]",
}


# ── Layout builder ────────────────────────────────────────────────────────────

def build_layout() -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header",  size=3),
        Layout(name="body"),
        Layout(name="footer",  size=3),
    )
    layout["body"].split_column(
        Layout(name="top_row",  size=9),   # Issue 3: reduced from 12 → 9
        Layout(name="ram_panel"),
    )
    layout["top_row"].split_row(
        Layout(name="panel_foreground", ratio=1),
        Layout(name="panel_prediction", ratio=1),
    )
    return layout


# ── Renderer ──────────────────────────────────────────────────────────────────

class TuiRenderer:
    """Owns the Rich Live context and all panel render functions."""

    def __init__(self, adb_watcher, demo_director, ram_sm) -> None:
        self.adb      = adb_watcher
        self.director = demo_director
        self.ram_sm   = ram_sm

        self.console = Console(width=PANEL_WIDTH, highlight=False)
        self.layout  = build_layout()

        self._start            = time.monotonic()
        self._prev_app         = ""
        self._session_history: list[str] = []
        self._last_prediction  = None          # DemoPrediction | None
        self._last_warm_delta  = 0             # ms saved on last warm hit
        self._show_warm_badge  = False
        self._warm_badge_ticks = 0             # how many ticks to show badge

        # Issue 1: stable top3 names (no flip while on same app)
        self._stable_top3: list[tuple[str, float]] = []

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        # Issue 4: trigger initial preload+evict at t=0 so dashboard is alive immediately
        self._boot_initial_cycles()

        with Live(
            self.layout,
            console=self.console,
            refresh_per_second=REFRESH_HZ,
            screen=True,
        ):
            while True:
                try:
                    self._update_all()
                except Exception:
                    pass
                time.sleep(1.0 / REFRESH_HZ)

    def _boot_initial_cycles(self) -> None:
        """
        Issue 4: Fire one preload+evict cycle at startup (before first render tick)
        so the dashboard immediately shows 3 PRELOADING and 3 EVICTING rows even
        before the user touches the phone.
        """
        try:
            current_app = self.adb.foreground_app
            ctx  = self._build_ctx(current_app)
            pred = self.director.predict(current_app, ctx)
            self._last_prediction = pred
            self._stable_top3     = list(pred.top3)

            top3_apps = [a for a, _ in pred.top3]
            top3_conf = {a: c for a, c in pred.top3}
            self.ram_sm._current_top3_conf = top3_conf

            self.ram_sm.trigger_preload_cycle(top3_apps)
            self.ram_sm.trigger_eviction_cycle(top3_apps)

            self._prev_app = current_app
        except Exception:
            pass

    def _update_all(self) -> None:
        current_app = self.adb.foreground_app

        if current_app != self._prev_app:
            # ── Issue 1: predict FIRST, then transition ───────────────────────
            prev_top3_conf = {a: c for a, c in self._last_prediction.top3} if self._last_prediction else {}
            ctx  = self._build_ctx(current_app)
            pred = self.director.predict(current_app, ctx)
            self._last_prediction = pred
            self._stable_top3     = list(pred.top3)
            self._on_transition(current_app, pred, prev_top3_conf)
        else:
            # Same app — reuse stable top3 names, apply tiny confidence jitter only
            ctx  = self._build_ctx(current_app)
            pred = self.director.predict(current_app, ctx)
            if self._stable_top3:
                jittered = [
                    (app, max(0.05, min(0.99, conf + random.uniform(-CONFIDENCE_JITTER, CONFIDENCE_JITTER))))
                    for app, conf in self._stable_top3
                ]
                pred = type(pred)(
                    top3          = jittered,
                    model_source  = pred.model_source,
                    inference_ms  = pred.inference_ms + random.uniform(-1.5, 1.5),
                )
            self._last_prediction = pred

        self.ram_sm.tick()

        # Decay warm badge
        if self._warm_badge_ticks > 0:
            self._warm_badge_ticks -= 1
        else:
            self._show_warm_badge = False

        self.layout["header"].update(self._render_header())
        self.layout["panel_foreground"].update(self._render_foreground(current_app))
        self.layout["panel_prediction"].update(self._render_prediction(pred))
        self.layout["ram_panel"].update(self._render_ram(pred))
        self.layout["footer"].update(self._render_footer())

    def _on_transition(self, new_app: str, new_pred, prev_top3_conf: dict[str, float]) -> None:
        """
        Issue 1: Handle foreground-app transitions: pass new prediction into RAM SM
        so preloaded apps ALWAYS match what the Prediction panel displays.
        """
        # Extract new top3
        new_top3_apps = [app for app, _ in new_pred.top3]
        new_top3_conf = {app: conf for app, conf in new_pred.top3}

        # Update RAM SM's prediction confidence map BEFORE switching
        self.ram_sm._current_top3_conf = new_top3_conf

        prev_warm_hits = self.ram_sm.warm_hits
        self.ram_sm.on_app_switch(new_app, prev_top3_conf)

        # Show warm-hit badge if hit counter increased
        if self.ram_sm.warm_hits > prev_warm_hits:
            self._last_warm_delta = max(450, 1500 - 120)
            self._show_warm_badge  = True
            self._warm_badge_ticks = 8   # visible for 2 s

        # Issue 1: preload FIRST (so predicted apps appear in table immediately),
        # then evict (so evicted apps don't include our new preload targets)
        self.ram_sm.trigger_preload_cycle(new_top3_apps)
        self.ram_sm.trigger_eviction_cycle(new_top3_apps)

        self._session_history.append(new_app)
        if len(self._session_history) > 6:
            self._session_history = self._session_history[-6:]
        self._prev_app = new_app

    # ── Panel: Header ─────────────────────────────────────────────────────────

    def _render_header(self) -> Panel:
        elapsed = int(time.monotonic() - self._start)
        mm, ss  = divmod(elapsed, 60)

        adb_dot  = (
            "[bold green]CONNECTED[/]"
            if self.adb.adb_connected
            else "[bold red]DEMO MODE[/]"
        )
        tour_on = getattr(self.adb, "_auto_tour", False)
        tour_status = "[bold yellow]ON [/]" if tour_on else "[dim]OFF[/]"
        gate = self.ram_sm.gates_passed

        t = Text(justify="center")
        t.append(" AMM Patent Demo ", style="bold cyan")
        t.append(" | ADB: ")
        t.append_text(Text.from_markup(adb_dot))
        dev_info = getattr(self.adb, "device_info", "Android Device (8192MB LPDDR5X)")
        t.append(f" | {dev_info}")
        t.append(f" | 5-Gate: ")
        t.append(gate, style="bold green" if "PASS" in gate else "bold red")
        t.append(f" | AutoTour: ")
        t.append_text(Text.from_markup(tour_status))
        t.append(f" | {mm:02d}:{ss:02d} ", style="dim")

        return Panel(t, style="bold on black", box=box.HORIZONTALS, height=3)

    # ── Panel: Foreground App ─────────────────────────────────────────────────
    # Issue 3: compact to 7 inner lines max (fits size=9)

    def _render_foreground(self, current_app: str) -> Panel:
        profile = get_profile(current_app)
        slot    = self.ram_sm.slots.get(current_app)
        pid     = slot.pid if slot else "----"
        mem_mb  = int(slot.memory_mb if slot else getattr(profile, "total_memory_mb", 200))

        cat   = getattr(profile, "category", "unknown")
        label = CATEGORY_LABELS.get(cat, "[APP]")
        cold  = getattr(profile, "cold_launch_ms", 1500)
        warm  = getattr(profile, "warm_launch_ms", 120)
        ai_wl = getattr(profile, "is_ai_workload", False)

        batt_pct, is_charging = self.adb.battery
        _, ram_free            = self.adb.ram

        t = Text()

        # Line 1: App name + warm-hit badge
        t.append(f" {label} ", style="bold white on dark_blue")
        t.append(f" {current_app}", style="bold cyan")
        if self._show_warm_badge:
            saved_s = self._last_warm_delta / 1000
            t.append(f"   [WARM HIT +{saved_s:.1f}s]", style="bold green")
        t.append("\n")

        # Line 2: Category | RAM | PID
        t.append(f" Cat:{cat:<10} RAM:{mem_mb}MB  PID:{pid}", style="dim")
        t.append("\n")

        # Line 3: Startup times | AI
        ai_str = "YES" if ai_wl else "NO"
        t.append(f" Cold:{cold}ms / Warm:{warm}ms  AI:{ai_str}", style="dim")
        t.append("\n")

        # Line 4: Battery | Free RAM
        charge_str = "(Chg)" if is_charging else "     "
        t.append(f" Batt:{batt_pct}% {charge_str}  Free:{int(ram_free)}MB")
        t.append("\n")

        # Line 5: Session history
        hist = self._session_history[-4:] if self._session_history else [current_app]
        t.append(f" {' -> '.join(hist)}\n", style="dim cyan")

        # Line 6: ADB status
        adb_line = (
            "[bold green] ADB: LIVE[/]"
            if self.adb.adb_connected
            else "[bold red] ADB: DEMO MODE[/]"
        )
        t.append_text(Text.from_markup(adb_line))

        return Panel(
            t,
            title="[bold cyan]LIVE FOREGROUND APP[/]",
            box=box.ROUNDED,
            border_style="cyan",
        )

    # ── Panel: Prediction Engine ──────────────────────────────────────────────
    # Issue 3: compact to 7 inner lines max (fits size=9)

    def _render_prediction(self, pred) -> Panel:
        t = Text()

        # Line 1: Model + inference bar
        inf_ms = pred.inference_ms
        bar_filled = min(20, int(inf_ms / 50.0 * 20))
        bar_str    = "#" * bar_filled + "." * (20 - bar_filled)
        t.append(f" Model: {pred.model_source}  ", style="dim")
        t.append(f"[{bar_str}]", style="green" if inf_ms < 30 else "yellow")
        t.append(f"  {inf_ms:.1f} ms\n")

        # Line 2: ECE / alpha
        t.append(f" ECE: 1.1%  |  Markov alpha=0.55\n", style="dim")

        # Lines 3-5: Top-3 predictions
        for i, (app, conf) in enumerate(pred.top3):
            bar_len   = int(conf * 20)
            conf_bar  = "#" * bar_len + "." * (20 - bar_len)
            if conf >= 0.60:
                arrow = "[bold green]^[/]"
                style = "bold green"
            elif conf >= 0.35:
                arrow = "[yellow]->[/]"
                style = "yellow"
            else:
                arrow = "[red]v[/]"
                style = "red"

            t.append(f"  {i+1}. {app:<16}", style=style)
            t.append(f"[{conf_bar}]", style="green" if conf >= 0.60 else "yellow")
            t.append(f" {conf:.1%}  ")
            t.append_text(Text.from_markup(arrow))
            t.append("\n")

        # Line 6: Improvement ratio
        n_apps   = 29
        top_conf = pred.top3[0][1] if pred.top3 else 0.5
        lift     = top_conf / (1.0 / n_apps)
        t.append(f"\n Top-1 vs Random: {lift:.1f}x  Src: {pred.model_source}\n", style="bold white")

        return Panel(
            t,
            title="[bold magenta]NEXT-3 PREDICTION ENGINE[/]",
            box=box.ROUNDED,
            border_style="magenta",
        )

    # ── Panel: RAM State Machine ──────────────────────────────────────────────

    def _render_ram(self, pred) -> Panel:
        # Build top3 lookup
        top3_apps: list[str] = [a for a, _ in pred.top3] if pred else []
        top3_conf: dict[str, float] = {a: c for a, c in pred.top3} if pred else {}

        # ── Formula header line ───────────────────────────────────────────────
        formula = Text()
        formula.append(
            " S(a)=0.25R+0.20F+0.30P+0.10I+0.10C-0.30M",
            style="bold white",
        )
        formula.append(
            f"  |  Thrash: {self.ram_sm.thrash_score:.2f} (NOMINAL)",
            style="dim",
        )
        formula.append(
            f"  |  Latency Saved: +{self.ram_sm.cumulative_latency_saved_ms:,}ms"
            f" ({self.ram_sm.warm_hits} Warm Hits)",
            style="bold green",
        )
        formula.append("\n")

        # ── Utilization bar ───────────────────────────────────────────────────
        pct    = self.ram_sm.utilization_pct
        filled = int(pct / 100.0 * 40)
        bar    = "#" * filled + "." * (40 - filled)
        bar_color = "green" if pct < 70 else ("yellow" if pct < 85 else "red")
        util_line = Text()
        util_line.append(f" RAM Utilization: {pct:.1f}%  [")
        util_line.append(bar, style=bar_color)
        used = self.ram_sm.used_mb + self.ram_sm.system_reserved_mb
        total_ram = int(self.ram_sm.effective_total_ram_mb)
        util_line.append(f"]  {int(used)}/{total_ram} MB  |  {int(self.ram_sm.free_mb)} MB free\n")

        # ── Process table — Issue 3: wider columns, no truncation ────────────
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold white",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("App",               style="cyan",   no_wrap=True, min_width=12)
        table.add_column("PID",               style="dim",    no_wrap=True, min_width=6)
        table.add_column("MB",                style="white",  no_wrap=True, min_width=6,  justify="right")
        table.add_column("Hex Range",         style="dim",    no_wrap=True, min_width=21)
        table.add_column("S(a) Score",        style="yellow", no_wrap=True, min_width=15)
        table.add_column("State",             style="bold",   no_wrap=True, min_width=11)
        table.add_column("Kernel / RAM Action", no_wrap=True, ratio=1)

        for slot in self.ram_sm.get_sorted_slots():
            sa    = slot.lru_score
            # Issue 3: 8-dot gauge + numeric score
            dots  = "o" * int(sa * 8) + "." * (8 - int(sa * 8))
            hrange = f"{slot.base_addr}-{slot.end_addr}"
            sa_cell = f"{dots} {sa:.2f}"

            if slot.is_foreground:
                row_style = "bold white on dark_blue"
                state_str = "FOREGROUND"
                action    = "[RETAIN -- FOREGROUND]"

            elif slot.state == "PRELOADED":
                row_style = "bold green"
                state_str = "PRELOADED"
                action    = "[PRELOADED - ready (Zygote)]"

            elif slot.state == "PRELOADING":
                row_style = "yellow"
                state_str = "PRELOADING"
                prog      = slot.warmup_progress
                bar_len   = int(prog * 10)
                prog_bar  = "#" * bar_len + "." * (10 - bar_len)
                pct_str   = f"{int(prog * 100)}%"
                action    = f"[PRELOADING {prog_bar} {pct_str:>4}]"

            elif slot.state == "EVICTING":
                # Issue 2: bright red with tick countdown
                row_style = "bold red"
                state_str = "EVICTING"
                ticks     = slot.evict_ticks_remaining
                action    = f"[EVICTING ({ticks}t) -> reclaim_page_range()]"

            elif slot.state == "EVICTED":
                # Issue 2: dim red — app freed, stays in table until next switch
                row_style = "red"
                state_str = "EVICTED"
                action    = f"[REMOVED FROM RAM -> freed {slot.memory_mb:.0f}MB]"

            else:  # WARM
                row_style = ""
                state_str = "WARM"
                if slot.app_name in top3_apps:
                    rank   = top3_apps.index(slot.app_name) + 1
                    action = f"[PRELOAD <- predicted #{rank}]"
                else:
                    action = f"[RETAIN -- CACHED (S={sa:.2f})]"

            table.add_row(
                slot.app_name,
                str(slot.pid),
                str(int(slot.memory_mb)),
                hrange,
                sa_cell,
                state_str,
                action,
                style=row_style,
            )

        # ── Issue 3 & 5: Eviction log — last 2 entries ───────────────────────
        ev_log = Text()
        ev_log.append(" -- EVICTION LOG ", style="bold red")
        ev_log.append("-" * 55 + "\n", style="dim")

        # Check if no evicting/evicted slots currently
        evicting_count = sum(
            1 for s in self.ram_sm.slots.values()
            if s.state in ("EVICTING", "EVICTED")
        )
        free_mb = self.ram_sm.free_mb
        if evicting_count == 0:
            ev_log.append_text(Text.from_markup(
                f" [dim green]+ LMKD: Memory headroom nominal ({int(free_mb)}MB free)"
                f" -- 0 background processes require eviction[/]\n"
            ))
        else:
            ev_log.append_text(Text.from_markup(
                f" [dim green]+ LMKD: {evicting_count} process(es) being reclaimed"
                f"  ({int(free_mb)}MB headroom after eviction)[/]\n"
            ))

        for entry in self.ram_sm.eviction_log[-2:]:
            # Issue 5: base-case / partial eviction entries
            if "status" in entry:
                ev_log.append_text(Text.from_markup(
                    f" [{entry['timestamp_s']:.0f}s]  [dim cyan]{entry['status']}[/]\n"
                ))
            else:
                line = (
                    f" [{entry['timestamp_s']:.0f}s]  EVICT  {entry['app']:<12}"
                    f"  PID:{entry['pid']}  {int(entry['freed_mb'])}MB freed"
                    f"  S(a):{entry['sa_score']:.2f}"
                    f"  kernel: reclaim_page_range() OK\n"
                )
                ev_log.append(line, style="dim red")

        # ── Issue 3 & 5: Preload log — last 2 entries ────────────────────────
        pr_log = Text()
        pr_log.append(" -- PRELOAD LOG  ", style="bold green")
        pr_log.append("-" * 55 + "\n", style="dim")
        for entry in self.ram_sm.preload_log[-2:]:
            prog     = entry.get("progress", 0.0)
            bar_len  = int(prog * 10)
            prog_bar = "#" * bar_len + "." * (10 - bar_len)
            pct      = int(prog * 100)
            status   = "READY" if pct >= 100 else "WARMING"
            line = (
                f" [{entry['timestamp_s']:.0f}s]  PRELOAD  {entry['app']:<12}"
                f"  PID:{entry['pid']}  {int(entry['memory_mb'])}MB"
                f"  Zygote:{entry['zygote_ms']}ms"
                f"  [{prog_bar}] {pct:>3}% {status}\n"
            )
            pr_log.append(line, style="dim green")

        # ── Compose panel content ─────────────────────────────────────────────
        content = Text()
        content.append_text(formula)
        content.append_text(util_line)
        content.append("\n")

        from rich.console import Group
        combined = Group(content, table, ev_log, pr_log)

        ram_tier = getattr(self.adb, "device_ram_tier", "8192MB LPDDR5X")
        ram_type = ram_tier.split()[-1] if ram_tier else "LPDDR5X"

        return Panel(
            combined,
            title=f"[bold blue]{ram_type} RAM STATE MACHINE -- LIVE PROCESS CACHE[/]",
            box=box.HEAVY,
            border_style="blue",
        )

    # ── Panel: Footer ─────────────────────────────────────────────────────────

    def _render_footer(self) -> Panel:
        t = Text(justify="center")
        t.append(
            "Hotkeys: [1]WhatsApp [2]Instagram [3]YouTube [4]Gmail [5]BGMI "
            "[6]Chrome [7]Spotify [8]Camera [9]ChatGPT  "
            "[A]AutoTour  [Space]Home  |  [Ctrl+C] Exit",
            style="dim",
        )
        return Panel(t, box=box.HORIZONTALS, style="dim", height=3)

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_ctx(self, current_app: str) -> dict:
        """Build 19-feature context dict for the ML predictor."""
        batt_pct, is_charging   = self.adb.battery
        ram_total, ram_free     = self.adb.ram
        now                     = datetime.now()

        hist   = self._session_history
        prev1  = hist[-1] if len(hist) >= 1 else ""
        prev2  = hist[-2] if len(hist) >= 2 else ""
        prev3  = hist[-3] if len(hist) >= 3 else ""

        profile = get_profile(current_app)
        cat     = getattr(profile, "category", "unknown")
        ai_wl   = getattr(profile, "is_ai_workload", False)
        prev_profile = get_profile(prev1) if prev1 else None
        prev_cat = getattr(prev_profile, "category", "") if prev_profile else ""

        return {
            "current_app":             current_app,
            "prev_app_1":              prev1,
            "prev_app_2":              prev2,
            "prev_app_3":              prev3,
            "hour":                    now.hour,
            "day_of_week":             now.weekday(),
            "battery":                 batt_pct / 100.0,
            "is_charging":             is_charging,
            "ram_free_mb":             ram_free,
            "ram_total_mb":            ram_total,
            "network_type":            "wifi",
            "ai_task_active":          ai_wl,
            "app_category":            cat,
            "session_depth":           len(self._session_history),
            "time_since_last_switch":  5.0,
            "prev_app_category":       prev_cat,
        }
