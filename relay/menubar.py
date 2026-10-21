"""Minimal macOS menu bar status UI for the relay.

Shows the relay state and offers pause / resume / stop controls. It reads the
status file the relay writes, so it can be started and stopped independently.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSVariableStatusItemLength,
)

from . import config as config_mod
from . import status as status_mod


class RelayMenuBar:
    def __init__(self, config: dict):
        self.config = config
        self.status_path = config_mod.path_of(config, "status_file")
        self.pid_file = self.status_path.parent / "relay.pid"
        self.pause_file = self.status_path.parent / "paused"
        self.status_bar = NSStatusBar.systemStatusBar()
        self.item = self.status_bar.statusItemWithLength_(NSVariableStatusItemLength)
        self.item.setTitle_("relay")
        self.menu = NSMenu.alloc().init()
        self.item.setMenu_(self.menu)
        self.lines: dict[str, NSMenuItem] = {}
        self._build()
        self._timer = None

    # ------------------------------------------------------------------- build
    def _build(self) -> None:
        for key, label in (
            ("state", "Relay: unknown"),
            ("chatgpt", "ChatGPT: unknown"),
            ("opencode", "OpenCode: unknown"),
            ("last", "Last exchange: never"),
            ("pending", "Pending: none"),
            ("errors", "Errors: none"),
        ):
            entry = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label, None, "")
            entry.setEnabled_(False)
            self.menu.addItem_(entry)
            self.lines[key] = entry
        self.menu.addItem_(NSMenuItem.separatorItem())
        pause = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Pause relay", None, "")
        pause.setTarget_(self)
        pause.setAction_("togglePause:")
        self.menu.addItem_(pause)
        self.pause_item = pause
        stop = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Stop relay", None, "")
        stop.setTarget_(self)
        stop.setAction_("stopRelay:")
        self.menu.addItem_(stop)
        restart = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Restart relay", None, "")
        restart.setTarget_(self)
        restart.setAction_("restartRelay:")
        self.menu.addItem_(restart)
        self.menu.addItem_(NSMenuItem.separatorItem())
        logs = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Show log", None, "")
        logs.setTarget_(self)
        logs.setAction_("showLog:")
        self.menu.addItem_(logs)

    # ------------------------------------------------------------------ actions
    def togglePause_(self, _sender) -> None:
        if self.pause_file.exists():
            self.pause_file.unlink(missing_ok=True)
        else:
            self.pause_file.write_text("paused\n")
        self.refresh()

    def stopRelay_(self, _sender) -> None:
        self._relayctl("stop")

    def restartRelay_(self, _sender) -> None:
        self._relayctl("stop")
        time.sleep(1)
        self._relayctl("start")

    def showLog_(self, _sender) -> None:
        log_path = config_mod.path_of(self.config, "log")
        subprocess.run(["/usr/bin/open", "-a", "Console", str(log_path)], check=False)

    def _relayctl(self, command: str) -> None:
        script = Path(__file__).resolve().parent.parent / "relayctl.py"
        subprocess.run(
            [sys.executable, str(script), command],
            check=False,
            env={**__import__("os").environ, "RELAY_CONFIG": str(config_mod.config_path())},
        )
        self.refresh()

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        payload = status_mod.read_status(self.status_path) or {}
        running = bool(payload.get("running"))
        paused = bool(payload.get("paused")) or self.pause_file.exists()
        self.lines["state"].setTitle_(
            f"Relay: {'paused' if paused else ('running' if running else 'stopped')}"
        )
        self.lines["chatgpt"].setTitle_(
            "ChatGPT: "
            + ("connected" if payload.get("conversation_connected") else "disconnected")
            + (f" ({payload.get('conversation')})" if payload.get("conversation") else "")
        )
        self.lines["opencode"].setTitle_(
            "OpenCode: "
            + ("attached" if payload.get("opencode_attached") else "not attached")
            + (f" ({payload.get('opencode_session')})" if payload.get("opencode_session") else "")
        )
        last = payload.get("last_exchange_at")
        if last:
            age = int(time.time() - float(last))
            self.lines["last"].setTitle_(
                f"Last exchange: {payload.get('last_status', '?')} ({age}s ago)"
            )
        else:
            self.lines["last"].setTitle_("Last exchange: never")
        self.lines["pending"].setTitle_(
            f"Pending: {payload.get('pending_task') or 'none'}"
        )
        errors = payload.get("errors") or []
        self.lines["errors"].setTitle_(
            f"Errors: {errors[-1][:60]}" if errors else "Errors: none"
        )
        self.pause_item.setTitle_("Resume relay" if paused else "Pause relay")
        self.item.setTitle_("relay" if running else "relay (off)")


def run() -> int:
    config = config_mod.load_config()
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    menubar = RelayMenuBar(config)

    from Foundation import NSTimer

    class _Ticker:
        """NSTimer needs a target/selector pair."""

        def __init__(self, bar: RelayMenuBar):
            self.bar = bar

        def tick_(self, _timer) -> None:
            self.bar.refresh()

    ticker = _Ticker(menubar)
    menubar._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        2.0, ticker, "tick:", None, True
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
