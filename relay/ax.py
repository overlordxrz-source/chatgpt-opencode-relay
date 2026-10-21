"""Low-level macOS Accessibility (AXUIElement) helpers for driving the ChatGPT app.

Everything here uses only public accessibility APIs: AXUIElement attribute
copies, AXUIElementPerformAction and AXUIElementCopyElementAtPosition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

import Quartz
from AppKit import NSRunningApplication
from ApplicationServices import (
    AXUIElementCopyAttributeNames,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyElementAtPosition,
    AXUIElementCreateApplication,
    AXUIElementPerformAction,
    AXUIElementSetAttributeValue,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)

CHATGPT_BUNDLE_IDS = ("com.openai.codex", "com.openai.chat")

AX_PRESS = "AXPress"
AX_SET_VALUE = "AXSetValue"


def _attr(element, name: str, default: Any = None) -> Any:
    try:
        err, value = AXUIElementCopyAttributeValue(element, name, None)
    except Exception:
        return default
    if err != 0:
        return default
    return value


def _attr_names(element) -> list[str]:
    try:
        err, names = AXUIElementCopyAttributeNames(element, None)
    except Exception:
        return []
    return list(names) if err == 0 else []


def _point(value) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    try:
        x, y = AXValueGetValue(value, kAXValueCGPointType, None)
        return (float(x), float(y))
    except Exception:
        return None


def _size(value) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    try:
        w, h = AXValueGetValue(value, kAXValueCGSizeType, None)
        return (float(w), float(h))
    except Exception:
        return None


@dataclass
class Node:
    """A snapshot of one accessibility element."""

    element: Any
    role: str
    title: str = ""
    value: str = ""
    description: str = ""
    identifier: str = ""
    help_text: str = ""
    position: Optional[tuple[float, float]] = None
    size: Optional[tuple[float, float]] = None
    children: list["Node"] = field(default_factory=list)
    parent: Optional["Node"] = None

    @property
    def text(self) -> str:
        """Best-effort human readable text of the node."""
        for candidate in (self.value, self.title, self.description):
            if candidate:
                return candidate
        return ""

    def matches(self, role: Optional[str] = None, text: Optional[str] = None, contains: Optional[str] = None) -> bool:
        if role and self.role != role:
            return False
        if text is not None and self.text != text:
            return False
        if contains is not None and contains.lower() not in self.text.lower():
            return False
        return True

    def find_all(self, role: Optional[str] = None, contains: Optional[str] = None, exact_text: Optional[str] = None) -> list["Node"]:
        out: list[Node] = []
        stack = [self]
        while stack:
            node = stack.pop()
            if (role is None or node.role == role) and (
                contains is None or contains.lower() in node.text.lower()
            ) and (exact_text is None or node.text == exact_text):
                out.append(node)
            stack.extend(reversed(node.children))
        return out

    def find(self, role: Optional[str] = None, contains: Optional[str] = None, exact_text: Optional[str] = None) -> Optional["Node"]:
        found = self.find_all(role=role, contains=contains, exact_text=exact_text)
        return found[0] if found else None


def _snapshot(element, depth: int, max_depth: int, budget: list[int]) -> Node:
    budget[0] -= 1
    node = Node(
        element=element,
        role=str(_attr(element, "AXRole", "") or ""),
        title=str(_attr(element, "AXTitle", "") or ""),
        value=str(_attr(element, "AXValue", "") or ""),
        description=str(_attr(element, "AXDescription", "") or ""),
        identifier=str(_attr(element, "AXIdentifier", "") or ""),
        help_text=str(_attr(element, "AXHelp", "") or ""),
        position=_point(_attr(element, "AXPosition")),
        size=_size(_attr(element, "AXSize")),
    )
    if depth >= max_depth or budget[0] <= 0:
        return node
    raw_children = _attr(element, "AXChildren", []) or []
    for child in list(raw_children)[:400]:
        snapshot = _snapshot(child, depth + 1, max_depth, budget)
        snapshot.parent = node
        node.children.append(snapshot)
    return node


class ChatGPTApp:
    """Handle on the running ChatGPT desktop application."""

    def __init__(self, bundle_ids: Iterable[str] = CHATGPT_BUNDLE_IDS):
        self.bundle_ids = tuple(bundle_ids)
        self._app: Any = None
        self._pid: Optional[int] = None

    # ---------------------------------------------------------------- lifecycle
    def is_running(self) -> bool:
        for bundle in self.bundle_ids:
            if NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle):
                return True
        return False

    def _resolve(self) -> Any:
        if self._app is not None:
            return self._app
        for bundle in self.bundle_ids:
            running = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle)
            if running:
                self._pid = int(running[0].processIdentifier())
                self._app = AXUIElementCreateApplication(self._pid)
                return self._app
        raise RuntimeError("ChatGPT desktop app is not running")

    def activate(self) -> bool:
        for bundle in self.bundle_ids:
            running = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle)
            if running:
                running[0].activateWithOptions_(1 << 1)  # activateIgnoringOtherApps
                return True
        return False

    def is_frontmost(self) -> bool:
        """Is the ChatGPT app the active application?"""
        try:
            from AppKit import NSWorkspace

            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return bool(app) and app.bundleIdentifier() in self.bundle_ids
        except Exception:
            return False

    @staticmethod
    def seconds_since_last_input() -> float:
        """Seconds since the last system-wide user input event."""
        try:
            return float(
                Quartz.CGEventSourceSecondsSinceLastEventType(
                    Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGAnyInputEventType
                )
            )
        except Exception:
            return 0.0

    @staticmethod
    def screen_locked() -> bool:
        """True when the screen is locked (the app cannot render)."""
        try:
            session = Quartz.CGSessionCopyCurrentDictionary()
            return bool(session and session.get("CGSSessionScreenIsLocked"))
        except Exception:
            return False

    def cpu_percent(self) -> float:
        """CPU usage of the app, in percent."""
        try:
            if self._pid is None:
                self._resolve()
            for process in NSRunningApplication.runningApplicationsWithBundleIdentifier_(
                self.bundle_ids[0]
            ):
                if int(process.processIdentifier()) == self._pid:
                    return float(process.cpuUsage())
        except Exception:
            pass
        return 0.0

    def restart(self) -> bool:
        """Quit and reopen the app (used only by the wedged-app watchdog)."""
        try:
            for process in NSRunningApplication.runningApplicationsWithBundleIdentifier_(
                self.bundle_ids[0]
            ):
                process.terminate()
            time.sleep(4)
            for process in NSRunningApplication.runningApplicationsWithBundleIdentifier_(
                self.bundle_ids[0]
            ):
                process.terminate()
            time.sleep(2)
            import subprocess

            subprocess.run(["open", "-a", "ChatGPT"], check=False)
            time.sleep(6)
            return True
        except Exception:
            return False

    def ax_window_title(self) -> str:
        """Title of the window the accessibility tree is describing.

        The app can keep several windows (one per conversation) open, and the
        window server may report a different one from the one in front. The
        accessibility tree is what the relay actually reads, so it is the
        authority for "which conversation am I looking at".
        """
        app = self._resolve()
        for attribute in ("AXFocusedWindow", "AXMainWindow"):
            err, value = AXUIElementCopyAttributeValue(app, attribute, None)
            if err == 0 and value is not None:
                err2, title = AXUIElementCopyAttributeValue(value, "AXTitle", None)
                if err2 == 0 and title:
                    return str(title)
        return ""

    def visible_window_title(self) -> str:
        """Title of the frontmost on-screen window of this app."""
        pid = self._pid
        if pid is None:
            self._resolve()
            pid = self._pid
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        for window in windows:
            if window.get("kCGWindowOwnerPID") == pid and window.get("kCGWindowName"):
                return str(window["kCGWindowName"])
        return ""

    def window_bounds(self) -> Optional[tuple[float, float, float, float]]:
        """Screen coordinates (x, y, w, h) of the app's main window."""
        pid = self._pid
        if pid is None:
            self._resolve()
            pid = self._pid
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        for window in windows:
            if window.get("kCGWindowOwnerPID") == pid and window.get("kCGWindowName"):
                bounds = window["kCGWindowBounds"]
                return (bounds["X"], bounds["Y"], bounds["Width"], bounds["Height"])
        return None

    # -------------------------------------------------------------------- tree
    def tree(self, max_depth: int = 34, budget: int = 6000) -> Node:
        app = self._resolve()
        return _snapshot(app, 0, max_depth, [budget])

    def window(self) -> Optional[Node]:
        app = self._resolve()
        wins = _attr(app, "AXWindows", []) or []
        if not wins:
            return None
        return _snapshot(list(wins)[0], 0, 34, [6000])

    def focused_element(self) -> Optional[Node]:
        app = self._resolve()
        element = _attr(app, "AXFocusedUIElement")
        if element is None:
            return None
        return _snapshot(element, 0, 3, [50])

    # ------------------------------------------------------------------ actions
    def press(self, node: Node) -> bool:
        try:
            err = AXUIElementPerformAction(node.element, AX_PRESS)
        except Exception:
            return False
        return err == 0

    def set_focus(self, node: Node) -> bool:
        """Give keyboard focus to an element (app level AXFocusedUIElement)."""
        app = self._resolve()
        try:
            err = AXUIElementSetAttributeValue(app, "AXFocusedUIElement", node.element)
        except Exception:
            return False
        return err == 0

    def set_value(self, node: Node, value: str) -> bool:
        try:
            err = AXUIElementSetAttributeValue(node.element, "AXValue", value)
        except Exception:
            return False
        return err == 0

    def element_at(self, x: float, y: float) -> Optional[Node]:
        app = self._resolve()
        try:
            err, element = AXUIElementCopyElementAtPosition(app, float(x), float(y), None)
        except Exception:
            return None
        if err != 0 or element is None:
            return None
        return _snapshot(element, 0, 2, [30])

    def click(self, x: float, y: float, click_count: int = 1) -> None:
        for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            event = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_count)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.03)

    def type_text(self, text: str) -> None:
        """Type text into whatever currently has keyboard focus."""
        script = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(script, len(text), text)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, script)
        time.sleep(0.02)

    def paste_text(self, text: str) -> bool:
        """Paste text into the focused element using real key events.

        Setting an accessibility value on a React-controlled editor does not
        fire the input events the app listens to, so the app never sees the
        text and refuses to submit it. A real paste does.
        """
        try:
            from AppKit import NSPasteboard, NSStringPboardType

            pasteboard = NSPasteboard.generalPasteboard()
            pasteboard.clearContents()
            pasteboard.setString_forType_(text, NSStringPboardType)
        except Exception:
            return False
        self.keystroke(9, command=True)  # cmd+v
        time.sleep(0.25)
        return True

    def clear_focused(self) -> None:
        """Select all and delete, so a leftover draft does not get appended."""
        self.keystroke(0, command=True)   # cmd+a
        time.sleep(0.05)
        self.keystroke(51)               # delete/backspace
        time.sleep(0.15)

    def keystroke(self, keycode: int, command: bool = False) -> None:
        flags = Quartz.kCGEventFlagMaskCommand if command else 0
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
            Quartz.CGEventSetFlags(event, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.02)

    def press_return(self) -> None:
        """Press the Return key."""
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, 36, down)  # return
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.02)
