"""Status file helpers - a tiny JSON snapshot other processes can read."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = time.time()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def read_status(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def render_status(payload: Optional[dict[str, Any]]) -> str:
    if not payload:
        return "relay: no status (not running?)"
    running = "running" if payload.get("running") else "stopped"
    paused = ", PAUSED" if payload.get("paused") else ""
    lines = [f"relay: {running}{paused}"]
    lines.append(
        "chatgpt: "
        + ("connected" if payload.get("conversation_connected") else "disconnected")
        + (f" ({payload.get('conversation')})" if payload.get("conversation") else "")
    )
    lines.append(
        "opencode: "
        + ("attached" if payload.get("opencode_attached") else "not attached")
        + (f" ({payload.get('opencode_session')})" if payload.get("opencode_session") else "")
    )
    last = payload.get("last_exchange_at")
    if last:
        age = int(time.time() - float(last))
        lines.append(
            f"last exchange: {payload.get('last_status', '?')} ({age}s ago, {payload.get('exchanges_today', 0)} today)"
        )
    else:
        lines.append("last exchange: never")
    if payload.get("pending_task"):
        lines.append(f"pending: {payload['pending_task']}")
    errors = payload.get("errors") or []
    if errors:
        lines.append(f"errors: {len(errors)} (last: {errors[-1][:120]})")
    return "\n".join(lines)
