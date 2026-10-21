#!/usr/bin/env python3
"""Live correlation check (marker: stale-reply-debug-20261021).

Delivers a unique prompt to a real OpenCode session through the relay's own
code path and asserts the reply the relay picks is the answer to *that*
prompt, not an earlier turn's text.

Usage: .venv/bin/python scripts/live_correlation_check.py
"""
import json
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from relay.controller import RelayConfig, RelayController  # noqa: E402
from relay.opencode import OpenCodeAdapter  # noqa: E402
from relay.store import Store  # noqa: E402

MARKER = "stale-reply-debug-20261021"
BRIDGE = pathlib.Path("/Users/overlord/tools/opencode-chatgpt-bridge-v2")


def bridge_token() -> str:
    env = dict(
        line.split("=", 1)
        for line in (BRIDGE / ".env").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    return env["OPENCODE_BRIDGE_TOKEN"].strip()


class NoChat:
    """The relay never posts to ChatGPT in this check."""

    def send(self, text: str) -> bool:
        print(f"    would post to ChatGPT: {text[:120]!r}")
        return True


def main() -> int:
    token = bridge_token()
    opencode = OpenCodeAdapter("http://127.0.0.1:8787/mcp", token)

    repo = os.path.expanduser("~/.opencode-chatgpt-bridge/relay-test-repo")
    os.makedirs(repo, exist_ok=True)
    created = opencode.create_session(repo, "live correlation check")
    session_id = created["opencodeSession"]["id"]
    print(f"  test session: {session_id}")

    config = RelayConfig(
        conversation="live",
        conversations=["live"],
        bridge_url="http://127.0.0.1:8787/mcp",
        bridge_env=str(BRIDGE / ".env"),
        default_session=session_id,
        allowed_sessions=[session_id],
        target_session=session_id,
        agent_timeout_ms=180_000,
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(os.path.join(tmp, "state.db"))
        controller = RelayController(config, NoChat(), opencode, store, log=lambda m: print("   ", m))

        unique = "live-correlation-ok-9317"
        attached = opencode.attach(session_id)
        bridge_session_id = attached["bridgeSession"]["bridgeSessionId"]
        accepted = opencode.send(bridge_session_id, f"Reply with exactly: {unique}", wait=False)
        entry = accepted.get("accepted") or {}
        request_created_at = float((entry.get("time") or {}).get("created") or 0)
        print(f"  delivered prompt {entry.get('id')} at {request_created_at}")

        text, status = controller._await_agent_reply(
            bridge_session_id, entry.get("id"), "Reply with exactly: " + unique, request_created_at
        )
        print(f"  relay picked: status={status} text={text[:80]!r}")

        ok = unique in text and status == "completed"
        print(f"\n{MARKER}: live correlation {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
        # A second, different prompt must not reuse the first answer.
        second = "live-correlation-two-4822"
        accepted2 = opencode.send(bridge_session_id, f"Reply with exactly: {second}", wait=False)
        entry2 = accepted2.get("accepted") or {}
        created2 = float((entry2.get("time") or {}).get("created") or 0)
        text2, status2 = controller._await_agent_reply(
            bridge_session_id, entry2.get("id"), "Reply with exactly: " + second, created2
        )
        print(f"  second relay picked: status={status2} text={text2[:80]!r}")
        ok2 = second in text2 and unique not in text2
        print(f"{MARKER}: distinct second answer {'PASS' if ok2 else 'FAIL'}")
        return 0 if ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
