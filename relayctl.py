#!/usr/bin/env python3
"""relayctl - control the ChatGPT <-> OpenCode desktop relay."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relay import config as config_mod
from relay import protocol, status as status_mod
from relay.ax import ChatGPTApp
from relay.chatgpt import ChatGPTAdapter
from relay.controller import RelayConfig, RelayController
from relay.opencode import OpenCodeAdapter
from relay.store import Store

BASE_DIR = Path(__file__).resolve().parent
PAUSE_FILE_NAME = "paused"


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)


def _class_logger(log_path: Path):
    handle = open(log_path, "a", buffering=1)

    def log(message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} {message}"
        handle.write(line + "\n")
        print(line, flush=True)

    return log


def _build(config_data: dict):
    relay_config = RelayConfig.from_dict(config_data)
    token = config_mod.bridge_token(config_data)
    app = ChatGPTApp()
    chatgpt = ChatGPTAdapter(
        app,
        relay_config.conversation,
        log=_log,
        index=relay_config.conversation_index,
        debug=bool(os.environ.get("RELAY_DEBUG")),
    )
    opencode = OpenCodeAdapter(relay_config.bridge_url, token, log=_log)
    store = Store(config_mod.path_of(config_data, "state_db"))
    def _rename(title: str) -> None:
        try:
            data = config_mod.load_config()
            data["chatgpt"]["conversation"] = title
            config_mod.save_config(data)
        except Exception as error:  # noqa: BLE001
            _log(f"relay: could not persist the renamed conversation: {error}")

    controller = RelayController(
        relay_config, chatgpt, opencode, store, log=_log, on_conversation_renamed=_rename
    )
    return relay_config, chatgpt, opencode, store, controller


def _ensure_bridge(config_data: dict) -> bool:
    """Make sure the bridge MCP server is up; start it if the script exists."""
    token = config_mod.bridge_token(config_data)
    url = config_data["opencode"]["bridge_url"]
    probe = OpenCodeAdapter(url, token, log=_log)
    try:
        probe.health()
        return True
    except Exception as error:  # noqa: BLE001
        _log(f"relay: bridge not reachable ({error}); trying to start it")
    env_path = Path(config_data["opencode"]["bridge_env"]).expanduser()
    starter = env_path.parent / "scripts" / "bridge-start.sh"
    if starter.exists():
        subprocess.run([str(starter)], cwd=str(env_path.parent), check=False)
        time.sleep(2)
        try:
            probe.health()
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


# --------------------------------------------------------------------- commands
def cmd_init(args) -> int:
    path = config_mod.config_path()
    if path.exists() and not args.force:
        print(f"config already exists: {path}")
    else:
        config_mod.save_config(config_mod.load_config(path), path)
        print(f"wrote {path}")
    return 0


def cmd_status(args) -> int:
    data = config_mod.load_config()
    path = config_mod.path_of(data, "status_file")
    payload = status_mod.read_status(path)
    if args.json:
        print(json.dumps(payload or {}, indent=2))
    else:
        print(status_mod.render_status(payload))
    return 0


def cmd_run(args) -> int:
    data = config_mod.load_config()
    log_path = config_mod.path_of(data, "log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = _class_logger(log_path)
    status_path = config_mod.path_of(data, "status_file")
    pause_file = status_path.parent / PAUSE_FILE_NAME

    relay_config, chatgpt, opencode, store, controller = _build(data)
    if not relay_config.conversation:
        log("relay: no chatgpt.conversation configured - run 'relayctl new-test-chat' or edit the config")
        return 2
    if not _ensure_bridge(data):
        log("relay: the opencode bridge is not running; start it and retry")
        return 3

    stopping = {"flag": False}

    # Keep the Mac awake while the relay runs: the ChatGPT app has to be
    # rendering for the accessibility tree to update. It is a child of the
    # relay, so it dies with it. RELAY_NO_CAFFEINATE=1 turns it off.
    caffeinate = None
    if not os.environ.get("RELAY_NO_CAFFEINATE") and shutil.which("caffeinate"):
        caffeinate = subprocess.Popen(
            ["caffeinate", "-i"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        log(f"relay: holding the idle assertion (caffeinate pid {caffeinate.pid})")

    def handle_signal(signum, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"relay: starting (conversation={relay_config.conversation!r})")

    seeded = False
    while not stopping["flag"]:
        try:
            if not seeded:
                # Seed every watched conversation. A command that is already
                # waiting (the newest assistant message with a command block)
                # is deliberately left unprocessed so that it still runs.
                # Seed only the conversation that is open: the relay answers
                # in the conversation a request came from, so it never needs to
                # move the app on startup.
                open_title = chatgpt.current_title()
                seed_targets = (
                    [open_title] if open_title in (relay_config.conversations or []) else []
                )
                for conversation in seed_targets:
                    if not conversation:
                        continue
                    chatgpt.conversation = conversation
                    messages = chatgpt.read_conversation(conversation)
                    pending = None
                    for message in messages:
                        if message.role == "assistant" and protocol.extract_commands(
                            message.text, relay_config.strict_protocol
                        ):
                            pending = message
                    keys = [m.key for m in messages]
                    for index, message in enumerate(messages):
                        if pending is not None and message.key == pending.key:
                            continue
                        store.mark_processed(
                            message.key, message.role, prev=keys[index - 1] if index else ""
                        )
                    log(
                        f"relay: seeded {conversation!r} with {len(messages)} messages"
                        + (" (1 pending command kept)" if pending else "")
                    )
                seeded = True
                if relay_config.post_init_prompt and seed_targets:
                    try:
                        allowed = controller.allowed_session_list()
                    except Exception as error:  # noqa: BLE001
                        log(f"relay: could not list allowed sessions: {error}")
                        allowed = []
                    log(f"relay: posting the initialisation prompt ({len(allowed)} allowed session(s))")
                    chatgpt.conversation = seed_targets[0]
                    chatgpt.send(protocol.build_init_prompt(allowed))

            paused = pause_file.exists()
            controller.status.paused = paused
            controller.status.running = True
            if not paused:
                controller.tick()
            controller.status.exchanges_today = store.count_since(86400)
            payload = {
                "running": True,
                "paused": paused,
                "conversation": relay_config.conversation,
                "conversation_connected": controller.status.conversation_connected,
                "opencode_attached": controller.status.opencode_attached,
                "opencode_session": controller.status.opencode_session,
                "last_exchange_at": controller.status.last_exchange_at,
                "last_status": controller.status.last_status,
                "pending_task": controller.status.pending_task,
                "errors": controller.status.errors[-5:],
                "exchanges_today": controller.status.exchanges_today,
                "pid": os.getpid(),
            }
            status_mod.write_status(status_path, payload)
        except Exception as error:  # noqa: BLE001 - keep the relay alive
            log(f"relay: tick error: {type(error).__name__}: {error}")
            time.sleep(3)
        time.sleep(relay_config.poll_seconds)

    log("relay: stopped")
    if caffeinate is not None:
        caffeinate.terminate()
    controller.status.running = False
    status_mod.write_status(
        status_path,
        {
            "running": False,
            "paused": pause_file.exists(),
            "conversation": relay_config.conversation,
            "conversation_connected": False,
            "opencode_attached": controller.status.opencode_attached,
            "opencode_session": controller.status.opencode_session,
            "last_exchange_at": controller.status.last_exchange_at,
            "last_status": controller.status.last_status,
            "pending_task": "",
            "errors": controller.status.errors[-5:],
            "exchanges_today": store.count_since(86400),
            "pid": os.getpid(),
        },
    )
    return 0


def cmd_start(args) -> int:
    data = config_mod.load_config()
    state_dir = config_mod.path_of(data, "status_file").parent
    state_dir.mkdir(parents=True, exist_ok=True)
    pid_file = state_dir / "relay.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().strip() or 0)
        if pid and _alive(pid):
            print(f"relay already running (pid {pid})")
            return 0
        pid_file.unlink()
    log_file = state_dir / "relay.out"
    env = dict(os.environ)
    env["RELAY_CONFIG"] = str(config_mod.config_path())
    python = str(BASE_DIR / ".venv" / "bin" / "python")
    if not os.path.exists(python):
        python = sys.executable
    # Make sure no older relay is still around.
    subprocess.run(["pkill", "-f", "relayctl.py run"], check=False)
    subprocess.run(["pkill", "-f", "caffeinate -i"], check=False)
    time.sleep(0.5)

    command = [python, str(BASE_DIR / "relayctl.py"), "run"]
    with open(log_file, "a") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    pid_file.write_text(str(process.pid))
    print(f"relay started (pid {process.pid})")
    return 0


def cmd_stop(args) -> int:
    data = config_mod.load_config()
    state_dir = config_mod.path_of(data, "status_file").parent
    pid_file = state_dir / "relay.pid"
    if not pid_file.exists():
        print("relay is not running")
        return 0
    pid = int(pid_file.read_text().strip() or 0)
    if pid and _alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not _alive(pid):
                break
            time.sleep(0.5)
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
    pid_file.unlink(missing_ok=True)
    print("relay stopped")
    return 0


def cmd_pause(args) -> int:
    data = config_mod.load_config()
    state_dir = config_mod.path_of(data, "status_file").parent
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / PAUSE_FILE_NAME).write_text("paused\n")
    print("relay paused")
    return 0


def cmd_resume(args) -> int:
    data = config_mod.load_config()
    state_dir = config_mod.path_of(data, "status_file").parent
    (state_dir / PAUSE_FILE_NAME).unlink(missing_ok=True)
    print("relay resumed")
    return 0


def cmd_logs(args) -> int:
    data = config_mod.load_config()
    path = config_mod.path_of(data, "log")
    if not path.exists():
        print("no log file yet")
        return 0
    lines = path.read_text(errors="replace").splitlines()
    for line in lines[-args.lines :]:
        print(line)
    return 0


def cmd_new_test_chat(args) -> int:
    """Create a disposable conversation and print its title."""
    app = ChatGPTApp()
    app.activate()
    time.sleep(1.0)
    tree = app.tree()
    button = tree.find(role="AXButton", exact_text="New chat")
    if button is None:
        print("could not find the 'New chat' button", file=sys.stderr)
        return 1
    app.press(button)
    time.sleep(2.0)
    window = app.window()
    title = window.title if window else ""
    print(f"new conversation title: {title!r}")
    if args.save:
        data = config_mod.load_config()
        data["chatgpt"]["conversation"] = title
        config_mod.save_config(data)
        print(f"saved to {config_mod.config_path()}")
    return 0


def cmd_bootstrap_chat(args) -> int:
    """Create a disposable conversation, teach it the protocol, remember its title."""
    data = config_mod.load_config()
    token = config_mod.bridge_token(data)
    opencode = OpenCodeAdapter(data["opencode"]["bridge_url"], token, log=_log)
    allowed = list(data["opencode"].get("allowed_sessions") or [])
    if not allowed and data["opencode"].get("default_session"):
        allowed = [data["opencode"]["default_session"]]
    try:
        sessions = [
            s
            for s in opencode.list_sessions(limit=50)
            if any(
                str(s.get("opencodeSessionId", "")).startswith(entry.strip())
                or entry.strip().lower() in str(s.get("title", "")).lower()
                for entry in allowed
            )
        ]
    except Exception as error:  # noqa: BLE001
        print(f"could not list opencode sessions: {error}", file=sys.stderr)
        sessions = []
    app = ChatGPTApp()
    app.activate()
    time.sleep(1.0)
    tree = app.tree()
    button = tree.find(role="AXButton", exact_text="New chat")
    if button is None:
        print("could not find the 'New chat' button", file=sys.stderr)
        return 1
    app.press(button)
    time.sleep(2.0)
    window = app.window()
    title = window.title if window else ""
    print(f"created conversation (title now {title!r})")

    chatgpt = ChatGPTAdapter(app, title, log=_log)
    prompt = protocol.build_init_prompt(sessions)
    if not chatgpt.send(prompt):
        print("could not post the initialization message", file=sys.stderr)
        return 1
    print("posted the initialization message; waiting for the title to settle...")
    deadline = time.time() + 120
    final = title
    while time.time() < deadline:
        time.sleep(4)
        window = app.window()
        if window and window.title and window.title != title:
            final = window.title
            break
    print(f"conversation title: {final!r}")

    # Give the conversation a stable name: the app renames chats on its own,
    # which would break the relay's ability to find it again.
    tree = app.tree()
    existing = {b.text.strip() for b in tree.find_all(role="AXButton")}
    base = getattr(args, "title", "") or "OpenCode Relay"
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base} {suffix}"
        suffix += 1
    if candidate != final:
        if chatgpt.rename_conversation(candidate):
            final = candidate
            print(f"renamed to {final!r}")
        else:
            print("could not rename the conversation", file=sys.stderr)

    data["chatgpt"]["conversation"] = final
    data["chatgpt"]["conversation_index"] = 0
    config_mod.save_config(data)
    print(f"saved to {config_mod.config_path()}")
    return 0


def cmd_sessions(args) -> int:
    """List OpenCode sessions and whether the relay may drive them."""
    data = config_mod.load_config()
    token = config_mod.bridge_token(data)
    opencode = OpenCodeAdapter(data["opencode"]["bridge_url"], token, log=_log)
    allowed = list(data["opencode"].get("allowed_sessions") or [])
    default = data["opencode"].get("default_session", "")
    if default and default not in allowed:
        allowed.append(default)
    sessions = opencode.list_sessions(limit=100)
    print(f"allowed_sessions: {allowed or '(none)'}\n")
    for session in sessions:
        sid = str(session.get("opencodeSessionId", ""))
        title = str(session.get("title", ""))
        ok = any(
            sid.startswith(entry.strip()) or entry.strip().lower() in title.lower()
            for entry in allowed
            if entry.strip()
        )
        mark = "ALLOW" if ok else "  -  "
        print(f"  {mark}  {sid[:18]}…  {title[:52]}  ({session.get('directory')})")
    if args.allow:
        target = args.allow.strip()
        for session in sessions:
            sid = str(session.get("opencodeSessionId", ""))
            title = str(session.get("title", ""))
            if target == sid or target.lower() in title.lower() or target.lower() in sid.lower():
                # Store the full session id so the entry is unambiguous.
                entry = sid if target.lower() in sid.lower() else target
                if entry not in allowed:
                    allowed.append(entry)
                    data["opencode"]["allowed_sessions"] = allowed
                    config_mod.save_config(data)
                    print(f"\nallowed now: {allowed}")
                else:
                    print(f"\nalready allowed: {allowed}")
                return 0
        print(f"no session matched {target!r}")
        return 1
    return 0


def cmd_selftest(args) -> int:
    """Checkpoint A/B: read a message, send a message, exercise OpenCode."""
    data = config_mod.load_config()
    relay_config, chatgpt, opencode, store, controller = _build(data)
    failures = 0

    print("== checkpoint A: chatgpt read ==")
    messages = chatgpt.read_messages()
    print(f"messages read: {len(messages)}")
    if not messages:
        print("FAIL: no messages read")
        failures += 1
    else:
        roles = {m.role for m in messages}
        print(f"roles seen: {sorted(roles)}")
        print(f"latest: [{messages[-1].role}] {messages[-1].text[:100]!r}")
        if not {"user", "assistant"} & roles:
            print("FAIL: no roles detected")
            failures += 1

    print("\n== checkpoint A2: chatgpt send ==")
    if args.send:
        ok = chatgpt.send(args.send)
        print(f"send returned {ok}")
        if not ok:
            print("FAIL: could not send")
            failures += 1
    else:
        print("skipped (pass --send TEXT to run it)")

    print("\n== checkpoint B: opencode attach + round trip ==")
    try:
        health = opencode.health()
        print(f"bridge ok={health.get('ok')} api={health.get('opencodeApi')}")
        sessions = opencode.list_sessions(limit=10)
        print(f"sessions: {len(sessions)}")
        for session in sessions[:5]:
            print(f"   {str(session.get('opencodeSessionId'))[:16]}… {str(session.get('title'))[:40]}")
        if args.session:
            wanted = args.session
            match = next(
                (s for s in sessions if wanted.lower() in str(s.get("title", "")).lower()
                 or str(s.get("opencodeSessionId", "")).startswith(wanted)),
                None,
            )
            if match is None:
                print(f"FAIL: no session matching {wanted!r}")
                failures += 1
            else:
                attached = opencode.attach(match["opencodeSessionId"])
                bridge_id = attached["bridgeSession"]["bridgeSessionId"]
                print(f"attached: {bridge_id}")
                if args.prompt:
                    result = opencode.send(bridge_id, args.prompt, wait=True, timeout_ms=180000)
                    completion = result.get("completion") or {}
                    texts = [
                        m.get("text", "")
                        for m in (completion.get("messages") or [])
                        if m.get("role") == "assistant"
                    ]
                    print(f"completed={completion.get('completed')} status={completion.get('outcome')}")
                    print(f"reply: {(texts[-1] if texts else '')[:200]!r}")
    except Exception as error:  # noqa: BLE001
        print(f"FAIL: {type(error).__name__}: {error}")
        failures += 1

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write a default config").add_argument("--force", action="store_true")
    sub.add_parser("status", help="show the relay status").add_argument("--json", action="store_true")
    sub.add_parser("run", help="run the relay in the foreground")
    sub.add_parser("start", help="start the relay in the background")
    sub.add_parser("stop", help="stop the relay")
    sub.add_parser("pause", help="pause the relay")
    sub.add_parser("resume", help="resume the relay")
    sub.add_parser("menubar", help="show the menu bar status UI")
    sessions = sub.add_parser("sessions", help="list OpenCode sessions and the allow list")
    sessions.add_argument("--allow", help="also add a session (id or title) to the allow list")
    logs = sub.add_parser("logs", help="show recent log lines")
    logs.add_argument("-n", "--lines", type=int, default=40)
    boot = sub.add_parser("bootstrap-chat", help="create a disposable conversation and teach it the protocol")
    boot.add_argument("--title", default="OpenCode Relay", help="title for the new conversation")
    chat = sub.add_parser("new-test-chat", help="create a disposable conversation")
    chat.add_argument("--save", action="store_true", help="save it as the watched conversation")
    test = sub.add_parser("selftest", help="run the checkpoint tests")
    test.add_argument("--send", help="text to send into the conversation")
    test.add_argument("--session", help="session id or title to attach to")
    test.add_argument("--prompt", help="prompt to send to the OpenCode session")

    args = parser.parse_args()
    handlers = {
        "init": cmd_init,
        "status": cmd_status,
        "run": cmd_run,
        "start": cmd_start,
        "stop": cmd_stop,
        "pause": cmd_pause,
        "resume": cmd_resume,
        "logs": cmd_logs,
        "menubar": lambda _args: __import__("relay.menubar", fromlist=["run"]).run(),
        "sessions": cmd_sessions,
        "menubar": lambda _args: __import__("relay.menubar", fromlist=["run"]).run(),
        "sessions": cmd_sessions,
        "new-test-chat": cmd_new_test_chat,
        "bootstrap-chat": cmd_bootstrap_chat,
        "selftest": cmd_selftest,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
