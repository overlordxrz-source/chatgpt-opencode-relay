"""Relay controller: ChatGPT conversation <-> OpenCode session."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import urllib.request
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from . import protocol
from .chatgpt import ChatGPTAdapter, Message
from .opencode import OpenCodeAdapter
from .store import Store


@dataclass
class RelayConfig:
    conversation: str
    bridge_url: str
    bridge_env: str
    default_session: str = ""
    target_session: str = ""
    opencode_service_config: str = ""
    allowed_sessions: list[str] = field(default_factory=list)
    poll_seconds: float = 3.0
    strict_protocol: bool = True
    max_exchanges_per_hour: int = 60
    max_exchanges_per_day: int = 200
    max_reply_chars: int = 6000
    agent_timeout_ms: int = 180_000
    max_loop_repeats: int = 3
    restart_if_wedged: bool = True
    wedge_seconds: float = 300.0
    lookback_messages: int = 8
    repeat_window_seconds: float = 300.0
    duplicate_window_seconds: float = 120.0
    post_init_prompt: bool = True
    log_path: str = ""
    conversation_index: int = 0
    conversations: list[str] = field(default_factory=list)
    switch_policy: str = "idle"
    idle_seconds: float = 120.0
    rotation_seconds: float = 45.0
    route_to: str = "live"

    @classmethod
    def from_dict(cls, data: dict) -> "RelayConfig":
        chatgpt = data.get("chatgpt", {})
        opencode = data.get("opencode", {})
        limits = data.get("limits", {})
        return cls(
            conversation=chatgpt.get("conversation", ""),
            conversations=list(
                chatgpt.get("conversations")
                or ([chatgpt.get("conversation", "")] if chatgpt.get("conversation") else [])
            ),
            switch_policy=chatgpt.get("switch_policy", "idle"),
            idle_seconds=float(chatgpt.get("idle_seconds", 120)),
            rotation_seconds=float(chatgpt.get("rotation_seconds", 45)),
            route_to=opencode.get("route_to", "live"),
            conversation_index=int(chatgpt.get("conversation_index", 0)),
            poll_seconds=float(chatgpt.get("poll_seconds", 3.0)),
            bridge_url=opencode.get("bridge_url", "http://127.0.0.1:8787/mcp"),
            bridge_env=opencode.get("bridge_env", ""),
            default_session=opencode.get("default_session", ""),
            target_session=opencode.get("target_session", ""),
            opencode_service_config=opencode.get(
                "opencode_service_config", ""
            ),
            allowed_sessions=list(opencode.get("allowed_sessions", []) or []),
            strict_protocol=bool(limits.get("strict_protocol", True)),
            max_exchanges_per_hour=int(limits.get("max_exchanges_per_hour", 60)),
            max_exchanges_per_day=int(limits.get("max_exchanges_per_day", 200)),
            max_reply_chars=int(limits.get("max_reply_chars", 6000)),
            agent_timeout_ms=int(limits.get("agent_timeout_ms", 180_000)),
            max_loop_repeats=int(limits.get("max_loop_repeats", 3)),
            restart_if_wedged=bool(limits.get("restart_if_wedged", True)),
            wedge_seconds=float(limits.get("wedge_seconds", 300)),
            lookback_messages=int(limits.get("lookback_messages", 8)),
            repeat_window_seconds=float(limits.get("repeat_window_seconds", 300)),
            duplicate_window_seconds=float(limits.get("duplicate_window_seconds", 120)),
            post_init_prompt=bool(data.get("post_init_prompt", True)),
            log_path=data.get("log_path", ""),
        )


@dataclass
class RelayStatus:
    running: bool = False
    paused: bool = False
    conversation_connected: bool = False
    chatgpt_busy: bool = False
    opencode_attached: bool = False
    opencode_session: str = ""
    bridge_session_id: str = ""
    last_exchange_at: float = 0.0
    last_status: str = ""
    pending_task: str = ""
    errors: list[str] = field(default_factory=list)
    exchanges_today: int = 0


SKIP_ROLES = {
    "idle",
    "system",
    "compaction",
    "agent-switched",
    "model-switched",
    "location-switched",
    "shell",
    "skill",
    "synthetic",
}


class RelayController:
    def __init__(
        self,
        config: RelayConfig,
        chatgpt: ChatGPTAdapter,
        opencode: OpenCodeAdapter,
        store: Store,
        log=print,
        on_conversation_renamed=None,
    ):
        self.config = config
        self.chatgpt = chatgpt
        self.opencode = opencode
        self.store = store
        self.log = log
        self.status = RelayStatus()
        self.on_conversation_renamed = on_conversation_renamed
        self.debug = bool(os.environ.get("RELAY_DEBUG"))
        # Per conversation state: the last handled message pair and the
        # conversation fingerprint.
        self._state: dict[str, dict] = {}
        self.active_conversation: str = ""
        self._workers: list[threading.Thread] = []
        # Assistant message ids already claimed by a worker, so two concurrent
        # workers can never answer the same request with the same message.
        self._claimed: set[str] = set()
        self._claim_lock = threading.Lock()
        self._unreadable_since: Optional[float] = None
        self._last_restart: Optional[float] = None
        self._last_visit: dict[str, float] = {}
        self._pending_text: Optional[str] = None
        self._pending_hits = 0
        self._last_seen_fingerprints: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ helpers
    def _post(self, text: str) -> bool:
        """Post a message into the conversation, queueing it if that fails.

        The user may be busy in another conversation; the reply is kept and
        delivered as soon as the conversation can be used again.
        """
        key = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
        self.store.mark_posted(key)
        ok = self.chatgpt.send(text)
        if not ok:
            self.status.errors.append(f"queued a reply for later: {text[:60]!r}")
            self.store.add_pending_reply(text)
        return ok

    def flush_pending_replies(self) -> None:
        """Deliver replies that could not be posted earlier."""
        for reply in self.store.pending_replies(limit=5):
            if self.chatgpt.send(reply["text"]):
                self.store.delete_pending_reply(int(reply["id"]))
                self.log("relay: delivered a queued reply")
            else:
                self.store.bump_pending_reply(int(reply["id"]))
                return

    def _new_messages(self, messages: list[Message]) -> tuple[list[Message], bool]:
        """Return (new messages, aligned).

        The accessibility tree exposes no message ids, so the visible window is
        diffed against the previous read: the longest suffix of the previous
        window that matches the prefix of the new one is the overlap, and
        everything after it is new.

        Diffing (instead of matching a single marker) is what makes repeated
        but identical messages work: the *sequence* changed even though the
        text did not.
        """
        if not messages:
            return [], False
        keys = [m.key for m in messages]
        if not self._window:
            self._window = keys
            return [], False
        if keys == self._window:
            return [], False

        overlap = 0
        for size in range(min(len(self._window), len(keys)), 0, -1):
            if self._window[-size:] == keys[:size]:
                overlap = size
                break
        self._window = keys
        new = messages[overlap:]
        if not new:
            return [], True
        return new, True

    def _is_stable(self, message: Message, settle: float = 1.5) -> bool:
        """True when the assistant message stopped changing.

        ChatGPT streams its answer; acting on a half rendered message would
        execute a truncated command and then the finished one.
        """
        time.sleep(settle)
        for candidate in self.chatgpt.read_messages():
            if candidate.role == message.role and candidate.text == message.text:
                return True
        return False

    def _is_new(
        self, message: Message, predecessor: str, is_newest: bool, state: dict
    ) -> bool:
        """Is this message physically new?

        The accessibility tree has no message ids, so identity is by content.
        A message is new when it was never processed. It can also be new when
        the *same text* shows up again - but only for the newest visible
        message: older messages keep their predecessor even when the app
        virtualises the window, so a changed predecessor there would mean
        "the window moved", not "a new message".
        """
        info = self.store.processed_info(message.key)
        if info is None:
            return True
        seen_at, previous = info
        if is_newest and previous and previous != predecessor:
            if (time.time() - seen_at) > self.config.repeat_window_seconds:
                return True
        return False

    def _newest_is_new(self, messages: list[Message]) -> bool:
        """True when the newest visible message is physically new.

        The accessibility tree exposes no message ids, so a message that is
        byte-identical to an earlier one cannot be told apart by content. It
        can be told apart by its predecessor: the same text after a different
        message is a different message.
        """
        if not messages:
            return False
        newest = messages[-1]
        predecessor = messages[-2].key if len(messages) > 1 else ""
        pair = (predecessor, newest.key)
        if self._last_seen_pair is None:
            self._last_seen_pair = pair
            return False
        if pair == self._last_seen_pair:
            return False
        self._last_seen_pair = pair
        return True

    def _budget_ok(self) -> tuple[bool, str]:
        if self.store.count_since(3600) >= self.config.max_exchanges_per_hour:
            return False, "hourly exchange limit reached"
        if self.store.count_since(86400) >= self.config.max_exchanges_per_day:
            return False, "daily exchange limit reached"
        return True, ""

    def _loop_guard(self, command: protocol.Command) -> tuple[bool, str]:
        now = time.time()
        history = self._last_seen_fingerprints.setdefault(command.fingerprint, [])
        history[:] = [t for t in history if now - t < 600]
        if len(history) >= self.config.max_loop_repeats:
            return False, "identical request repeated too often (loop protection)"
        history.append(now)
        return True, ""

    def _live_session(self) -> Optional[str]:
        """The allowed session that is currently running an agent turn.

        This is the conversation the user is talking to right now, so a
        request from ChatGPT lands in the agent they are actually using. The
        running state lives in the opencode background service (the process
        that owns the TUI session), so that is where it is read from.
        """
        running = self._running_sessions()
        for session in self.allowed_session_list():
            session_id = str(session.get("opencodeSessionId", ""))
            if session_id and session_id in running:
                return session_id
        return None

    def _running_sessions(self) -> set[str]:
        """Session ids that the opencode background service reports as running."""
        service_file = Path(
            self.config.opencode_service_config
            or (Path.home() / ".local/state/opencode/service.json")
        ).expanduser()
        if not service_file.exists():
            return set()
        try:
            service = json.loads(service_file.read_text())
            url = service.get("url")
            password = service.get("password")
            if not url or not password:
                return set()
            token = base64.b64encode(f"opencode:{password}".encode()).decode()
            request = urllib.request.Request(
                f"{url}/api/session/active",
                headers={"Authorization": f"Basic {token}"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read())
        except Exception:
            return set()
        active = (payload or {}).get("data") or {}
        return {
            str(session_id)
            for session_id, state in active.items()
            if isinstance(state, dict) and state.get("type") == "running"
        }

    def _allowed(self) -> list[str]:
        allowed = list(self.config.allowed_sessions or [])
        if not allowed and self.config.default_session:
            allowed = [self.config.default_session]
        return allowed

    def _is_allowed(self, session_id: str, title: str) -> bool:
        allowed = self._allowed()
        if not allowed:
            return False
        for entry in allowed:
            entry = entry.strip()
            if not entry:
                continue
            if session_id.startswith(entry) or entry.lower() in (title or "").lower():
                return True
        return False

    def allowed_session_list(self) -> list[dict]:
        """Sessions the relay is allowed to drive."""
        sessions = self.opencode.list_sessions(limit=50)
        return [s for s in sessions if self._is_allowed(str(s.get("opencodeSessionId", "")), str(s.get("title", "")))]

    def _resolve_session(self, wanted: str) -> tuple[Optional[str], str, list[dict]]:
        """Return (opencode session id, label, candidates).

        `target_session` is the explicit override: the session the relay always
        delivers to (the live agent the user is talking to).
        """
        target = self.config.target_session
        if target:
            sessions = self.opencode.list_sessions(limit=100)
            match = next(
                (
                    item
                    for item in sessions
                    if str(item.get("opencodeSessionId", "")).startswith(target)
                    or target.lower() in str(item.get("title", "")).lower()
                ),
                None,
            )
            if match:
                return (
                    match.get("opencodeSessionId"),
                    str(match.get("title", "")),
                    sessions,
                )
        wanted = (wanted or self.config.default_session or "").strip()
        sessions = self.opencode.list_sessions(limit=50)
        if not wanted:
            if self.config.default_session:
                return None, self.config.default_session, sessions
            if len(sessions) == 1:
                only = sessions[0]
                return only.get("opencodeSessionId"), str(only.get("title", "")), sessions
            return None, "", sessions
        wanted_lower = wanted.lower()
        exact = [s for s in sessions if str(s.get("opencodeSessionId", "")).startswith(wanted)]
        if len(exact) == 1:
            return exact[0].get("opencodeSessionId"), str(exact[0].get("title", "")), sessions
        partial = [
            s
            for s in sessions
            if wanted_lower in str(s.get("title", "")).lower()
            or wanted_lower in str(s.get("opencodeSessionId", "")).lower()
        ]
        if len(partial) == 1:
            return partial[0].get("opencodeSessionId"), str(partial[0].get("title", "")), sessions
        return None, wanted, sessions

    # ------------------------------------------------------------------- command
    def handle_command(self, command: protocol.Command) -> None:
        ok, reason = self._budget_ok()
        if not ok:
            self.log(f"relay: refusing command ({reason})")
            reason = (
                f"{reason}. This is a safety budget in the relay, not an "
                f"OpenCode or ChatGPT limit. Raise it in "
                f"~/.chatgpt-opencode-relay/config.json (limits.max_exchanges_per_hour) "
                f"or wait for the window to roll over."
            )
            self.store.record_exchange(
                session=command.session or "default",
                request=command.message,
                status="refused",
                response=reason,
            )
            self._post(protocol.format_reply(command.session or "default", "refused", reason))
            return
        duplicates = self.store.recent_exchange_with_request(
            command.message, self.config.duplicate_window_seconds
        )
        if duplicates:
            self.log("relay: skipping a duplicate request (idempotency guard)")
            return
        # Reserve the request *before* the worker starts: two paths (seeding and
        # the poll loop) can pick up the same command at the same moment.
        reservation = self.store.reserve_exchange(command.session or "default", command.message)
        ok, reason = self._loop_guard(command)
        if not ok:
            self.log(f"relay: refusing command ({reason})")
            self.store.record_exchange(
                session=command.session or "default",
                request=command.message,
                status="refused",
                response=reason,
            )
            self._post(protocol.format_reply(command.session or "default", "refused", reason))
            return

        wanted = command.session
        if self.config.route_to == "live":
            live = self._live_session()
            if live:
                wanted = live
        session_id, label, sessions = self._resolve_session(wanted)
        if session_id and not self._is_allowed(session_id, label):
            allowed = ", ".join(self._allowed()) or "(none configured)"
            body = (
                f"The relay is not allowed to drive the session {label or session_id!r}.\n"
                f"Allowed sessions: {allowed}"
            )
            self.log(f"relay: refusing session {label or session_id!r} (not in the allowlist)")
            self.store.record_exchange(
                session=label or session_id,
                request=command.message,
                status="refused",
                response="session not in the allowlist",
            )
            self._post(protocol.format_reply(label or session_id, "refused", body))
            return
        if not session_id:
            names = "\n".join(
                f"- {s.get('title')} ({str(s.get('opencodeSessionId'))[:16]}…)" for s in sessions[:12]
            )
            body = (
                f"No unique OpenCode session matched {command.session!r}.\n"
                f"Available sessions:\n{names or '(none found)'}"
            )
            self._post(protocol.format_reply(command.session or "default", "error", body))
            return

        self.status.pending_task = command.message[:120]
        self.log(
            f"relay: delivering to session {label!r} ({session_id}) "
            f"(requested {command.session or 'default'!r})"
        )
        # The exchange runs in the background: an agent turn can take many
        # minutes, and the relay must keep watching the conversation meanwhile.
        worker = threading.Thread(
            target=self._run_exchange,
            args=(command, session_id, label, reservation),
            name="relay-exchange",
            daemon=True,
        )
        self._workers.append(worker)
        worker.start()

    def _run_exchange(
        self,
        command: protocol.Command,
        session_id: str,
        label: str,
        reservation: int = 0,
    ) -> None:
        """Send the request to the agent and deliver its answer to ChatGPT."""
        started = time.time()
        try:
            attached = self.opencode.attach(session_id)
            bridge_session_id = attached["bridgeSession"]["bridgeSessionId"]
            self.status.bridge_session_id = bridge_session_id
            self.status.opencode_attached = True
            self.status.opencode_session = label

            # Hand the request over without waiting: the agent may need many
            # minutes, and the answer is collected below.
            accepted = self.opencode.send(
                bridge_session_id, command.message, wait=False
            )
            entry = accepted.get("accepted") or {}
            accepted_id = entry.get("id")
            request_created_at = float((entry.get("time") or {}).get("created") or 0)
            self.log(
                f"relay: request delivered to the agent (id {accepted_id}, "
                f"created {request_created_at})"
            )

            reply_text, status = self._await_agent_reply(
                bridge_session_id, accepted_id, command.message, request_created_at
            )

            permissions = []
            try:
                permissions = (
                    self.opencode.permissions(bridge_session_id) or {}
                ).get("permissions") or []
            except Exception:
                permissions = []

            detail = None
            if permissions:
                detail = (
                    "The agent is waiting on a permission request; a human must approve it."
                )
            if len(reply_text) > self.config.max_reply_chars:
                reply_text = reply_text[: self.config.max_reply_chars] + "\n…[truncated]"
            duration = time.time() - started
            body = reply_text
            if permissions:
                body += "\n\n[permission pending] " + detail
            self._post(
                protocol.format_reply(
                    label or session_id, status, body, duration=duration, detail=detail
                )
            )
            if reservation:
                self.store.finish_exchange(reservation, status, reply_text)
            else:
                self.store.record_exchange(
                    session=label or session_id,
                    request=command.message,
                    status=status,
                    response=reply_text,
                    started_at=started,
                )
            self.status.last_status = status
            self.status.last_exchange_at = time.time()
        except Exception as error:  # noqa: BLE001 - report and keep running
            message = f"{type(error).__name__}: {error}"
            self.log(f"relay: exchange failed: {message}")
            self.status.errors.append(message)
            self._post(protocol.format_reply(command.session or "default", "error", message))
            if reservation:
                self.store.finish_exchange(reservation, "error", message)
            else:
                self.store.record_exchange(
                    session=command.session or "default",
                    request=command.message,
                    status="error",
                    response=message,
                    started_at=started,
                )
        finally:
            self.status.pending_task = ""

    def _await_agent_reply(
        self,
        bridge_session_id: str,
        accepted_id: Optional[str],
        request: str,
        request_created_at: float = 0.0,
    ) -> tuple[str, str]:
        """Wait for the answer to *this* request.

        Correlation rules (stale-reply-debug-20261021):

        1. Only messages created **after** this request count. The request
           timestamp comes from the inbox entry the bridge accepted, so an
           answer from an earlier turn can never be mistaken for this one.
        2. The bridge's ``sinceMessageId`` cursor is honoured, and if the
           cursor cannot be found the window is ignored rather than treated
           as "everything is new".
        3. The first assistant message after the request is the answer - not
           the newest one, which may belong to a later, concurrent request.
        4. The agent must have finished its turn (not running) and the text
           must be unchanged over two consecutive polls.
        """
        budget = self.config.agent_timeout_ms / 1000
        deadline = time.time() + budget
        if not request_created_at:
            request_created_at = time.time() * 1000
        candidate: Optional[str] = None
        candidate_id: Optional[str] = None
        stable_hits = 0
        while time.time() < deadline:
            try:
                # Read the whole window and filter by time. The cursor is only
                # a hint: a server that does not own the session (the bridge
                # spawns its own, the agent runs in the background service)
                # never resolves the cursor, yet its window does contain the
                # answer. The timestamp is the authoritative guard.
                page = self.opencode.messages(bridge_session_id, limit=200)
                messages = page.get("messages") or []
                fresh = [
                    m
                    for m in messages
                    if float(m.get("createdAt") or 0) >= request_created_at
                ]
                with self._claim_lock:
                    claimed = set(self._claimed)
                # The agent's answer can arrive as *several* assistant messages
                # (one turn is split into segments), so the reply is the whole
                # trailing run of assistant messages, not just the first.
                run: list[dict] = []
                for message in reversed(fresh):
                    role = message.get("role")
                    if role in SKIP_ROLES and not message.get("text"):
                        # A turn ends with an idle marker, and the app also emits
                        # system/compaction events: skip them, do not stop here.
                        continue
                    if role != "assistant":
                        break  # a new user message starts a new turn
                    if not message.get("text"):
                        continue  # an empty assistant segment
                    if str(message.get("id")) in claimed:
                        break
                    run.append(message)
                run.reverse()
                if run:
                    text = "\n\n".join(m.get("text", "") for m in run).strip()
                    ids = [str(m.get("id")) for m in run]
                    signature = "|".join(ids)
                    if text == candidate and signature == candidate_id:
                        stable_hits += 1
                    else:
                        candidate = text
                        candidate_id = signature
                        stable_hits = 1
                    if stable_hits >= 3:
                        with self._claim_lock:
                            self._claimed.update(ids)
                        return candidate, "completed"
            except Exception as error:  # noqa: BLE001
                self.log(f"relay: error while waiting for the agent: {error}")
            time.sleep(0.5)
        if candidate:
            return candidate, "completed"
        return (f"(the agent did not answer within {int(budget)}s)", "timeout")

    # ---------------------------------------------------------------------- loop
    def tick(self) -> None:
        """One relay cycle.

        The relay works with the conversation that is open: a request sent
        from one whitelisted conversation is answered in that same
        conversation. Only when another app is in front does the relay open a
        whitelisted conversation itself.
        """
        self.flush_pending_replies()
        whitelisted = [c for c in (self.config.conversations or [self.config.conversation]) if c]
        if not whitelisted:
            return
        try:
            open_title = self.chatgpt.current_title()
        except Exception:
            open_title = ""
        if open_title in whitelisted:
            targets = [open_title]
        else:
            # The frontmost conversation is not whitelisted: the relay stays
            # out of it and does not move the app anywhere else.
            return
        for conversation in targets:
            try:
                self._tick_conversation(conversation)
            except Exception as error:  # noqa: BLE001 - one conversation must not kill the loop
                self.log(f"relay: error in {conversation!r}: {type(error).__name__}: {error}")

    def _tick_conversation(self, conversation: str) -> None:
        state = self._state.setdefault(conversation, {"fingerprint": None, "pair": None})
        self.chatgpt.conversation = conversation
        self.active_conversation = conversation

        # One accessibility walk per cycle, reused by every check below.
        tree = self.chatgpt.refresh_tree()
        messages = self.chatgpt.read_conversation(conversation, tree=tree)
        if not messages:
            if self.status.conversation_connected:
                self.log(f"relay: {conversation!r} is not readable right now (idling)")
            self.status.conversation_connected = False
            self.watchdog(self.config.wedge_seconds)
            return
        if not self.status.conversation_connected:
            self.log(f"relay: {conversation!r} is readable again")
        self.status.conversation_connected = True
        # A streaming answer is not actionable yet: wait for the next cycle.
        # (Reported as connected-but-busy, not as disconnected.)
        self.status.chatgpt_busy = self.chatgpt.is_generating(tree)
        if self.status.chatgpt_busy:
            return

        # Identify the conversation by content: ChatGPT renames conversations,
        # so the title alone is not a stable identifier.
        fingerprint = self.chatgpt.fingerprint(messages)
        # The fingerprint is over the last messages, so it changes whenever the
        # conversation grows. It is only used to recognise a renamed
        # conversation - it must never stop the relay from working.
        state["fingerprint"] = fingerprint
        self.status.conversation_connected = True

        title = self.chatgpt.current_title()
        if title and title != conversation and title in self.config.conversations:
            # The app renamed one of the watched conversations; follow it.
            self.log(f"relay: {conversation!r} was renamed to {title!r}; following it")
            self.config.conversations = [
                title if item == conversation else item for item in self.config.conversations
            ]
            if self.on_conversation_renamed:
                self.on_conversation_renamed(title)

        if self.debug:
            self.log(
                f"relay[debug]: {conversation!r} visible={len(messages)} "
                f"generating={self.chatgpt.is_generating()}"
            )

        # A collapsed answer hides the end of a command block (the closing tag).
        # Expand it and read again before deciding there is no command.
        if messages and messages[-1].role == "assistant":
            newest = messages[-1].text
            if "[OPENCODE_SEND]" in newest and "[/OPENCODE_SEND]" not in newest:
                if self.chatgpt.expand_latest(tree):
                    tree = self.chatgpt.refresh_tree()
                    messages = self.chatgpt.read_conversation(conversation, tree=tree)
                    if not messages:
                        return

        # Only the most recent messages are eligible: older ones that scroll
        # into view later must never be executed.
        keys = [message.key for message in messages]
        for index, message in enumerate(messages[-self.config.lookback_messages :]):
            position = len(messages) - self.config.lookback_messages + index
            predecessor = keys[position - 1] if position > 0 else ""
            is_newest = position == len(messages) - 1
            if not self._is_new(message, predecessor, is_newest, state):
                continue
            if message.role != "assistant":
                self.store.mark_processed(message.key, message.role, prev=predecessor)
                continue
            if self.store.is_posted(message.key):
                continue
            commands = protocol.extract_commands(message.text, self.config.strict_protocol)
            if not commands:
                self.store.mark_processed(message.key, message.role, prev=predecessor)
                continue
            # A command block that is closed is a finished message: the parser
            # needs the closing tag, so a half streamed reply never matches.
            self.store.mark_processed(message.key, message.role, prev=predecessor)
            command = commands[0]
            self.log(f"relay: command for session {command.session!r}: {command.message[:80]!r}")
            self.handle_command(command)

    def wait_until_stable(self, message: Message) -> bool:
        """Wait until an assistant message stops changing (generation finished)."""
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.chatgpt.is_generating():
                time.sleep(1.0)
                continue
            current = self.chatgpt.read_messages()
            for candidate in reversed(current):
                if candidate.role == "assistant" and candidate.key == message.key:
                    if candidate.text == self._pending_text:
                        self._pending_hits += 1
                    else:
                        self._pending_text = candidate.text
                        self._pending_hits = 1
                    if self._pending_hits >= 2:
                        return True
                    break
            time.sleep(1.0)
        return False
