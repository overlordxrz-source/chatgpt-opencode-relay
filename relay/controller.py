"""Relay controller: ChatGPT conversation <-> OpenCode session."""

from __future__ import annotations

import hashlib
import os
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
    allowed_sessions: list[str] = field(default_factory=list)
    poll_seconds: float = 3.0
    strict_protocol: bool = True
    max_exchanges_per_hour: int = 60
    max_exchanges_per_day: int = 200
    max_reply_chars: int = 6000
    agent_timeout_ms: int = 180_000
    max_loop_repeats: int = 3
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
            conversation_index=int(chatgpt.get("conversation_index", 0)),
            poll_seconds=float(chatgpt.get("poll_seconds", 3.0)),
            bridge_url=opencode.get("bridge_url", "http://127.0.0.1:8787/mcp"),
            bridge_env=opencode.get("bridge_env", ""),
            default_session=opencode.get("default_session", ""),
            allowed_sessions=list(opencode.get("allowed_sessions", []) or []),
            strict_protocol=bool(limits.get("strict_protocol", True)),
            max_exchanges_per_hour=int(limits.get("max_exchanges_per_hour", 60)),
            max_exchanges_per_day=int(limits.get("max_exchanges_per_day", 200)),
            max_reply_chars=int(limits.get("max_reply_chars", 6000)),
            agent_timeout_ms=int(limits.get("agent_timeout_ms", 180_000)),
            max_loop_repeats=int(limits.get("max_loop_repeats", 3)),
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
    opencode_attached: bool = False
    opencode_session: str = ""
    bridge_session_id: str = ""
    last_exchange_at: float = 0.0
    last_status: str = ""
    pending_task: str = ""
    errors: list[str] = field(default_factory=list)
    exchanges_today: int = 0


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
        """Return (opencode session id, label, candidates)."""
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

        session_id, label, sessions = self._resolve_session(command.session)
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
        started = time.time()
        try:
            attached = self.opencode.attach(session_id)
            bridge_session_id = attached["bridgeSession"]["bridgeSessionId"]
            self.status.bridge_session_id = bridge_session_id
            self.status.opencode_attached = True
            self.status.opencode_session = label

            result = self.opencode.send(
                bridge_session_id,
                command.message,
                wait=True,
                timeout_ms=self.config.agent_timeout_ms,
            )
            completion = result.get("completion") or {}
            messages = completion.get("messages") or []
            reply_text = "\n\n".join(
                m.get("text", "") for m in messages if m.get("role") == "assistant" and m.get("text")
            ).strip()
            if not reply_text:
                reply_text = "(the agent did not return any text)"

            permissions = []
            try:
                permissions = (self.opencode.permissions(bridge_session_id) or {}).get("permissions") or []
            except Exception:
                permissions = []

            status = "completed" if completion.get("completed") else "timeout"
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
                protocol.format_reply(label or session_id, status, body, duration=duration, detail=detail)
            )
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
            self.store.record_exchange(
                session=command.session or "default",
                request=command.message,
                status="error",
                response=message,
                started_at=started,
            )
        finally:
            self.status.pending_task = ""

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

        # Do not act on a message that is still streaming.
        if self.chatgpt.is_generating():
            return

        messages = self.chatgpt.read_conversation(conversation)
        if not messages:
            self.status.conversation_connected = False
            return

        # Identify the conversation by content: ChatGPT renames conversations,
        # so the title alone is not a stable identifier.
        fingerprint = self.chatgpt.fingerprint(messages)
        stored = state.get("fingerprint")
        if stored and stored != fingerprint:
            # The user moved to a different conversation; that is fine, the
            # relay simply follows whatever is in front.
            self.status.conversation_connected = False
            return
        if stored != fingerprint:
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
            if not self._is_stable(message):
                # The reply is still streaming: wait for the final text so the
                # same command is never executed twice.
                continue
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
