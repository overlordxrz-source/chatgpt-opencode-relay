"""Relay command protocol.

ChatGPT is asked to answer with exactly one command block:

    [OPENCODE_SEND]
    session: Project Ferret
    message: Investigate the latest kernel build.
    [/OPENCODE_SEND]

The relay answers with:

    [OPENCODE_REPLY]
    session: Project Ferret
    status: completed
    duration: 42s

    <the agent's reply>
    [/OPENCODE_REPLY]

Safety rules baked in here:
  * blocks inside fenced code blocks are ignored (so ChatGPT can *show* the
    protocol without the relay acting on it),
  * in "strict" mode (default) a command message must contain nothing but the
    block, so prose that merely mentions the protocol is never executed,
  * only assistant messages are scanned - a human quoting the protocol in chat
    cannot trigger the relay by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

SEND_OPEN = "[OPENCODE_SEND]"
SEND_CLOSE = "[/OPENCODE_SEND]"
REPLY_OPEN = "[OPENCODE_REPLY]"
REPLY_CLOSE = "[/OPENCODE_REPLY]"

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_BLOCK = re.compile(
    re.escape(SEND_OPEN) + r"\s*(.*?)\s*" + re.escape(SEND_CLOSE), re.DOTALL
)
_INLINE = re.compile(
    re.escape(SEND_OPEN) + r"\s*(.*?)\s*" + re.escape(SEND_CLOSE), re.DOTALL
)


@dataclass
class Command:
    session: str
    message: str
    raw: str

    @property
    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(f"{self.session}\x00{self.message}".encode()).hexdigest()[:16]


def _strip_fences(text: str) -> str:
    """Blank out fenced code blocks, keeping character offsets roughly stable."""
    return _FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _parse_fields(body: str) -> dict[str, str]:
    """Parse the block body.

    ChatGPT writes the block either as lines ("session: x\nmessage: y") or on a
    single line ("session: x message: y"), so both are supported.
    """
    body = body.strip()
    if "\n" not in body:
        fields: dict[str, str] = {}
        session = re.search(r"\bsession\s*:\s*(.*?)(?=\s+\bmessage\s*:|$)", body, re.IGNORECASE)
        if session:
            fields["session"] = session.group(1).strip()
        message = re.search(r"\bmessage\s*:\s*(.*)$", body, re.IGNORECASE | re.DOTALL)
        if message:
            fields["message"] = message.group(1).strip()
        return fields

    fields = {}
    current: Optional[str] = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current:
                fields[current] += "\n"
            continue
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key in ("session", "message", "agent", "model"):
                current = key
                fields[current] = value.strip()
            else:
                current = None
            continue
        if current:
            fields[current] = (fields[current] + "\n" + line.strip()).strip()
    return fields


def extract_commands(text: str, strict: bool = True) -> list[Command]:
    """Return the relay commands contained in a ChatGPT message."""
    if SEND_OPEN not in text:
        return []
    scrubbed = _strip_fences(text)
    matches = list(_BLOCK.finditer(scrubbed))
    if len(matches) != 1:
        # No block, or several: never guess which one was meant.
        return []
    match = matches[0]
    before = scrubbed[: match.start()]
    after = scrubbed[match.end() :]
    if strict:
        # Nothing but the block itself may be present.
        if before.strip() or after.strip():
            return []
    else:
        # The block must start the message (so "here is an example: <block>"
        # is not a command), and the trailing text must not contain further
        # protocol markers (so our own replies are never re-executed).
        if before.strip():
            return []
        if "OPENCODE_" in after:
            return []
    body = match.group(1)
    fields = _parse_fields(body)
    message = fields.get("message", "").strip()
    session = fields.get("session", "").strip()
    if not message:
        return []
    return [Command(session=session, message=message, raw=match.group(0))]


def format_reply(
    session_label: str,
    status: str,
    body: str,
    duration: Optional[float] = None,
    detail: Optional[str] = None,
) -> str:
    lines = [REPLY_OPEN, f"session: {session_label}", f"status: {status}"]
    if duration is not None:
        lines.append(f"duration: {duration:.0f}s")
    if detail:
        lines.append(f"detail: {detail}")
    lines.append("")
    lines.append(body.rstrip())
    lines.append(REPLY_CLOSE)
    return "\n".join(lines)


def build_init_prompt(sessions: list[dict] | None = None) -> str:
    """The initialization message posted into the watched conversation."""
    listing = ""
    if sessions:
        lines = []
        for session in sessions[:12]:
            title = session.get("title") or "(untitled)"
            identifier = str(session.get("opencodeSessionId", ""))[:16]
            lines.append(f"- {title}  (id: {identifier})")
        listing = "\n\nSessions you can address:\n" + "\n".join(lines)
    return INITIAL_PROMPT.replace("[/OPENCODE_RELAY]", listing + "\n[/OPENCODE_RELAY]")


INITIAL_PROMPT = """\
[OPENCODE_RELAY]
This conversation is connected to a local OpenCode coding agent on the user's Mac.

To send work to that agent, reply with EXACTLY this block and nothing else
(no explanation before or after it):

[OPENCODE_SEND]
session: <session name or id>
message: <the instruction for the agent>
[/OPENCODE_SEND]

Rules:
- Output only the block. No preamble, no explanation, no code fences.
- `session` is the name of one of the sessions you were told about, or the id.
- The agent's answer comes back as an [OPENCODE_REPLY] block; read it and
  summarise it for the user.
- If the agent reports a permission request, ask the user before approving it.
[/OPENCODE_RELAY]
"""
