"""ChatGPT desktop adapter built purely on macOS accessibility APIs.

Responsibilities:
  * open/verify the one conversation the relay is allowed to watch
  * read that conversation's messages with their roles
  * detect that the assistant finished generating
  * put text into the composer and submit it
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from .ax import ChatGPTApp, Node

USER_MARKER = "You said:"
ASSISTANT_MARKER = "ChatGPT said:"
IGNORED_TEXT = {
    "Loading older messages…",
    "Loading older messages...",
    "Response complete",
}


@dataclass
class Message:
    role: str  # "user" | "assistant"
    text: str

    @property
    def key(self) -> str:
        import hashlib

        digest = hashlib.sha256(f"{self.role}\x00{self.text}".encode("utf-8", "replace")).hexdigest()
        return digest[:16]


class ChatGPTAdapter:
    def __init__(self, app: ChatGPTApp, conversation: str, log=print, index: int = 0):
        self.app = app
        self.conversation = conversation
        self.index = index
        self.log = log

    # ------------------------------------------------------------------ helpers
    def _window_title(self) -> str:
        node = self.app.window()
        return node.title if node else ""

    def current_title(self) -> str:
        return self._window_title()

    @staticmethod
    def fingerprint(messages: list[Message]) -> str:
        """Content based identity of a conversation (titles get renamed)."""
        import hashlib

        tail = "|".join(message.key for message in messages[-3:])
        return hashlib.sha256(tail.encode()).hexdigest()[:16]

    def _conversation_node(self, tree: Node) -> Optional[Node]:
        return tree.find(description=None) if False else _find_by_description(tree, "Conversation")

    def open_conversation(self) -> bool:
        """Make sure the watched conversation is the one on screen."""
        tree = self.app.tree()
        if self._window_title() == self.conversation:
            return True
        matches = [b for b in tree.find_all(role="AXButton") if b.text.strip() == self.conversation]
        button = matches[self.index] if len(matches) > self.index else None
        if button is None and matches:
            button = matches[0]
        if button is None:
            wanted = self.conversation.lower()
            # Case-insensitive, then prefix, then substring.
            for candidate in tree.find_all(role="AXButton"):
                text = candidate.text.strip().lower()
                if text and text == wanted:
                    button = candidate
                    break
            if button is None:
                for candidate in tree.find_all(role="AXButton"):
                    text = candidate.text.strip().lower()
                    if text and (text.startswith(wanted) or wanted.startswith(text)) and len(text) > 6:
                        button = candidate
                        break
        if button is None:
            for candidate in tree.find_all(role="AXButton"):
                if self.conversation.lower() in candidate.text.lower():
                    button = candidate
                    break
        if button is None:
            self.log(f"chatgpt: conversation {self.conversation!r} not found in sidebar")
            return False
        self.app.press(button)
        for _ in range(20):
            time.sleep(0.25)
            if self._window_title() == self.conversation:
                return True
        self.log(f"chatgpt: window title is {self._window_title()!r} after opening {self.conversation!r}")
        return False

    def rename_conversation(self, new_title: str) -> bool:
        """Rename the open conversation through the app's own menu."""
        tree = self.app.tree()
        popup = next((p for p in tree.find_all(role="AXPopUpButton") if p.title == "Chat actions"), None)
        if popup is None:
            self.log("chatgpt: no 'Chat actions' button in the header")
            return False
        if not self.app.press(popup):
            return False
        time.sleep(0.8)
        tree = self.app.tree()
        rename = next((i for i in tree.find_all(role="AXMenuItem") if i.text.strip() == "Rename"), None)
        if rename is None:
            self.log("chatgpt: no 'Rename' menu item")
            return False
        self.app.press(rename)
        time.sleep(1.0)
        tree = self.app.tree()
        field = tree.find(role="AXTextField") or tree.find(role="AXTextArea")
        if field is None:
            self.log("chatgpt: rename field not found")
            return False
        self.app.set_value(field, new_title)
        time.sleep(0.3)
        self.app.press_return()
        time.sleep(1.2)
        return self._window_title() == new_title

    # ----------------------------------------------------------------- messages
    def read_messages(self) -> list[Message]:
        tree = self.app.tree()
        container = _find_by_description(tree, "Conversation")
        if container is None:
            return []
        return _extract_messages(container)

    def latest_assistant_message(self) -> Optional[Message]:
        for message in reversed(self.read_messages()):
            if message.role == "assistant":
                return message
        return None

    def is_generating(self) -> bool:
        """True while the assistant is still streaming a response."""
        tree = self.app.tree()
        for node in tree.find_all(role="AXButton"):
            title = node.title.lower()
            if title in ("stop", "stop generating", "stop response"):
                return True
        return False

    # --------------------------------------------------------------------- send
    def focus_composer(self) -> Optional[Node]:
        """Put keyboard focus in the message composer.

        The composer moves depending on the conversation (bottom of the thread
        vs. centred in a new chat), so it is located in the accessibility tree
        rather than by fixed coordinates.
        """
        tree = self.app.tree()
        composer = tree.find(role="AXTextArea")
        if composer is None:
            self.log("chatgpt: no AXTextArea (composer) in the tree")
            return None

        self.app.press(composer)
        time.sleep(0.2)
        focused = self.app.focused_element()
        if focused is not None and focused.role == "AXTextArea":
            return focused

        if self.app.set_focus(composer):
            time.sleep(0.3)
            focused = self.app.focused_element()
            if focused is not None and focused.role == "AXTextArea":
                return focused

        if composer.position and composer.size:
            x = composer.position[0] + composer.size[0] / 2
            y = composer.position[1] + composer.size[1] / 2
            self.app.click(x, y)
            time.sleep(0.4)
            focused = self.app.focused_element()
            if focused is not None and focused.role == "AXTextArea":
                return focused
        return None

    def send(self, text: str) -> bool:
        """Type a message into the composer and submit it."""
        if not self.open_conversation():
            return False
        self.app.activate()
        time.sleep(0.4)

        composer = self.focus_composer()
        if composer is None:
            focused = self.app.focused_element()
            self.log(
                f"chatgpt: composer not focused (focused={focused.role if focused else None})"
            )
            return False

        if not self.app.set_value(composer, text):
            self.log("chatgpt: AXSetValue failed, falling back to keystrokes")
            self.app.type_text(text)
            time.sleep(0.3)

        # The accessibility value updates a moment after the write, so poll
        # briefly instead of trusting a single read.
        accepted = False
        for _ in range(15):
            time.sleep(0.2)
            current = self.app.focused_element()
            if current is None or current.role != "AXTextArea":
                continue
            value = current.value or ""
            if value and text[:24] in value:
                accepted = True
                break
        if not accepted:
            self.log("chatgpt: composer did not accept the text")
            return False

        tree = self.app.tree()
        send_button = tree.find(role="AXButton", exact_text="Send")
        if send_button is not None:
            self.app.press(send_button)
        else:
            self.app.press_return()
        time.sleep(0.8)
        return True


def _find_by_description(node: Node, description: str) -> Optional[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.description == description:
            return current
        stack.extend(reversed(current.children))
    return None


def _paragraph_text(group: Node) -> str:
    """Join the inline runs of one rendered paragraph back into readable text.

    The app exposes every inline run (often every word) as its own static text
    element, and inline code spans as nested groups. Empty runs are line
    breaks. Control subtrees (buttons) carry no message text.
    """
    pieces: list[str] = []

    def collect(node: Node) -> None:
        for child in node.children:
            if child.role in (
                "AXButton", "AXCheckBox", "AXPopUpButton", "AXImage",
                "AXMenu", "AXMenuItem", "AXTextArea", "AXHeading",
            ):
                continue
            if child.role == "AXStaticText":
                text = child.text.strip()
                if text in IGNORED_TEXT or text in (USER_MARKER, ASSISTANT_MARKER):
                    continue
                if re.fullmatch(r"Worked for \d+.*", text):
                    continue
                pieces.append(text)
                continue
            collect(child)

    collect(group)
    out = ""
    for piece in pieces:
        if not piece:
            out += "\n"
            continue
        if out and not out.endswith((" ", "\n")):
            out += " "
        out += piece
    return out.strip()


def _extract_messages(container: Node) -> list[Message]:
    """Walk the conversation subtree and rebuild ordered (role, text) messages."""
    messages: list[Message] = []
    role: Optional[str] = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(piece for piece in buffer if piece).strip()
        if role and text:
            messages.append(Message(role=role, text=text))
        buffer = []

    def walk(node: Node) -> None:
        nonlocal role
        if node.role == "AXHeading":
            # The heading's own AXValue is the level ("4"); the label lives in a
            # child static text ("You said:" / "ChatGPT said:").
            label_parts = [child.text for child in node.children if child.role == "AXStaticText"]
            label = " ".join(" ".join(label_parts).split()) or " ".join(node.text.split())
            if USER_MARKER in label:
                flush()
                role = "user"
                return
            if ASSISTANT_MARKER in label:
                flush()
                role = "assistant"
                return
            return  # other headings ("Latest response", level markers) are not roles
        if node.role in ("AXButton", "AXCheckBox", "AXPopUpButton", "AXImage", "AXMenu", "AXMenuItem", "AXTextArea"):
            return  # controls, the composer and their labels are not message content
        if node.role == "AXStaticText":
            text = node.text.strip()
            if not text or text in IGNORED_TEXT or text in (USER_MARKER, ASSISTANT_MARKER):
                return
            if re.fullmatch(r"Worked for \d+.*", text):
                return
            if role:
                buffer.append(text)
            return
        if node.role == "AXGroup":
            # A group holding static text directly is one rendered paragraph.
            if any(child.role == "AXStaticText" for child in node.children):
                text = _paragraph_text(node)
                if text and role:
                    buffer.append(text)
                return
        for child in node.children:
            walk(child)

    for child in container.children:
        walk(child)
    flush()
    return messages
