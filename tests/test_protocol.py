"""Protocol and extraction rules (marker: stale-reply-debug-20261021).

Covers the two defects found while the five-turn conversation was running:

  * ChatGPT prefixes a command block with a sentence of prose ("Let's verify
    it from this conversation."), which the parser used to reject because it
    demanded the block start the message.
  * A long answer is collapsed behind "Show more", so the accessibility tree
    only exposes the text up to the cut - which truncated a command block
    before its closing tag.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relay.protocol import extract_commands  # noqa: E402

MARKER = "stale-reply-debug-20261021"


class ProtocolTests(unittest.TestCase):
    def test_marker(self):
        self.assertEqual(MARKER, "stale-reply-debug-20261021")

    def test_plain_block(self):
        commands = extract_commands(
            "[OPENCODE_SEND]\nsession: ses_x\nmessage: hello\n[/OPENCODE_SEND]", strict=False
        )
        self.assertEqual(1, len(commands))
        self.assertEqual("hello", commands[0].message)

    def test_block_after_prose_is_a_command(self):
        """ChatGPT often adds a sentence before the block."""
        text = (
            "That's a proper fix. Let's verify it from this conversation. "
            "[OPENCODE_SEND]\nsession: ses_x\nmessage: hello\n[/OPENCODE_SEND]\n"
            "ChatGPT can make mistakes. Check important info."
        )
        commands = extract_commands(text, strict=False)
        self.assertEqual(1, len(commands), f"{MARKER}: prose before the block broke it")
        self.assertEqual("hello", commands[0].message)

    def test_inline_block(self):
        commands = extract_commands(
            "[OPENCODE_SEND] session: ses_x message: hello [/OPENCODE_SEND]", strict=False
        )
        self.assertEqual(1, len(commands))
        self.assertEqual("hello", commands[0].message)

    def test_fenced_example_is_not_a_command(self):
        text = "Example:\n```\n[OPENCODE_SEND] session: x message: y [/OPENCODE_SEND]\n```"
        self.assertEqual([], extract_commands(text, strict=False))

    def test_truncated_block_is_not_a_command(self):
        """A collapsed message has no closing tag, so it must not fire."""
        text = (
            "That's a proper fix. [OPENCODE_SEND]\nsession: ses_x\n"
            "message: Resume our five-turn apple-pecan pie discussion. This is the FIN"
        )
        self.assertEqual([], extract_commands(text, strict=False))

    def test_documented_protocol_is_not_a_command(self):
        """A message that quotes the protocol before the block is not a command."""
        text = (
            "You can talk to the agent with an [OPENCODE_REPLY] block. "
            "[OPENCODE_SEND]\nsession: ses_x\nmessage: hello\n[/OPENCODE_SEND]"
        )
        self.assertEqual([], extract_commands(text, strict=False))

    def test_our_own_reply_is_never_a_command(self):
        text = (
            "[OPENCODE_REPLY]\nsession: x\nstatus: completed\n\ndone\n[/OPENCODE_REPLY]"
        )
        self.assertEqual([], extract_commands(text, strict=False))

    def test_two_blocks_are_ambiguous(self):
        text = (
            "[OPENCODE_SEND] session: a message: one [/OPENCODE_SEND]\n"
            "[OPENCODE_SEND] session: b message: two [/OPENCODE_SEND]"
        )
        self.assertEqual([], extract_commands(text, strict=False))

    def test_strict_mode_requires_only_the_block(self):
        text = "prose [OPENCODE_SEND] session: x message: y [/OPENCODE_SEND] more prose"
        self.assertEqual([], extract_commands(text, strict=True))
        self.assertEqual(1, len(extract_commands(text, strict=False)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
