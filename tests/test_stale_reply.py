"""Regression tests for the stale-reply bug (marker: stale-reply-debug-20261021).

The relay used to answer a *new* request with an *old* assistant message:

  * the bridge's ``sinceMessageId`` cursor silently degraded to "return the
    whole transcript" when the cursor id was not in the window, and
  * the relay took the newest assistant message in that window, and
  * accepted it even while the agent was still running.

So a request that arrived while the agent was mid-turn was answered with
whatever the last completed turn had said.

These tests pin the fix: a reply must be created after its own request, must
come from the right session, and must never be an idle marker or an earlier
answer.
"""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relay.controller import RelayConfig, RelayController  # noqa: E402
from relay.store import Store  # noqa: E402

MARKER = "stale-reply-debug-20261021"
STALE_TEXT = "STALE ANSWER FROM AN EARLIER TURN - MUST NEVER BE RETURNED"


class FakeOpenCode:
    """A session with a backlog of old messages plus one live prompt."""

    def __init__(self, backlog=None, polls_before_reply: int = 1, cursor_breaks: bool = False):
        now = time.time() * 1000
        self.transcript = list(backlog or [])
        self.polls_before_reply = polls_before_reply
        self.cursor_breaks = cursor_breaks
        self.polls = 0
        self.pending = []  # (prompt_id, reply_text, created_at)
        self.sent = []
        self._counter = 0

    # -- controller surface ------------------------------------------------
    def list_sessions(self, limit: int = 50):
        return [
            {
                "opencodeSessionId": "ses_fake",
                "title": "fake session",
                "directory": "/tmp/fake",
            }
        ]

    def attach(self, session_id: str):
        return {"bridgeSession": {"bridgeSessionId": "bs_fake"}}

    def send(self, bridge_session_id, text, wait=False, timeout_ms=120000):
        self._counter += 1
        prompt_id = f"prompt_{self._counter}"
        created = time.time() * 1000
        self.transcript.append(
            {
                "id": prompt_id,
                "role": "user",
                "text": text,
                "createdAt": created,
            }
        )
        reply = f"REPLY::{text}"
        self.pending.append((prompt_id, reply, created + 1))
        self.sent.append((prompt_id, text, created))
        return {"ok": True, "accepted": {"id": prompt_id, "time": {"created": created}}}

    def messages(self, bridge_session_id, limit=50, since_message_id=None):
        self.polls += 1
        self._materialize()
        if since_message_id:
            index = next(
                (i for i, m in enumerate(self.transcript) if m["id"] == since_message_id),
                -1,
            )
            if index < 0:
                if self.cursor_breaks:
                    return {"messages": list(self.transcript), "matchedSince": False}
                return {"messages": [], "matchedSince": False}
            return {"messages": self.transcript[index + 1 :], "matchedSince": True}
        return {"messages": list(self.transcript), "matchedSince": True}

    def status(self, bridge_session_id):
        return {"status": {"running": bool(self.pending), "outcome": None}}

    def permissions(self, bridge_session_id):
        return {"permissions": []}

    # -- internals --------------------------------------------------------
    def _materialize(self):
        if self.pending and self.polls >= self.polls_before_reply:
            prompt_id, reply, created = self.pending.pop(0)
            self.transcript.append(
                {
                    "id": f"reply_{prompt_id}",
                    "role": "assistant",
                    "text": reply,
                    "createdAt": created,
                }
            )


class FakeChatGPT:
    def __init__(self):
        self.posted = []

    def send(self, text: str) -> bool:
        self.posted.append(text)
        return True


def make_controller(fake_opencode, fake_chatgpt, tmpdir, **overrides):
    config = RelayConfig(
        conversation="Test",
        conversations=["Test"],
        bridge_url="http://127.0.0.1:1/mcp",
        bridge_env="",
        default_session="ses_fake",
        allowed_sessions=["ses_fake"],
        agent_timeout_ms=2000,
        duplicate_window_seconds=0,
        max_loop_repeats=99,
        **overrides,
    )
    store = Store(os.path.join(tmpdir, "state.db"))
    return RelayController(config, fake_chatgpt, fake_opencode, store, log=lambda *a: None)


class StaleReplyTests(unittest.TestCase):
    def test_marker(self):
        self.assertEqual(MARKER, "stale-reply-debug-20261021")

    def test_five_sequential_distinct_prompts_each_get_their_own_reply(self):
        """stale-reply-debug-20261021: five distinct prompts, five matching answers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old = time.time() * 1000 - 600_000
            backlog = [
                {"id": "old_user", "role": "user", "text": "an old question", "createdAt": old},
                {
                    "id": "old_assistant",
                    "role": "assistant",
                    "text": STALE_TEXT,
                    "createdAt": old + 1,
                },
                {"id": "old_idle", "role": "idle", "text": "", "createdAt": old + 2},
            ]
            fake = FakeOpenCode(backlog=backlog, polls_before_reply=1)
            chatgpt = FakeChatGPT()
            controller = make_controller(fake, chatgpt, tmpdir)

            prompts = [
                "turn one: what is the capital of France?",
                "turn two: what is 17 plus 25?",
                "turn three: name three primary colours",
                "turn four: which ocean is the largest?",
                "turn five: what is the boiling point of water at sea level?",
            ]
            for prompt in prompts:
                controller.handle_command(_command(prompt))
                # the worker runs in a thread; give it time to finish
                _wait_for(controller)

            exchanges = controller.store.recent_exchanges(10)
            self.assertEqual(len(exchanges), len(prompts), "one exchange per prompt")

            for exchange, prompt in zip(reversed(exchanges), prompts):
                response = str(exchange["response"])
                self.assertNotIn(
                    STALE_TEXT,
                    response,
                    f"{MARKER}: stale answer reused for {prompt!r}",
                )
                self.assertIn(
                    prompt,
                    response,
                    f"{MARKER}: response does not match its own request {prompt!r}",
                )

            for posted in chatgpt.posted:
                self.assertNotIn(STALE_TEXT, posted, f"{MARKER}: stale text posted to ChatGPT")

    def test_unmatched_cursor_never_yields_the_old_answer(self):
        """stale-reply-debug-20261021: a broken cursor must not become 'new' messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old = time.time() * 1000 - 600_000
            backlog = [
                {
                    "id": "old_assistant",
                    "role": "assistant",
                    "text": STALE_TEXT,
                    "createdAt": old,
                }
            ]
            fake = FakeOpenCode(backlog=backlog, polls_before_reply=99, cursor_breaks=True)
            chatgpt = FakeChatGPT()
            controller = make_controller(fake, chatgpt, tmpdir)

            controller.handle_command(_command("a brand new request"))
            _wait_for(controller)

            exchanges = controller.store.recent_exchanges(5)
            self.assertTrue(exchanges, f"{MARKER}: no exchange recorded")
            response = str(exchanges[0]["response"])
            self.assertNotIn(STALE_TEXT, response, f"{MARKER}: stale answer returned")
            self.assertEqual(
                exchanges[0]["status"], "timeout", f"{MARKER}: expected a timeout, not an answer"
            )

    def test_running_agent_is_not_completed_by_an_older_answer(self):
        """stale-reply-debug-20261021: no early completion while the agent works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old = time.time() * 1000 - 600_000
            backlog = [
                {
                    "id": "old_assistant",
                    "role": "assistant",
                    "text": STALE_TEXT,
                    "createdAt": old,
                }
            ]
            fake = FakeOpenCode(backlog=backlog, polls_before_reply=10_000)
            chatgpt = FakeChatGPT()
            controller = make_controller(fake, chatgpt, tmpdir)

            controller.handle_command(_command("please do the thing"))
            _wait_for(controller)

            exchanges = controller.store.recent_exchanges(5)
            self.assertTrue(exchanges)
            self.assertNotIn(STALE_TEXT, str(exchanges[0]["response"]))
            self.assertEqual(exchanges[0]["status"], "timeout")

    def test_reply_must_be_newer_than_its_request(self):
        """stale-reply-debug-20261021: ordering is by creation time, not arrival."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = FakeOpenCode(backlog=[], polls_before_reply=1)
            chatgpt = FakeChatGPT()
            controller = make_controller(fake, chatgpt, tmpdir)

            controller.handle_command(_command("hello agent"))
            _wait_for(controller)

            exchanges = controller.store.recent_exchanges(1)
            response = str(exchanges[0]["response"])
            self.assertIn("hello agent", response)
            prompt_created = fake.sent[0][2]
            reply = [m for m in fake.transcript if m["role"] == "assistant"][0]
            self.assertGreater(
                reply["createdAt"],
                prompt_created,
                f"{MARKER}: reply is not newer than its request",
            )


def _command(text: str):
    from relay.protocol import Command

    return Command(session="ses_fake", message=text, raw="")


def _wait_for(controller, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        workers = [w for w in getattr(controller, "_workers", []) if w.is_alive()]
        if not workers:
            return
        time.sleep(0.05)
    raise AssertionError(f"{MARKER}: worker threads did not finish in time")


if __name__ == "__main__":
    unittest.main(verbosity=2)
