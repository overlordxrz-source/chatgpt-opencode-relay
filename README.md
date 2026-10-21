# ChatGPT ↔ OpenCode Desktop Relay

A small macOS relay that lets **regular ChatGPT conversations** (on your
desktop *or* your iPhone) talk to your **local OpenCode agent**.

```
iPhone ChatGPT  ──sync──►  ChatGPT Desktop (the conversation you selected)
                                   │  accessibility API reads / writes
                                   ▼
                             relay (this project)
                                   │  MCP over loopback
                                   ▼
                    opencode-chatgpt-bridge (existing, v2)
                                   │
                                   ▼
                        your real OpenCode session
```

The relay never talks to OpenAI's APIs and never touches your ChatGPT
credentials. It only reads and writes the **one conversation** you select, in
the ChatGPT desktop app, through macOS accessibility.

---

## 1. Requirements

* macOS with the **ChatGPT desktop app** (the one that ships `com.openai.codex`)
* **Python 3.11+** (3.14 tested) and `pip`
* The **opencode-chatgpt-bridge** (v2 port) running on loopback, with its
  `OPENCODE_BRIDGE_TOKEN` in its `.env`
* `opencode` 2.x installed and signed in
* The terminal/app that runs the relay needs **Accessibility** permission for
  the ChatGPT app (System Settings → Privacy & Security → Accessibility)

## 2. Install

```bash
cd ~/tools/chatgpt-opencode-relay
python3 -m venv .venv
.venv/bin/pip install mcp pyobjc-framework-Quartz pyobjc-framework-Cocoa pyobjc-framework-ApplicationServices
```

## 3. Configure

```bash
.venv/bin/python relayctl.py init            # writes ~/.chatgpt-opencode-relay/config.json
```

`config.json` (see `relay/config.py` for the defaults):

| Key | Meaning |
|---|---|
| `chatgpt.conversation` | Title of the conversation to watch |
| `chatgpt.conversation_index` | Which sidebar entry with that title (0 = first) |
| `chatgpt.poll_seconds` | How often the conversation is read (default 3 s) |
| `opencode.bridge_url` | `http://127.0.0.1:8787/mcp` |
| `opencode.bridge_env` | Path to the bridge `.env` (the token is read from here, never copied) |
| `opencode.default_session` | OpenCode session used when the command names none |
| `opencode.allowed_sessions` | **Allow list**: only these OpenCode sessions may be driven |
| `limits.*` | Budgets: exchanges per hour/day, reply size, agent timeout, look-back |
| `paths.*` | State database, status file, log file |

## 4. Prepare the conversation

```bash
.venv/bin/python relayctl.py bootstrap-chat --title "OpenCode Relay"
```

This creates a **new** ChatGPT conversation, renames it, and posts the
initialisation message that teaches ChatGPT the protocol. The title is saved
to the config.

## 5. Run

```bash
.venv/bin/python relayctl.py start     # background
.venv/bin/python relayctl.py status    # health
.venv/bin/python relayctl.py logs -n 50
.venv/bin/python relayctl.py stop
.venv/bin/python relayctl.py menubar   # optional menu bar UI
```

`relayctl pause` / `resume` stop and restart the loop without killing it.

## 6. Protocol

ChatGPT answers with exactly one block:

```
[OPENCODE_SEND]
session: <session name or id>
message: <the instruction for the agent>
[/OPENCODE_SEND]
```

and the relay answers with:

```
[OPENCODE_REPLY]
session: <session title>
status: completed
duration: 42s

<the agent's answer>
[/OPENCODE_REPLY]
```

Rules that keep this safe (see `relay/protocol.py`):

* only **assistant** messages are inspected, so quoting the protocol never
  triggers anything;
* blocks inside fenced code blocks are ignored;
* exactly one block per message, and it must **start** the message;
* the session must be on the **allow list**;
* duplicate requests are dropped (idempotency window);
* budgets per hour/day and a loop guard stop runaway exchanges;
* permission requests are reported, never approved.

## 7. How it reads ChatGPT

Everything goes through the public macOS accessibility API (`AXUIElement`):

* message roles come from the app's own `AXHeading` elements
  ("You said:" / "ChatGPT said:") - not from text matching;
* the composer is the `AXTextArea` named "Ask ChatGPT";
* the send button is the `AXButton` named "Send";
* conversations are identified by **content**, because the app renames chats.

## 8. Safety

* The relay reads/writes exactly one conversation.
* OpenCode sessions are restricted by `opencode.allowed_sessions`.
* The relay never approves a permission request.
* Logs contain message *shapes* and ids, not the full text of your
  conversations.
* The bridge token stays in the bridge's `.env` (mode 600); the relay reads it
  at runtime and never copies it into its own config.
* Nothing is installed as a login item; start it yourself.

## 9. Limitations

* **The Mac must be unlocked and awake.** Accessibility is unavailable while
  the screen is locked, so the relay cannot read or post while you are away -
  it will resume when the screen is unlocked again.
* The relay reads the **desktop** app. Phone messages only reach it after
  ChatGPT syncs them to the desktop app (seconds when the app is running and
  online).
* Posting a reply activates the ChatGPT app, which briefly takes focus.
* The app virtualises long conversations; the relay only considers the last
  `limits.lookback_messages` messages.
* If ChatGPT does not answer with a command block, nothing happens - that is
  the point of the strict protocol.
* OpenAI's Terms of Use forbid using the Services to "automatically or
  programmatically extract data or Output" and to interfere with the service.
  This relay is a personal, single-conversation, human-supervised tool on your
  own machine, but you should decide whether it fits your use of the service.

## 10. Tests

```bash
.venv/bin/python relayctl.py selftest --send "hello" --session "<name>" --prompt "say hi"
```

`scripts/` also contains the checkpoint scripts used during development.
