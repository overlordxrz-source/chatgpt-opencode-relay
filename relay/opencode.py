"""OpenCode adapter.

Reuses the existing opencode-chatgpt-bridge MCP server (loopback, bearer
token) instead of re-implementing the OpenCode communication layer. Every
operation here is a call to a tool that bridge already provides: session
discovery, attaching to an existing session, sending a message, incremental
message fetch, status and permission inspection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional

import httpx2 as httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


@dataclass
class BridgeToolError(RuntimeError):
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class _BridgeWorker:
    """Owns one MCP session to the bridge, driven from a background thread."""

    def __init__(self, url: str, token: str, log=print):
        self.url = url
        self.token = token
        self.log = log
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stack: Optional[contextlib.AsyncExitStack] = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- plumbing
    def start(self) -> None:
        if self._thread is not None:
            return
        # Create the loop here so callers never race the worker thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="mcp-bridge", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> ClientSession:
        await self._disconnect()
        stack = contextlib.AsyncExitStack()
        await stack.__aenter__()
        headers = {"Authorization": f"Bearer {self.token}"}
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(600.0))
        )
        streams = await stack.enter_async_context(
            streamable_http_client(self.url, http_client=http_client)
        )
        read_stream, write_stream = streams[0], streams[1]
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self._stack = stack
        self._session = session
        return session

    async def _disconnect(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
        self._stack = None
        self._session = None

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            await self._connect()
        assert self._session is not None
        return self._session

    async def _call_async(self, tool: str, arguments: dict[str, Any]) -> Any:
        session = await self._ensure_session()
        result = await session.call_tool(tool, arguments, read_timeout_seconds=600)
        if getattr(result, "isError", False):
            detail = ""
            for item in getattr(result, "content", []) or []:
                detail = getattr(item, "text", "") or detail
            raise BridgeToolError(f"{tool} failed: {detail[:400]}")
        payload = getattr(result, "structuredContent", None)
        if payload is None:
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                if text:
                    try:
                        payload = json.loads(text)
                        break
                    except json.JSONDecodeError:
                        continue
        if payload is None:
            payload = {}
        if isinstance(payload, dict) and payload.get("ok") is False:
            raise BridgeToolError(str(payload.get("error", "bridge tool failed")))
        return payload

    # -------------------------------------------------------------------- public
    def call(self, tool: str, arguments: Optional[dict[str, Any]] = None, timeout: float = 300.0) -> Any:
        self.start()
        assert self._loop is not None
        with self._lock:
            future = asyncio.run_coroutine_threadsafe(
                self._call_async(tool, arguments or {}), self._loop
            )
            try:
                return future.result(timeout)
            except Exception:
                # Drop the broken session so the next call reconnects.
                self._session = None
                raise

    def close(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread = None


class OpenCodeAdapter:
    """High level operations the relay needs from OpenCode."""

    def __init__(self, url: str, token: str, log=print):
        self._worker = _BridgeWorker(url, token, log=log)

    def call(self, tool: str, arguments: Optional[dict[str, Any]] = None, timeout: float = 300.0) -> Any:
        return self._worker.call(tool, arguments, timeout=timeout)

    # ------------------------------------------------------------------ helpers
    def health(self) -> dict:
        return self.call("bridge_health", {}, timeout=30)

    def list_sessions(self, limit: int = 20) -> list[dict]:
        payload = self.call("opencode_list_sessions", {"limit": limit}, timeout=60)
        return list(payload.get("opencodeSessions") or [])

    def create_session(self, repo_path: str, title: str, model: str | None = None) -> dict:
        args: dict[str, Any] = {"repoPath": repo_path, "title": title}
        if model:
            args["model"] = model
        return self.call("opencode_create_session", args, timeout=120)

    def attach(self, opencode_session_id: str) -> dict:
        return self.call("opencode_attach_session", {"opencodeSessionId": opencode_session_id}, timeout=120)

    def send(self, bridge_session_id: str, text: str, wait: bool = False, timeout_ms: int = 240_000) -> dict:
        # The bridge caps opencode_send_message at 300s; stay inside that.
        timeout_ms = min(max(int(timeout_ms), 1000), 300_000)
        return self.call(
            "opencode_send_message",
            {
                "bridgeSessionId": bridge_session_id,
                "text": text,
                "wait": wait,
                "timeoutMs": timeout_ms,
            },
            timeout=(timeout_ms / 1000) + 120,
        )

    def messages(self, bridge_session_id: str, limit: int = 50, since_message_id: str | None = None) -> dict:
        args: dict[str, Any] = {"bridgeSessionId": bridge_session_id, "limit": limit}
        if since_message_id:
            args["sinceMessageId"] = since_message_id
        return self.call("opencode_get_messages", args, timeout=120)

    def status(self, bridge_session_id: str) -> dict:
        return self.call("opencode_get_session_status", {"bridgeSessionId": bridge_session_id}, timeout=60)

    def permissions(self, bridge_session_id: str) -> dict:
        return self.call("opencode_list_permissions", {"bridgeSessionId": bridge_session_id}, timeout=60)
