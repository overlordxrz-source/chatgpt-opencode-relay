"""Persistent relay state (SQLite).

Tracks processed messages, relay-posted messages, exchange history, queued
replies and the counters used for loop and budget protection.

The exchange runs in a background thread, so the connection is shared between
threads (check_same_thread=False) and every access is serialised with a lock.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class Store:
    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS processed (
                    key TEXT PRIMARY KEY,
                    role TEXT,
                    kind TEXT,
                    seen_at REAL,
                    prev TEXT
                );
                CREATE TABLE IF NOT EXISTS posts (
                    key TEXT PRIMARY KEY,
                    posted_at REAL
                );
                CREATE TABLE IF NOT EXISTS exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL,
                    finished_at REAL,
                    session TEXT,
                    request TEXT,
                    status TEXT,
                    response TEXT
                );
                CREATE TABLE IF NOT EXISTS pending_replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL,
                    text TEXT,
                    attempts INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(processed)")}
            if "prev" not in columns:
                self._db.execute("ALTER TABLE processed ADD COLUMN prev TEXT")
            self._db.commit()

    # ------------------------------------------------------------------ kv store
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._db.commit()

    # ---------------------------------------------------------------- messages
    def is_processed(self, key: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM processed WHERE key=?", (key,)).fetchone()
        return row is not None

    def mark_processed(self, key: str, role: str, kind: str = "chat", prev: str = "") -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO processed(key, role, kind, seen_at, prev) VALUES(?,?,?,?,?)",
                (key, role, kind, time.time(), prev),
            )
            self._db.commit()

    def processed_info(self, key: str) -> Optional[tuple[float, str]]:
        """Return (seen_at, predecessor key) for a processed message."""
        with self._lock:
            row = self._db.execute(
                "SELECT seen_at, prev FROM processed WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        return (float(row["seen_at"]), str(row["prev"] or ""))

    def recently_processed(self, key: str, ttl_seconds: float = 60.0) -> bool:
        info = self.processed_info(key)
        if info is None:
            return False
        return (time.time() - info[0]) < ttl_seconds

    def mark_posted(self, key: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO posts(key, posted_at) VALUES(?,?)", (key, time.time())
            )
            self._db.commit()

    def is_posted(self, key: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM posts WHERE key=?", (key,)).fetchone()
        return row is not None

    # ---------------------------------------------------------- pending replies
    def add_pending_reply(self, text: str) -> int:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO pending_replies(created_at, text, attempts) VALUES(?,?,0)",
                (time.time(), text),
            )
            self._db.commit()
            return int(cursor.lastrowid or 0)

    def pending_replies(self, limit: int = 5) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._db.execute(
                    "SELECT * FROM pending_replies ORDER BY id ASC LIMIT ?", (limit,)
                )
            )

    def bump_pending_reply(self, reply_id: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE pending_replies SET attempts = attempts + 1 WHERE id=?", (reply_id,)
            )
            self._db.commit()

    def delete_pending_reply(self, reply_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM pending_replies WHERE id=?", (reply_id,))
            self._db.commit()

    # --------------------------------------------------------------- exchanges
    def record_exchange(
        self,
        session: str,
        request: str,
        status: str,
        response: str = "",
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO exchanges(started_at, finished_at, session, request, status, response)"
                " VALUES(?,?,?,?,?,?)",
                (
                    started_at or time.time(),
                    finished_at or time.time(),
                    session,
                    request,
                    status,
                    response,
                ),
            )
            self._db.commit()

    def count_since(self, seconds: float, executed_only: bool = True) -> int:
        """Count exchanges in a time window.

        Refusals (budget, loop guard, allow list) are not work, so they do not
        consume the budget.
        """
        cutoff = time.time() - seconds
        with self._lock:
            if executed_only:
                row = self._db.execute(
                    "SELECT COUNT(*) AS n FROM exchanges WHERE started_at >= ? "
                    "AND status NOT IN ('refused', 'error')",
                    (cutoff,),
                ).fetchone()
            else:
                row = self._db.execute(
                    "SELECT COUNT(*) AS n FROM exchanges WHERE started_at >= ?", (cutoff,)
                ).fetchone()
        return int(row["n"])

    def recent_exchange_with_request(self, request: str, seconds: float) -> int:
        """How often this exact request was already executed recently."""
        cutoff = time.time() - seconds
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM exchanges WHERE request=? AND started_at >= ?",
                (request, cutoff),
            ).fetchone()
        return int(row["n"])

    def recent_exchanges(self, limit: int = 10) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._db.execute("SELECT * FROM exchanges ORDER BY id DESC LIMIT ?", (limit,))
            )

    def last_exchange(self) -> Optional[sqlite3.Row]:
        rows = self.recent_exchanges(1)
        return rows[0] if rows else None

    def clear_exchanges(self) -> int:
        """Delete the exchange history (used to reset a spent budget)."""
        with self._lock:
            cursor = self._db.execute("DELETE FROM exchanges")
            self._db.commit()
        return int(cursor.rowcount or 0)

    def close(self) -> None:
        with self._lock:
            self._db.close()
