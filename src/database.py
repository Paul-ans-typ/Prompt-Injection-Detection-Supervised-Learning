"""
Persistent chat history — SQLite via aiosqlite.

Schema
------
sessions   — one row per A/B test conversation
exchanges  — one row per user turn (both side results stored together)

The DB file lives at  data/chat_history.db  (created on first startup).
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    threshold   REAL NOT NULL DEFAULT 0.5,
    a_config    TEXT NOT NULL DEFAULT '{}',
    b_config    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_EXCHANGES = """
CREATE TABLE IF NOT EXISTS exchanges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    user_text   TEXT    NOT NULL,

    a_verdict     TEXT,
    a_probability REAL,
    a_detector    TEXT,
    a_llm         TEXT,
    a_response    TEXT,
    a_blocked     INTEGER NOT NULL DEFAULT 0,
    a_detect_ms   REAL,
    a_total_ms    REAL,

    b_verdict     TEXT,
    b_probability REAL,
    b_detector    TEXT,
    b_llm         TEXT,
    b_response    TEXT,
    b_blocked     INTEGER NOT NULL DEFAULT 0,
    b_detect_ms   REAL,
    b_total_ms    REAL
);
"""

_CREATE_IDX = """
CREATE INDEX IF NOT EXISTS idx_exchanges_session
    ON exchanges(session_id, seq);
"""


async def init_db() -> None:
    """Create tables and indexes if they do not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute(_CREATE_SESSIONS)
        await db.execute(_CREATE_EXCHANGES)
        await db.execute(_CREATE_IDX)
        await db.commit()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

async def create_session(
    threshold: float,
    a_config: dict,
    b_config: dict,
    title: str = "New conversation",
) -> str:
    """Insert a new session row and return its UUID."""
    sid = str(uuid.uuid4())
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, threshold, a_config, b_config) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, title, now, now, threshold,
             json.dumps(a_config), json.dumps(b_config)),
        )
        await db.commit()
    return sid


async def list_sessions() -> list[dict]:
    """Return all sessions newest-first, each with a message_count field."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   s.threshold, s.a_config, s.b_config,
                   COUNT(e.id) AS message_count
              FROM sessions s
              LEFT JOIN exchanges e ON e.session_id = s.id
             GROUP BY s.id
             ORDER BY s.updated_at DESC
            """
        )
        rows = await cur.fetchall()
    return [_session_row(r) for r in rows]


async def get_session(session_id: str) -> Optional[dict]:
    """Return a session with all its exchanges, or None if not found."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        s = await cur.fetchone()
        if s is None:
            return None
        cur = await db.execute(
            "SELECT * FROM exchanges WHERE session_id = ? ORDER BY seq",
            (session_id,),
        )
        exchanges = await cur.fetchall()
    return {**_session_row(s), "exchanges": [_exchange_row(e) for e in exchanges]}


async def update_session_title(session_id: str, title: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        await db.commit()


async def delete_session(session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Exchanges
# ---------------------------------------------------------------------------

async def save_exchange(
    session_id: str,
    user_text: str,
    side_a: dict,
    side_b: dict,
) -> int:
    """Append one exchange to the session; returns the new exchange id."""
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cur = await db.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM exchanges WHERE session_id = ?",
            (session_id,),
        )
        (seq,) = await cur.fetchone()
        cur = await db.execute(
            """
            INSERT INTO exchanges
              (session_id, seq, created_at, user_text,
               a_verdict, a_probability, a_detector, a_llm,
               a_response, a_blocked, a_detect_ms, a_total_ms,
               b_verdict, b_probability, b_detector, b_llm,
               b_response, b_blocked, b_detect_ms, b_total_ms)
            VALUES
              (?, ?, ?, ?,
               ?, ?, ?, ?, ?, ?, ?, ?,
               ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, seq, now, user_text,
                side_a.get("verdict"),    side_a.get("probability"),
                side_a.get("detector"),   side_a.get("llm"),
                side_a.get("response"),   int(side_a.get("blocked", False)),
                side_a.get("detect_ms"),  side_a.get("total_ms"),
                side_b.get("verdict"),    side_b.get("probability"),
                side_b.get("detector"),   side_b.get("llm"),
                side_b.get("response"),   int(side_b.get("blocked", False)),
                side_b.get("detect_ms"),  side_b.get("total_ms"),
            ),
        )
        exchange_id = cur.lastrowid
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await db.commit()
    return exchange_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_row(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["a_config"] = json.loads(d["a_config"])
    d["b_config"] = json.loads(d["b_config"])
    return d


def _exchange_row(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["a_blocked"] = bool(d["a_blocked"])
    d["b_blocked"] = bool(d["b_blocked"])
    return d
