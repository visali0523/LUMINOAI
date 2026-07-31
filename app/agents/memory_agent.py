"""
memory_agent.py — Memory Agent

Stores per-student learning goals, conversation history, and logged
mistakes in SQLite so they survive a server restart.
"""
from datetime import datetime, timezone

from app.agents.db import get_connection


def init_db():
    """Create tables if they don't exist yet. Safe to call every startup."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            goal_text TEXT NOT NULL,
            achieved INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            route TEXT,
            concept TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------
def add_goal(session_id: str, goal_text: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO goals (session_id, goal_text, achieved, created_at) VALUES (?, ?, 0, ?)",
        (session_id, goal_text, _now()),
    )
    conn.commit()
    conn.close()


def get_goals(session_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT goal_text, achieved, created_at FROM goals WHERE session_id = ? ORDER BY id DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"goal_text": r["goal_text"], "achieved": bool(r["achieved"]), "created_at": r["created_at"]} for r in rows]


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
def log_conversation(session_id: str, question: str, answer: str, route: str | None = None, concept: str | None = None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO conversations (session_id, question, answer, route, concept, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, question, answer, route, concept, _now()),
    )
    conn.commit()
    conn.close()


def get_recent_conversations(session_id: str, limit: int = 10) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT question, answer, route, concept, created_at FROM conversations "
        "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Mistakes log
# ---------------------------------------------------------------------------
def log_mistake(session_id: str, concept: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO mistakes (session_id, concept, created_at) VALUES (?, ?, ?)",
        (session_id, concept, _now()),
    )
    conn.commit()
    conn.close()


def get_mistakes(session_id: str, limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT concept, created_at FROM mistakes WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]