"""
db.py — shared SQLite connection used by memory_agent and knowledge_tracing_agent,
so student data (goals, conversation history, mistakes, mastery) survives a
server restart, unlike a plain in-memory dict.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "lumino_ai.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn