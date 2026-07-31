"""
knowledge_tracing_agent.py — Knowledge Tracing Agent

Tracks per-student, per-concept mastery (0.0 to 1.0), persisted in SQLite
so it survives a server restart. Mastery = correct_count / attempts.
"""
from app.agents.db import get_connection


def _init_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mastery (
            session_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            correct_count INTEGER NOT NULL DEFAULT 0,
            mastery REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (session_id, concept)
        )
    """)
    conn.commit()
    conn.close()


_init_table()


def difficulty_tier(mastery: float) -> str:
    if mastery < 0.4:
        return "beginner"
    if mastery < 0.75:
        return "intermediate"
    return "advanced"


def update_mastery(session_id: str, concept: str, correct: bool) -> float:
    """Record one attempt for a concept and return the new mastery value."""
    conn = get_connection()
    row = conn.execute(
        "SELECT attempts, correct_count FROM mastery WHERE session_id = ? AND concept = ?",
        (session_id, concept),
    ).fetchone()

    if row is None:
        attempts, correct_count = 0, 0
    else:
        attempts, correct_count = row["attempts"], row["correct_count"]

    attempts += 1
    if correct:
        correct_count += 1
    new_mastery = correct_count / attempts

    conn.execute(
        """
        INSERT INTO mastery (session_id, concept, attempts, correct_count, mastery)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id, concept) DO UPDATE SET
            attempts = excluded.attempts,
            correct_count = excluded.correct_count,
            mastery = excluded.mastery
        """,
        (session_id, concept, attempts, correct_count, new_mastery),
    )
    conn.commit()
    conn.close()
    return new_mastery


def get_all_mastery(session_id: str) -> dict:
    """Returns {concept: {"attempts", "correct_count", "mastery"}} for a student."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT concept, attempts, correct_count, mastery FROM mastery WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    conn.close()
    return {
        r["concept"]: {"attempts": r["attempts"], "correct_count": r["correct_count"], "mastery": r["mastery"]}
        for r in rows
    }


def get_mastered_concepts(session_id: str, threshold: float = 0.8) -> list[str]:
    """Concepts at or above the given mastery threshold, most recently mastered first
    is not tracked (no timestamp column) — returned in insertion order instead."""
    all_mastery = get_all_mastery(session_id)
    return [c for c, d in all_mastery.items() if d["mastery"] >= threshold]


def reset_mastery(session_id: str):
    """Wipe all mastery data for a student — used by the Reset Progress button."""
    conn = get_connection()
    conn.execute("DELETE FROM mastery WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()