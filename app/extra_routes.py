"""
extra_routes.py
Backend routes to match the Day 3 Streamlit frontend contract:
  POST /agent/ask
  POST /agent/feedback
  GET  /kt/mastery/{student_id}
  POST /kt/mastery/{student_id}      (manual override, useful for testing)
  GET  /recommend/{student_id}
  GET  /mentor/check/{student_id}
  POST /memory/goal
  GET  /memory/goals/{student_id}

All storage is in-memory (dicts) — same pattern as bandit.py. Swap for a
real DB later.

KNOWN STUB LIMITATIONS (flagged so nothing looks silently "done"):
  - difficulty is hardcoded to "beginner" (no adaptive difficulty engine yet).
  - /recommend next_topics is always [] (no prerequisite knowledge graph yet).
  - tutoring_reward feedback is accepted but not yet fed into any RL model.
  - No persistent memory DB yet, so route can only be "rag" / "general" /
    "low_confidence" — the "Previous Memory?" branch from your diagram isn't
    built yet.
  - extract_concept() makes 1 extra Groq API call per answered question
    (skipped for low-confidence answers) — adds a bit of latency + cost.

HOW TO WIRE THIS IN (app/main.py, after `app = FastAPI(...)`):
    from app.extra_routes import router as extra_router
    app.include_router(extra_router)
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.vector_store import query_chunks
from app.groq_client import (
    generate_answer,
    generate_general_answer,
    extract_concept,
    generate_mcqs,
    generate_mentor_message,
    generate_project_ideas,
)
from app import bandit

router = APIRouter()

# ---------------------------------------------------------------------------
# Query Understanding Engine config
# ---------------------------------------------------------------------------
# Cosine similarity (0..1) the TOP retrieved chunk must clear for us to trust
# the document enough to answer from it. Below this -> treated as "document
# doesn't have it", falls through to General Question (LLM knowledge) branch.
# Tune this by testing real questions against your uploaded PDFs.
RAG_CONFIDENCE_THRESHOLD = 0.45

# Phrases that signal the RAG-restricted model itself doesn't have the answer
# (this is the "LLM self-rated confidence" half of the Quality Checker — no
# extra API call needed, we just read the model's own refusal).
LOW_CONFIDENCE_PHRASES = [
    "not present in the uploaded document",
    "i don't know",
    "i'm not sure",
    "cannot find",
]


def _looks_low_confidence(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in LOW_CONFIDENCE_PHRASES)

# ---------------------------------------------------------------------------
# In-memory "databases" (per student_id). Replace with real DB later.
# ---------------------------------------------------------------------------
_mastery_db: dict[str, dict[str, dict]] = {}   # student_id -> concept -> {attempts, correct_count, mastery}
_goals_db: dict[str, list[dict]] = {}          # student_id -> [ {goal_text, achieved, created_at} ]


def _tier(mastery: float) -> str:
    if mastery < 0.4:
        return "beginner"
    if mastery < 0.75:
        return "intermediate"
    return "advanced"


def _record_attempt(student_id: str, concept: str, correct: bool):
    student = _mastery_db.setdefault(student_id, {})
    entry = student.setdefault(concept, {"attempts": 0, "correct_count": 0, "mastery": 0.0})
    entry["attempts"] += 1
    if correct:
        entry["correct_count"] += 1
    entry["mastery"] = entry["correct_count"] / entry["attempts"]


def _mastery_with_tier(student_id: str) -> dict:
    student = _mastery_db.get(student_id, {})
    return {c: {**d, "tier": _tier(d["mastery"])} for c, d in student.items()}


# ---------------------------------------------------------------------------
# AGENT: /agent/ask, /agent/feedback
# ---------------------------------------------------------------------------
class AgentAskRequest(BaseModel):
    question: str
    doc_id: str | None = None
    session_id: str = "default_student"
    style: str = "auto"
    tutoring_mode: str = "auto"


@router.post("/agent/ask")
def agent_ask(req: AgentAskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    style = bandit.select_style(req.session_id) if req.style == "auto" else req.style
    tutoring_mode_used = "direct_answer" if req.tutoring_mode == "auto" else req.tutoring_mode

    # --- Query Understanding Engine: "Document Found?" -----------------
    chunks: list[dict] = []
    retrieval_confidence = 0.0
    if req.doc_id:
        chunks = query_chunks(req.question, doc_id=req.doc_id)
        retrieval_confidence = chunks[0]["score"] if chunks else 0.0

    document_found = req.doc_id is not None and retrieval_confidence >= RAG_CONFIDENCE_THRESHOLD

    # --- Route to RAG Search or LLM Knowledge ---------------------------
    if document_found:
        result = generate_answer(req.question, chunks, style=style)
        route = "rag"
    else:
        result = generate_general_answer(req.question, style=style)
        route = "general"

    # --- Quality Checker: retrieval score threshold + LLM self-refusal --
    low_confidence = _looks_low_confidence(result["answer"])
    if route == "rag":
        quality_ok = not low_confidence and retrieval_confidence >= RAG_CONFIDENCE_THRESHOLD
    else:
        quality_ok = not low_confidence

    if not quality_ok:
        return {
            "answer": "I'm not confident I have a good answer to that yet — could you rephrase "
                      "the question, or add a bit more detail / context?",
            "sources": [],
            "route": "low_confidence",
            "concept": None,
            "style_used": style,
            "tutoring_mode_used": tutoring_mode_used,
            "difficulty": "beginner",
        }

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "route": route,
        "concept": extract_concept(req.question, result["answer"]),
        "style_used": result.get("style_used", style),
        "tutoring_mode_used": tutoring_mode_used,
        "difficulty": "beginner",  # TODO: adaptive difficulty not implemented yet
    }


class AgentFeedbackRequest(BaseModel):
    session_id: str
    style: str | None = None
    thumbs_reward: float | None = None
    tutoring_mode: str | None = None
    tutoring_reward: float | None = None
    concept: str | None = None
    correct: bool | None = None


@router.post("/agent/feedback")
def agent_feedback(req: AgentFeedbackRequest):
    if req.style is not None and req.thumbs_reward is not None:
        bandit.update_reward(req.session_id, req.style, req.thumbs_reward)

    if req.concept is not None and req.correct is not None:
        _record_attempt(req.session_id, req.concept, req.correct)

    # tutoring_reward accepted but not yet used by any RL model (stub)

    return {"message": "Feedback recorded."}


# ---------------------------------------------------------------------------
# KNOWLEDGE TRACING: /kt/mastery
# ---------------------------------------------------------------------------
class MasteryUpdateRequest(BaseModel):
    topic: str
    mastery: float  # 0.0 to 1.0


@router.get("/kt/mastery/{student_id}")
def get_mastery(student_id: str):
    return {"student_id": student_id, "mastery": _mastery_with_tier(student_id)}


@router.post("/kt/mastery/{student_id}")
def update_mastery(student_id: str, req: MasteryUpdateRequest):
    """Manual override — handy for seeding/testing the Progress tab directly."""
    student = _mastery_db.setdefault(student_id, {})
    entry = student.setdefault(req.topic, {"attempts": 0, "correct_count": 0, "mastery": 0.0})
    entry["mastery"] = max(0.0, min(1.0, req.mastery))
    entry["attempts"] = max(entry["attempts"], 1)
    entry["correct_count"] = max(entry["correct_count"], round(entry["mastery"] * entry["attempts"]))
    return {"student_id": student_id, "mastery": _mastery_with_tier(student_id)}


@router.delete("/kt/mastery/{student_id}")
def reset_mastery(student_id: str):
    """Wipe all mastery/progress data for a student — for testing/demo resets."""
    _mastery_db.pop(student_id, None)
    return {"message": f"Mastery data for '{student_id}' has been reset."}


# ---------------------------------------------------------------------------
# RECOMMENDATION AGENT
# ---------------------------------------------------------------------------
@router.get("/recommend/{student_id}")
def get_recommendations(student_id: str):
    student = _mastery_db.get(student_id, {})

    revision_topics = sorted(
        (
            {"concept": c, "mastery": d["mastery"], "attempts": d["attempts"]}
            for c, d in student.items() if d["mastery"] < 0.5
        ),
        key=lambda t: t["mastery"],
    )

    next_topics: list[dict] = []  # TODO: needs a prerequisite knowledge graph

    return {"student_id": student_id, "revision_topics": revision_topics, "next_topics": next_topics}


# ---------------------------------------------------------------------------
# PROACTIVE MENTOR AGENT
# ---------------------------------------------------------------------------
@router.get("/mentor/check/{student_id}")
def mentor_check(student_id: str):
    student = _mastery_db.get(student_id, {})
    messages = []

    if student:
        weakest_concept, weakest = min(student.items(), key=lambda kv: kv[1]["mastery"])

        if weakest["mastery"] < 0.4 and weakest["attempts"] >= 2:
            messages.append({
                "type": "struggle",
                "concept": weakest_concept,
                "message": generate_mentor_message(weakest_concept, weakest["mastery"], "struggle"),
            })
        elif all(d["mastery"] >= 0.8 for d in student.values()):
            messages.append({
                "type": "ready",
                "concept": None,
                "message": generate_mentor_message("", 0.0, "ready"),
            })
        else:
            messages.append({
                "type": "progress",
                "concept": weakest_concept,
                "message": generate_mentor_message(weakest_concept, weakest["mastery"], "progress"),
            })

    return {"student_id": student_id, "messages": messages}


@router.get("/mentor/project-ideas/{student_id}")
def mentor_project_ideas(student_id: str):
    """Mini-project suggestions based on concepts the student has mastered (tier == advanced)."""
    student = _mastery_db.get(student_id, {})
    mastered = [c for c, d in student.items() if d["mastery"] >= 0.8]
    ideas = generate_project_ideas(mastered)
    return {"student_id": student_id, "mastered_concepts": mastered, "ideas": ideas}


# ---------------------------------------------------------------------------
# PRACTICE AGENT (MCQ generation for weak concepts)
# ---------------------------------------------------------------------------
class PracticeGenerateRequest(BaseModel):
    student_id: str
    concept: str
    num_questions: int = 3


@router.post("/practice/generate")
def practice_generate(req: PracticeGenerateRequest):
    questions = generate_mcqs(req.concept, req.num_questions)
    if not questions:
        raise HTTPException(
            status_code=502,
            detail="Couldn't generate practice questions right now — try again in a moment.",
        )
    return {"concept": req.concept, "questions": questions}


# ---------------------------------------------------------------------------
# MEMORY AGENT: goals
# ---------------------------------------------------------------------------
class GoalCreateRequest(BaseModel):
    session_id: str
    goal_text: str


@router.post("/memory/goal")
def add_goal(req: GoalCreateRequest):
    goal = {
        "goal_text": req.goal_text,
        "achieved": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _goals_db.setdefault(req.session_id, []).append(goal)
    return {"session_id": req.session_id, "goals": _goals_db[req.session_id]}


@router.get("/memory/goals/{student_id}")
def get_goals(student_id: str):
    return {"student_id": student_id, "goals": _goals_db.get(student_id, [])}