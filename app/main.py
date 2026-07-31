"""
AI-Optimized Learning Engine - Day 1 + Day 2 + Day 3 + Day 4 Backend
FastAPI + ChromaDB + Groq (Llama-3) RAG + Bandit RL + Multi-Agent
(Router, Memory, Knowledge Tracing, Knowledge Graph, Tutoring, Recommendation,
Mentor, Challenge/MCQ+Project)

Run: uvicorn app.main:app --reload
"""
import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import UPLOAD_DIR
from app.pdf_processor import process_pdf
from app.vector_store import add_chunks, query_chunks, list_documents, delete_document
from app.groq_client import generate_answer
from app import bandit
from app.agents import memory_agent
from app.agents import knowledge_tracing_agent as kt
from app.agents import knowledge_graph_agent as kg
from app.agents import recommendation_agent as rec
from app.agents import mentor_agent
from app.agents import orchestrator
from app.agents import challenge_agents as challenge_agent  # Day 4: MCQ test + project suggestions

app = FastAPI(
    title="AI-Optimized Learning Engine",
    description="Adaptive AI Learning Assistant - Hybrid RAG + Knowledge Tracing + RL + Multi-Agent",
    version="0.4.0-day4",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Creates DB tables if they don't exist (idempotent - CREATE TABLE IF NOT EXISTS).
# Called directly at module import time (not via @app.on_event) so it's reliable
# across every ASGI runner, including test clients that skip the lifespan events.
memory_agent.init_db()


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None
    session_id: str = "default_student"
    style: str = "auto"


class AskResponse(BaseModel):
    answer: str
    sources: list[int]
    style_used: str
    was_auto_selected: bool


class FeedbackRequest(BaseModel):
    session_id: str = "default_student"
    style: str
    reward: float


# ---------------- Day 3: Multi-Agent models ----------------

class AgentAskRequest(BaseModel):
    question: str
    doc_id: str | None = None
    session_id: str = "default_student"
    style: str = "auto"
    tutoring_mode: str = "auto"


class AgentAskResponse(BaseModel):
    answer: str
    sources: list[int] = []
    route: str
    concept: str | None = None
    style_used: str | None = None
    style_auto: bool = False
    tutoring_mode_used: str | None = None
    tutoring_mode_auto: bool = False
    difficulty: str | None = None
    mastery: float | None = None


class AgentFeedbackRequest(BaseModel):
    session_id: str = "default_student"
    concept: str | None = None
    style: str | None = None
    tutoring_mode: str | None = None
    thumbs_reward: float | None = None
    tutoring_reward: float | None = None
    correct: bool | None = None


class GoalRequest(BaseModel):
    session_id: str = "default_student"
    goal_text: str


# ---------------- Day 4: Challenge Agent models (MCQ test) ----------------

class MCQSubmitRequest(BaseModel):
    test_id: str
    session_id: str = "default_student"
    answers: dict[str, int]


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "AI Learning Engine is running!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be uploaded.")

    doc_id = str(uuid.uuid4())[:8] + "_" + os.path.splitext(file.filename)[0].replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        records = process_pdf(save_path, doc_id)
        if not records:
            raise HTTPException(status_code=422, detail="No text found in PDF. It may be a scanned image PDF.")
        add_chunks(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_created": len(records),
        "message": "PDF processed successfully and is ready to search!",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    auto_selected = req.style == "auto"
    style = bandit.select_style(req.session_id) if auto_selected else req.style

    chunks = query_chunks(req.question, doc_id=req.doc_id)
    result = generate_answer(req.question, chunks, style=style)

    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        style_used=result.get("style_used", style),
        was_auto_selected=auto_selected,
    )


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    if req.reward not in (0.0, 1.0):
        raise HTTPException(status_code=400, detail="reward must be either 0.0 (👎) or 1.0 (👍).")

    try:
        bandit.update_reward(req.session_id, req.style, req.reward)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Feedback recorded. Thank you!", "updated_stats": bandit.get_stats(req.session_id)}


@app.get("/bandit/stats/{session_id}")
def bandit_stats(session_id: str):
    return {"session_id": session_id, "stats": bandit.get_stats(session_id)}


@app.delete("/bandit/stats/{session_id}")
def reset_bandit_stats(session_id: str):
    bandit.reset_session(session_id)
    return {"message": f"Bandit history for {session_id} has been reset."}


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    delete_document(doc_id)
    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    return {"message": f"{doc_id} has been removed."}


# =============================================================================
# Day 3: Multi-Agent Architecture endpoints
# =============================================================================

@app.post("/agent/ask", response_model=AgentAskResponse)
def agent_ask(req: AgentAskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    result = orchestrator.handle_question(
        question=req.question,
        doc_id=req.doc_id,
        session_id=req.session_id,
        style_override=req.style,
        mode_override=req.tutoring_mode,
    )

    return AgentAskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        route=result["route"],
        concept=result.get("concept"),
        style_used=result.get("style_used"),
        style_auto=result.get("style_auto", False),
        tutoring_mode_used=result.get("tutoring_mode_used"),
        tutoring_mode_auto=result.get("tutoring_mode_auto", False),
        difficulty=result.get("difficulty"),
        mastery=result.get("mastery"),
    )


@app.post("/agent/feedback")
def agent_feedback(req: AgentFeedbackRequest):
    updated = {}

    if req.thumbs_reward is not None and req.style:
        bandit.update_reward(req.session_id, req.style, req.thumbs_reward)
        updated["explanation_style_stats"] = bandit.get_stats(req.session_id)

    if req.tutoring_reward is not None and req.tutoring_mode:
        bandit.update_tutoring_reward(req.session_id, req.tutoring_mode, req.tutoring_reward)
        updated["tutoring_mode_stats"] = bandit.get_tutoring_stats(req.session_id)

    if req.correct is not None:
        if not req.concept:
            raise HTTPException(status_code=400, detail="If 'correct' is given, 'concept' must also be given.")
        new_mastery = kt.update_mastery(req.session_id, req.concept, req.correct)
        updated["new_mastery"] = round(new_mastery, 3)
        if not req.correct:
            memory_agent.log_mistake(req.session_id, req.concept)
        else:
            try:
                updated["recommendations"] = rec.recommend(req.session_id)
            except Exception:
                pass

    if not updated:
        raise HTTPException(status_code=400, detail="At least one feedback field (thumbs_reward/tutoring_reward/correct) is required.")

    return {"message": "Feedback recorded. Thank you!", "updated": updated}


@app.get("/kt/mastery/{session_id}")
def get_mastery(session_id: str):
    all_mastery = kt.get_all_mastery(session_id)
    return {
        "session_id": session_id,
        "mastery": {
            c: {**v, "mastery": round(v["mastery"], 3), "tier": kt.difficulty_tier(v["mastery"])}
            for c, v in all_mastery.items()
        },
    }


@app.delete("/kt/mastery/{session_id}")
def reset_mastery_endpoint(session_id: str):
    """Wipe all progress data for a student — used by the Reset Progress button."""
    kt.reset_mastery(session_id)
    return {"message": f"Mastery data for '{session_id}' has been reset."}


@app.get("/kg/prerequisites/{concept}")
def get_prerequisites(concept: str):
    return {
        "concept": concept,
        "direct_prerequisites": kg.get_prerequisites(concept),
        "all_prerequisites": kg.get_all_prerequisites(concept),
        "leads_to": kg.get_children(concept),
    }


@app.get("/kg/gaps/{session_id}/{concept}")
def get_gaps(session_id: str, concept: str, threshold: float = 0.5):
    all_mastery = kt.get_all_mastery(session_id)
    mastery_lookup = {c: v["mastery"] for c, v in all_mastery.items()}
    gaps = kg.detect_gaps(concept, mastery_lookup, threshold=threshold)
    return {"concept": concept, "session_id": session_id, "gaps": gaps, "ready": len(gaps) == 0}


@app.get("/recommend/{session_id}")
def get_recommendations(session_id: str):
    return {"session_id": session_id, **rec.recommend(session_id)}


@app.get("/mentor/check/{session_id}")
def mentor_check(session_id: str):
    """
    Proactive Mentor Agent: struggle / progress / ready messages, plus a
    mini-project suggestion for any concept that just crossed the "advanced"
    tier (>= 0.85 mastery, at least 3 attempts).
    """
    messages = mentor_agent.check_progress(session_id)

    all_mastery = kt.get_all_mastery(session_id)
    for concept, data in all_mastery.items():
        if data.get("mastery", 0) >= 0.85 and data.get("attempts", 0) >= 3:
            try:
                idea = challenge_agent.suggest_project(session_id, concept, data["mastery"])
                messages.append({"type": "project", "concept": concept, "message": idea})
            except Exception:
                pass

    return {"session_id": session_id, "messages": messages}


@app.get("/mentor/project-ideas/{student_id}")
def get_project_ideas_endpoint(student_id: str):
    """Mini-project recommendation based on the student's mastered concepts."""
    mastered = kt.get_mastered_concepts(student_id)
    if not mastered:
        return {"ideas": "Keep learning! Once you master a few concepts, I'll suggest a hands-on project for you."}

    target_concept = mastered[-1]
    try:
        idea = challenge_agent.suggest_project(student_id, target_concept, 1.0)
        return {"ideas": idea}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mentor/assign-test/{session_id}/{concept}")
def assign_mentor_test(session_id: str, concept: str, n_questions: int = 5):
    if n_questions < 1 or n_questions > 15:
        raise HTTPException(status_code=400, detail="n_questions must be between 1-15.")

    all_mastery = kt.get_all_mastery(session_id)
    mastery = all_mastery.get(concept, {}).get("mastery", 0.5)

    try:
        test = challenge_agent.generate_mcq(session_id, concept, mastery=mastery, n_questions=n_questions)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return test


@app.post("/mentor/submit-test")
def submit_mentor_test(req: MCQSubmitRequest):
    try:
        graded = challenge_agent.submit_mcq(req.test_id, req.session_id, req.answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    concept = graded["concept"]
    for r in graded["results"]:
        kt.update_mastery(req.session_id, concept, r["is_correct"])
        if not r["is_correct"]:
            memory_agent.log_mistake(req.session_id, concept)

    updated_all = kt.get_all_mastery(req.session_id)
    graded["updated_mastery"] = round(updated_all.get(concept, {}).get("mastery", 0.0), 3)

    return graded


@app.post("/memory/goal")
def add_goal(req: GoalRequest):
    memory_agent.add_goal(req.session_id, req.goal_text)
    return {"message": "Goal added.", "goals": memory_agent.get_goals(req.session_id)}


@app.get("/memory/goals/{session_id}")
def get_goals(session_id: str):
    return {"session_id": session_id, "goals": memory_agent.get_goals(session_id)}


@app.get("/memory/history/{session_id}")
def get_history(session_id: str, limit: int = 10):
    return {"session_id": session_id, "history": memory_agent.get_recent_conversations(session_id, limit)}