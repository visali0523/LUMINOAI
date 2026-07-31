"""
orchestrator.py — Router Agent / Tutoring Decision Engine

Full pipeline for a single question:
  1. Concept extraction (LLM) happens FIRST, from the question alone, so the
     right difficulty level can be baked into the answer prompt itself.
  2. Look up current mastery for that concept -> difficulty tier.
  3. Adaptive style (bandit) + adaptive tutoring mode (bandit).
  4. Query Understanding: RAG vs General Knowledge, based on document
     retrieval similarity score (not just "was a doc_id given").
  5. Generate the answer via the chosen route, with style/tutoring_mode/
     difficulty all passed straight into the prompt.
  6. Quality Checker: retrieval score threshold + LLM self-refusal text check.
  7. Log the exchange to the Memory Agent.
"""
from app.vector_store import query_chunks
from app.groq_client import generate_answer, generate_general_answer, extract_concept
from app import bandit
from app.agents import knowledge_tracing_agent as kt
from app.agents import memory_agent

RAG_CONFIDENCE_THRESHOLD = 0.45

LOW_CONFIDENCE_PHRASES = [
    "not in the uploaded document",
    "not present in the uploaded document",
    "i don't know",
    "i'm not sure",
    "cannot find",
]


def _looks_low_confidence(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in LOW_CONFIDENCE_PHRASES)


def _pick_tutoring_mode(session_id: str, mode_override: str) -> tuple[str, bool]:
    """If mode_override != 'auto', use it directly. Otherwise let the
    tutoring-mode bandit (RL) pick based on this student's past feedback."""
    if mode_override != "auto":
        return mode_override, False
    return bandit.select_tutoring_mode(session_id), True


def handle_question(question: str, doc_id: str | None, session_id: str,
                     style_override: str = "auto", mode_override: str = "auto") -> dict:

    # --- Concept + difficulty lookup (BEFORE generating the answer) --------
    concept = extract_concept(question)
    if not concept or concept.strip().lower() in ("unknown concept", "none", ""):
        concept = None

    current_mastery = None
    if concept:
        all_mastery = kt.get_all_mastery(session_id)
        current_mastery = all_mastery.get(concept, {}).get("mastery")

    difficulty = kt.difficulty_tier(current_mastery) if current_mastery is not None else "beginner"

    style_auto = style_override == "auto"
    style = bandit.select_style(session_id) if style_auto else style_override

    tutoring_mode_used, tutoring_mode_auto = _pick_tutoring_mode(session_id, mode_override)

    # --- Query Understanding Engine: "Document Found?" ----------------------
    chunks: list[dict] = []
    retrieval_confidence = 0.0
    if doc_id:
        chunks = query_chunks(question, doc_id=doc_id)
        retrieval_confidence = chunks[0]["score"] if chunks else 0.0

    document_found = doc_id is not None and retrieval_confidence >= RAG_CONFIDENCE_THRESHOLD

    # --- Route to RAG Search or General LLM Knowledge -----------------------
    if document_found:
        result = generate_answer(question, chunks, style=style,
                                  tutoring_mode=tutoring_mode_used, difficulty=difficulty)
        route = "rag"
    else:
        result = generate_general_answer(question, style=style,
                                          tutoring_mode=tutoring_mode_used, difficulty=difficulty)
        route = "general"

    # --- Quality Checker ------------------------------------------------------
    low_confidence = _looks_low_confidence(result["answer"])
    quality_ok = not low_confidence and (route != "rag" or retrieval_confidence >= RAG_CONFIDENCE_THRESHOLD)

    if not quality_ok:
        return {
            "answer": "I'm not confident I have a good answer to that yet — could you rephrase "
                      "the question, or add a bit more detail / context?",
            "sources": [],
            "route": "low_confidence",
            "concept": None,
            "style_used": style,
            "style_auto": style_auto,
            "tutoring_mode_used": None,
            "tutoring_mode_auto": False,
            "difficulty": "beginner",
            "mastery": None,
        }

    try:
        memory_agent.log_conversation(session_id, question, result["answer"], route=route, concept=concept)
    except Exception:
        pass  # never let logging failures break the main answer flow

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "route": route,
        "concept": concept,
        "style_used": result.get("style_used", style),
        "style_auto": style_auto,
        "tutoring_mode_used": result.get("tutoring_mode_used", tutoring_mode_used),
        "tutoring_mode_auto": tutoring_mode_auto,
        "difficulty": result.get("difficulty_used", difficulty),
        "mastery": current_mastery,
    }