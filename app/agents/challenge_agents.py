"""
challenge_agents.py — Challenge Agent (Day 4: MCQ tests + project suggestions)

Generates MCQ practice tests (via the LLM) and mini-project suggestions for
concepts the student has mastered. Test answer keys are kept server-side
only (in _tests_db) — the response sent to the frontend never includes the
correct answer, so students can't see it in the browser network tab.

Note: groq_client.generate_mcqs(context, num_questions) returns a raw JSON
STRING (not parsed), where each question object looks like:
    {"question": "...", "options": ["a","b","c","d"], "answer": "<exact option text>"}
We parse that JSON here and convert "answer" (exact text) into
"correct_index" (position in options), which is safer to grade against.

Note: groq_client.generate_project_ideas(concept) takes a single concept
string, not a list — suggest_project() passes just the one concept through.
"""
import json
import uuid

from app.groq_client import generate_mcqs, generate_project_ideas

# test_id -> {"session_id", "concept", "questions": [{"question","options","correct_index"}, ...]}
_tests_db: dict[str, dict] = {}


def _parse_mcqs(raw: str) -> list[dict]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        raw_questions = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    valid = []
    for q in raw_questions:
        question_text = q.get("question")
        options = q.get("options")
        answer = q.get("answer")
        if not (
            isinstance(question_text, str)
            and isinstance(options, list) and len(options) == 4
            and isinstance(answer, str) and answer in options
        ):
            continue  # skip anything malformed rather than crash the whole test
        valid.append({
            "question": question_text,
            "options": options,
            "correct_index": options.index(answer),
        })
    return valid


def generate_mcq(session_id: str, concept: str, mastery: float = 0.5, n_questions: int = 5) -> dict:
    """Generate an MCQ test for a concept. Raises RuntimeError if generation/parsing fails."""
    raw = generate_mcqs(concept, num_questions=n_questions)
    questions = _parse_mcqs(raw)
    if not questions:
        raise RuntimeError("Couldn't generate the practice test right now — try again in a moment.")

    test_id = str(uuid.uuid4())
    _tests_db[test_id] = {"session_id": session_id, "concept": concept, "questions": questions}

    # Strip correct_index before sending to the frontend.
    safe_questions = [{"question": q["question"], "options": q["options"]} for q in questions]
    return {"test_id": test_id, "concept": concept, "questions": safe_questions}


def submit_mcq(test_id: str, session_id: str, answers: dict[str, int]) -> dict:
    """
    Grade a submitted test. `answers` maps question index (as a string, e.g.
    "0", "1") to the chosen option index. Returns per-question results plus
    the concept, so the caller can feed each result into Knowledge Tracing.
    """
    test = _tests_db.get(test_id)
    if test is None:
        raise ValueError("Test not found or already submitted.")
    if test["session_id"] != session_id:
        raise ValueError("This test does not belong to this student.")

    results = []
    correct_count = 0
    for i, q in enumerate(test["questions"]):
        chosen = answers.get(str(i))
        is_correct = chosen is not None and chosen == q["correct_index"]
        if is_correct:
            correct_count += 1
        results.append({
            "question": q["question"],
            "chosen_index": chosen,
            "correct_index": q["correct_index"],
            "is_correct": is_correct,
        })

    del _tests_db[test_id]  # one-time use — free the memory once graded

    return {
        "concept": test["concept"],
        "score": correct_count,
        "total": len(test["questions"]),
        "results": results,
    }


def suggest_project(session_id: str, concept: str, mastery: float) -> str:
    """Suggest mini-project ideas built around a mastered concept."""
    return generate_project_ideas(concept)