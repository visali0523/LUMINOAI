"""
mentor_agent.py — Proactive Mentor Agent

Rule-based logic decides WHEN to alert the student (struggling / steady
progress / all mastered); the wording is generated fresh each time by the
LLM (via groq_client.generate_mentor_message) so it doesn't repeat the
exact same sentence forever.

Note: groq_client.generate_mentor_message(prompt, system_instruction)
returns a dict shaped {"answer": "..."} — we build the prompt text here
and unwrap the "answer" key, falling back to a fixed message if the LLM
call fails for any reason.
"""
from app.agents import knowledge_tracing_agent as kt
from app.groq_client import generate_mentor_message

MENTOR_SYSTEM_INSTRUCTION = "You are a warm, encouraging AI learning mentor."


def _ask_mentor(prompt: str, fallback: str) -> str:
    result = generate_mentor_message(prompt, MENTOR_SYSTEM_INSTRUCTION)
    return result.get("answer") or fallback


def check_progress(session_id: str) -> list[dict]:
    all_mastery = kt.get_all_mastery(session_id)
    messages = []

    if not all_mastery:
        return messages

    weakest_concept, weakest = min(all_mastery.items(), key=lambda kv: kv[1]["mastery"])

    if weakest["mastery"] < 0.4 and weakest["attempts"] >= 2:
        prompt = (
            f"The student is struggling with the concept '{weakest_concept}' "
            f"(current mastery: {int(weakest['mastery'] * 100)}%). Write ONE short, warm, "
            f"encouraging message (1-2 sentences) suggesting they review it. Vary your "
            f"wording and tone each time — don't sound robotic or repeat stock phrases."
        )
        fallback = f"You're struggling with '{weakest_concept}' ({int(weakest['mastery'] * 100)}%). Let's review the basics."
        messages.append({"type": "struggle", "concept": weakest_concept, "message": _ask_mentor(prompt, fallback)})

    elif all(d["mastery"] >= 0.8 for d in all_mastery.values()):
        prompt = (
            "The student has mastered all their current tracked concepts. Write ONE short, "
            "warm, encouraging message (1-2 sentences) congratulating them and nudging them "
            "toward something more advanced. Vary your wording each time."
        )
        fallback = "You've mastered your current topics — ready for something more advanced?"
        messages.append({"type": "ready", "concept": None, "message": _ask_mentor(prompt, fallback)})

    else:
        prompt = (
            f"The student is making steady, in-progress mastery on '{weakest_concept}' "
            f"(currently {int(weakest['mastery'] * 100)}%) — not struggling, not fully mastered "
            f"yet. Write ONE short, warm, encouraging message (1-2 sentences) that acknowledges "
            f"their progress and nudges them to keep practicing this concept. Vary your wording "
            f"each time — don't sound robotic or repeat stock phrases."
        )
        fallback = f"You're making steady progress on '{weakest_concept}' ({int(weakest['mastery'] * 100)}%). Keep it up!"
        messages.append({"type": "progress", "concept": weakest_concept, "message": _ask_mentor(prompt, fallback)})

    return messages