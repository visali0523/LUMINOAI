"""
Tutoring Decision Engine (Adaptive Learning Agent + RL Reward Engine glue)

ஒரு கேள்விக்கு பதில் சொல்றதுக்கு முன்ன, இந்த agent முடிவு பண்றது:
  1. explanation_style : analogy / summary / detailed   (bandit - Day 2)
  2. tutoring_mode      : hint_first / direct_answer     (bandit - Day 3, "Explain instead
                          of simply answering" problem-ஐ தீர்க்கும் - Socratic method)
  3. difficulty_tier    : beginner / intermediate / advanced (BKT mastery-ல் இருந்து)

இதுவே "Adaptive Difficulty" மற்றும் "No Learning Strategy" (AI-ஐயே சார்ந்திருக்குறது)
problems-ஐ தீர்க்கிறது.
"""
from app import bandit
from app.agents import knowledge_tracing_agent as kt


def decide_strategy(session_id: str, concept: str | None, style_override: str = "auto",
                     mode_override: str = "auto") -> dict:
    """
    Full tutoring strategy-ஐ முடிவு பண்ணும்.

    style_override / mode_override: "auto" கொடுத்தா bandit தேர்வு பண்ணும்,
    இல்லனா manual value-ஐயே use பண்ணும் (UI-ல் override பண்ண option).
    """
    style_auto = style_override == "auto"
    mode_auto = mode_override == "auto"

    style = bandit.select_style(session_id) if style_auto else style_override
    tutoring_mode = bandit.select_tutoring_mode(session_id) if mode_auto else mode_override

    if concept:
        mastery = kt.get_mastery(session_id, concept)
        difficulty = kt.difficulty_tier(mastery)
    else:
        mastery = None
        difficulty = "intermediate"  # concept தெரியலனா, safe middle-ground default

    return {
        "style": style,
        "style_auto": style_auto,
        "tutoring_mode": tutoring_mode,
        "tutoring_mode_auto": mode_auto,
        "difficulty": difficulty,
        "mastery": round(mastery, 3) if mastery is not None else None,
    }
