"""
Router Agent (Intelligent Router)

ஒரு மாணவன் கேட்கும் கேள்விக்கு, யார் பதில் சொல்லணும்-ன்னு decide பண்றது:
  1. Memory Agent  - "என் progress என்ன?", "நான் எங்க தப்பு பண்ணேன்?" போன்ற meta-questions
  2. RAG Agent     - Upload செய்த document-ல் நல்ல relevant content இருக்கா
  3. LLM Agent     - Document-ல் இல்லனா / document இல்லனா, general knowledge-ல் இருந்து பதில்

இதுவே "Static RAG" problem-ஐ தீர்க்கிறது - document-ல் இல்லனா "தெரியாது"-ன்னு
சொல்லாம, intelligently general knowledge-க்கு switch ஆகும் (Hybrid RAG).
"""
import re
from app.config import RAG_RELEVANCE_THRESHOLD

# மாணவனோட progress/history பத்தி கேட்கிற மாதிரி keywords/patterns
_MEMORY_PATTERNS = [
    r"\bmy progress\b", r"\bhow am i doing\b", r"\bweak (concept|topic|area)s?\b",
    r"\bwhat did i (get wrong|struggle)\b", r"\bmy mistakes?\b", r"\bmy goals?\b",
    r"\bwhat have i (learned|studied)\b", r"\bmy history\b", r"\bremind me\b",
    r"\bwhat.* i (asked|studied) (before|earlier|previously)\b",
    r"\bmy (mastery|score|performance)\b",
]


def is_memory_query(question: str) -> bool:
    """இது ஒரு 'என் history/progress' மாதிரி meta-question-ஆ-ன்னு check பண்ணும்."""
    q = question.lower()
    return any(re.search(pattern, q) for pattern in _MEMORY_PATTERNS)


def decide_source(chunks: list[dict], threshold: float = RAG_RELEVANCE_THRESHOLD) -> str:
    """
    Retrieved chunks-ஓட similarity score பார்த்து, "rag" (document-ல் நல்ல match)
    அல்லது "general" (document-ல் இல்ல, LLM general knowledge use பண்ணும்)-ன்னு decide பண்ணும்.
    """
    if not chunks:
        return "general"
    top_score = max(c.get("score", 0.0) for c in chunks)
    return "rag" if top_score >= threshold else "general"


def route(question: str, chunks: list[dict]) -> str:
    """
    முழு routing decision: "memory" | "rag" | "general"
    Router Agent-ஓட main entry point.
    """
    if is_memory_query(question):
        return "memory"
    return decide_source(chunks)
