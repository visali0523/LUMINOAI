"""
Module that calls the Groq API (Llama-3) to generate responses.

Day 1: Strictly from RAG context only.
Day 2: Bandit-based selection of "style" (analogy/summary/detailed).
Day 3: + "tutoring_mode" (hint_first/direct_answer - Socratic teaching),
       + "difficulty" tier (from BKT mastery),
       + generate_general_answer() - Hybrid RAG fallback (if not found in the
         document, answer from general knowledge, clearly labeled).
"""
from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

STYLE_INSTRUCTIONS = {
    "analogy": (
        "Explain the answer using a simple, relatable real-world analogy or example "
        "that a school/college student would easily understand. Keep it conversational."
    ),
    "summary": (
        "Give a short, crisp, bullet-point summary. No fluff, no long paragraphs. "
        "Maximum 4-5 bullet points."
    ),
    "detailed": (
        "Give a thorough, step-by-step, in-depth explanation covering all nuances "
        "from the source material. Use headings if helpful."
    ),
    "default": (
        "Give a clear, well-structured, moderately detailed explanation."
    ),
}

TUTORING_MODE_INSTRUCTIONS = {
    "hint_first": (
        "IMPORTANT TEACHING RULE: Do NOT give the final/complete answer immediately. "
        "Instead, act like a Socratic tutor: ask a guiding question or give a small hint "
        "that helps the student work towards the answer themselves. Encourage them to think. "
        "Only give the full direct answer if the student's question explicitly says they "
        "already tried and are stuck, or explicitly asks for the answer directly."
    ),
    "direct_answer": (
        "Give the complete, direct, correct answer clearly and confidently."
    ),
}

DIFFICULTY_INSTRUCTIONS = {
    "beginner": (
        "The student is a BEGINNER at this topic. Use very simple language, basic vocabulary, "
        "and foundational examples. Avoid jargon; define any technical term you must use."
    ),
    "intermediate": (
        "The student has a MODERATE understanding of this topic. You can use standard "
        "terminology but still explain non-obvious concepts."
    ),
    "advanced": (
        "The student has ADVANCED / mastered understanding of this topic. You can use precise "
        "technical terminology, discuss edge cases, and go into deeper nuance without over-explaining basics."
    ),
}

RAG_SYSTEM_PROMPT_TEMPLATE = """You are an AI study tutor for a student. You must answer ONLY using the CONTEXT provided below, which comes from the student's own textbook/syllabus PDF.

STRICT RULES:
1. Answer strictly based on the CONTEXT. Do not use outside knowledge.
2. If the answer is not present in the CONTEXT, clearly say: "The answer to this question is not in the uploaded document." Do not make up information.
3. Always mention which page number(s) the answer came from, if available.
4. {style_instruction}
5. {tutoring_instruction}
6. {difficulty_instruction}

CONTEXT:
{context}
"""

GENERAL_SYSTEM_PROMPT_TEMPLATE = """You are an AI study tutor for a student. The student's uploaded document did NOT contain a good answer to this question, so you must answer using your own general knowledge instead.

RULES:
1. Begin your answer by clearly noting (in English, naturally) that this wasn't found in their uploaded material and this is general knowledge.
2. Be accurate. If you are not confident about a fact, say so honestly instead of guessing.
3. {style_instruction}
4. {tutoring_instruction}
5. {difficulty_instruction}
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        page_info = f"[Page {c['page']}]" if c.get("page") else ""
        parts.append(f"{page_info}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _resolve_instructions(style: str, tutoring_mode: str, difficulty: str) -> tuple[str, str, str]:
    return (
        STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["default"]),
        TUTORING_MODE_INSTRUCTIONS.get(tutoring_mode, TUTORING_MODE_INSTRUCTIONS["direct_answer"]),
        DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["intermediate"]),
    )


def generate_answer(question: str, chunks: list[dict], style: str = "default",
                     tutoring_mode: str = "direct_answer", difficulty: str = "intermediate") -> dict:
    """
    RAG Agent's answer generator: STRICTLY document-context-based (Day 1/2 behaviour,
    kept for backward compatibility with the simple /ask endpoint).
    """
    if _client is None:
        return {
            "answer": "⚠️ GROQ_API_KEY is not set. Put your Groq API key in the .env file.",
            "sources": [], "style_used": style, "tutoring_mode_used": tutoring_mode, "difficulty_used": difficulty,
        }

    if not chunks:
        return {
            "answer": "The answer to this question is not in the uploaded document.",
            "sources": [], "style_used": style, "tutoring_mode_used": tutoring_mode, "difficulty_used": difficulty,
        }

    context = build_context(chunks)
    style_instruction, tutoring_instruction, difficulty_instruction = _resolve_instructions(
        style, tutoring_mode, difficulty
    )

    system_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(
        style_instruction=style_instruction,
        tutoring_instruction=tutoring_instruction,
        difficulty_instruction=difficulty_instruction,
        context=context,
    )

    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    answer_text = completion.choices[0].message.content
    sources = sorted({c["page"] for c in chunks if c.get("page")})

    return {
        "answer": answer_text,
        "sources": sources,
        "style_used": style,
        "tutoring_mode_used": tutoring_mode,
        "difficulty_used": difficulty,
    }


def generate_general_answer(question: str, style: str = "default",
                             tutoring_mode: str = "direct_answer", difficulty: str = "intermediate") -> dict:
    """
    LLM Agent's answer generator: for questions not found in the document
    (Hybrid RAG fallback), gives an answer from general knowledge (clearly labelled).
    """
    if _client is None:
        return {
            "answer": "⚠️ GROQ_API_KEY is not set. Put your Groq API key in the .env file.",
            "sources": [], "style_used": style, "tutoring_mode_used": tutoring_mode, "difficulty_used": difficulty,
        }

    style_instruction, tutoring_instruction, difficulty_instruction = _resolve_instructions(
        style, tutoring_mode, difficulty
    )

    system_prompt = GENERAL_SYSTEM_PROMPT_TEMPLATE.format(
        style_instruction=style_instruction,
        tutoring_instruction=tutoring_instruction,
        difficulty_instruction=difficulty_instruction,
    )

    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=1024,
    )

    answer_text = completion.choices[0].message.content

    return {
        "answer": answer_text,
        "sources": [],
        "style_used": style,
        "tutoring_mode_used": tutoring_mode,
        "difficulty_used": difficulty,
    }
    
def generate_mentor_message(prompt: str, system_instruction: str = "You are a helpful AI study mentor.") -> dict:
    """
    LLM Agent for the Mentor feature to generate motivational or guiding messages.
    """
    if _client is None:
        return {
            "answer": "⚠️ GROQ_API_KEY is not set. Put your Groq API key in the .env file.",
        }

    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=1024,
    )

    answer_text = completion.choices[0].message.content

    return {
        "answer": answer_text
    }
def extract_concept(text: str) -> str:
    """
    LLM Agent to extract the core educational concept from a user's question.
    Used for tracking student mistakes and conversation history.
    """
    if _client is None:
        return "Unknown Concept"

    system_prompt = (
        "You are an API tool that extracts the core educational concept or topic "
        "from a student's question. Return ONLY the concept in 1-4 words. "
        "Do not add any conversational text, punctuation, or explanation. "
        "Example: If the user asks 'Why does the apple fall?', return 'Gravity'."
    )

    try:
        completion = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1, # Low temperature for consistent, strict output
            max_tokens=15,
        )
        concept = completion.choices[0].message.content.strip()
        return concept
    except Exception:
        return "Unknown Concept"
    
import json

def generate_mcqs(context: str, num_questions: int = 3) -> str:
    """
    LLM Agent to generate Multiple Choice Questions based on the provided context/topic.
    Expected to return a JSON string representing a list of questions.
    """
    if _client is None:
        return json.dumps([{"question": "⚠️ GROQ_API_KEY is not set.", "options": ["A", "B", "C", "D"], "answer": "A"}])

    system_prompt = (
        f"You are an expert educational assessor. Generate {num_questions} multiple-choice questions "
        "based strictly on the provided text or concept. Return ONLY a valid JSON array of objects. "
        "Each object must have the keys: 'question' (string), 'options' (array of 4 strings), and 'answer' (the exact string of the correct option). "
        "Do not include any conversational text or markdown formatting like ```json."
    )

    try:
        completion = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps([{"question": f"Error generating MCQs: {str(e)}", "options": ["A", "B"], "answer": "A"}])


def generate_project_ideas(concept: str) -> str:
    """
    LLM Agent to generate practical project ideas or real-world applications for a given concept.
    """
    if _client is None:
        return "⚠️ GROQ_API_KEY is not set. Put your Groq API key in the .env file."

    system_prompt = (
        "You are an inspiring STEM/Humanities mentor. Suggest 3 practical, engaging, and "
        "doable project ideas or real-world applications for a student to apply the given concept. "
        "Format the output nicely using markdown with bullet points or numbered lists. "
        "Keep the descriptions clear, creative, and motivating."
    )

    try:
        completion = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Give me project ideas for: {concept}"},
            ],
            temperature=0.7,  # Slightly higher temperature for more creative ideas
            max_tokens=1024,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating project ideas: {str(e)}"