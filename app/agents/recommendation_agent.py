"""
recommendation_agent.py — Recommendation Agent

Combines Knowledge Tracing (mastery) with the Knowledge Graph (prerequisites)
to produce:
  - revision_topics: concepts the student is weak in (mastery < 0.5)
  - next_topics: concepts NOT yet attempted whose prerequisites are all
    mastered (readiness = fraction of prerequisites mastered)
"""
from app.agents import knowledge_tracing_agent as kt
from app.agents import knowledge_graph_agent as kg


def recommend(session_id: str) -> dict:
    all_mastery = kt.get_all_mastery(session_id)
    mastery_lookup = {c: d["mastery"] for c, d in all_mastery.items()}

    revision_topics = sorted(
        (
            {"concept": c, "mastery": d["mastery"], "attempts": d["attempts"]}
            for c, d in all_mastery.items() if d["mastery"] < 0.5
        ),
        key=lambda t: t["mastery"],
    )

    # Candidate "next topics": anything in the knowledge graph the student
    # hasn't attempted yet, whose prerequisites are (fully or mostly) mastered.
    attempted = set(all_mastery.keys())
    candidates = set(kg.PREREQUISITES.keys()) | {p for prereqs in kg.PREREQUISITES.values() for p in prereqs}
    next_topics = []
    for concept in candidates - attempted:
        prereqs = kg.get_all_prerequisites(concept)
        if not prereqs:
            continue  # no prerequisite info -> not enough signal to recommend it yet
        mastered_count = sum(1 for p in prereqs if mastery_lookup.get(p, 0.0) >= 0.7)
        readiness = mastered_count / len(prereqs)
        if readiness >= 0.7:
            next_topics.append({"concept": concept, "readiness": readiness})

    next_topics.sort(key=lambda t: -t["readiness"])

    return {"revision_topics": revision_topics, "next_topics": next_topics}