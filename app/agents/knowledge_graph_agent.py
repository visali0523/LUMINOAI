"""
knowledge_graph_agent.py — Knowledge Graph Agent

Holds a small hardcoded prerequisite graph (concept -> list of concepts it
depends on). This is a STUB starter graph — extend PREREQUISITES with real
concepts from your syllabus. Without entries here, every concept is treated
as having no prerequisites (always "ready").
"""

# concept -> list of concepts that must be learned first
PREREQUISITES: dict[str, list[str]] = {
    "Loops": ["Variables"],
    "Functions": ["Variables", "Loops"],
    "Recursion": ["Functions"],
    "Machine Learning": ["Python", "Statistics", "Linear Algebra"],
    "Neural Networks": ["Machine Learning"],
    # Add more concept: [prerequisite, ...] entries as your syllabus grows.
}


def get_prerequisites(concept: str) -> list[str]:
    """Direct prerequisites only."""
    return PREREQUISITES.get(concept, [])


def get_all_prerequisites(concept: str) -> list[str]:
    """Transitive prerequisites (prerequisites of prerequisites, etc.)."""
    seen: set[str] = set()
    to_visit = list(get_prerequisites(concept))
    while to_visit:
        current = to_visit.pop()
        if current in seen:
            continue
        seen.add(current)
        to_visit.extend(get_prerequisites(current))
    return sorted(seen)


def get_children(concept: str) -> list[str]:
    """Concepts that list this concept as a direct prerequisite (what it unlocks)."""
    return sorted(c for c, prereqs in PREREQUISITES.items() if concept in prereqs)


def detect_gaps(concept: str, mastery_lookup: dict[str, float], threshold: float = 0.5) -> list[str]:
    """Which of this concept's prerequisites does the student have low mastery in?"""
    prereqs = get_all_prerequisites(concept)
    return [p for p in prereqs if mastery_lookup.get(p, 0.0) < threshold]