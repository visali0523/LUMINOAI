"""
Generic Epsilon-Greedy Multi-Armed Bandit engine (RL / Reward Engine).

இது ஒரு reusable bandit core - பல வேற "decisions"-க்கும் (namespaces) பயன்படுத்தலாம்:
  1. "explanation_style" -> analogy / summary / detailed (Day 2)
  2. "tutoring_mode"      -> hint_first / direct_answer   (Day 3 - Socratic teaching)

Each "arm" = ஒரு option. ஒவ்வொரு feedback-ம் ஒரு "reward" (0.0 - 1.0).
Epsilon% நேரம் explore (random try), மீதி exploit (இதுவரைக்கும் அதிக reward தந்த arm).

State per (bandit_name, session_id) JSON file-ல் local-ஆ persist ஆகும்
(privacy-friendly - cloud-க்கு போகாது; production-ல் இதையே Redis/DB-க்கு swap பண்ணலாம்).
"""
import json
import os
import random
from app.config import BANDIT_STATE_PATH, BANDIT_EPSILON

STYLES = ["analogy", "summary", "detailed"]
TUTORING_MODES = ["hint_first", "direct_answer"]


def _load_state() -> dict:
    if not os.path.exists(BANDIT_STATE_PATH):
        return {}
    try:
        with open(BANDIT_STATE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _save_state(state: dict):
    with open(BANDIT_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _get_bucket(state: dict, bandit_name: str, session_id: str, arms: list[str]) -> dict:
    state.setdefault(bandit_name, {})
    state[bandit_name].setdefault(session_id, {})
    bucket = state[bandit_name][session_id]
    for arm in arms:
        bucket.setdefault(arm, {"count": 0, "total_reward": 0.0})
    return bucket


# ---------------------------------------------------------------------------
# Generic core (namespace-aware) - எந்த decision-க்கும் இதையே பயன்படுத்தலாம்
# ---------------------------------------------------------------------------

def select_arm(bandit_name: str, session_id: str, arms: list[str], epsilon: float = BANDIT_EPSILON) -> str:
    state = _load_state()
    bucket = _get_bucket(state, bandit_name, session_id, arms)

    untried = [a for a in arms if bucket[a]["count"] == 0]
    if untried:
        _save_state(state)
        return random.choice(untried)

    if random.random() < epsilon:
        _save_state(state)
        return random.choice(arms)

    avg_rewards = {a: bucket[a]["total_reward"] / bucket[a]["count"] for a in arms}
    _save_state(state)
    return max(avg_rewards, key=avg_rewards.get)


def update_arm_reward(bandit_name: str, session_id: str, arm: str, reward: float):
    state = _load_state()
    bucket = _get_bucket(state, bandit_name, session_id, [arm])
    bucket[arm]["count"] += 1
    bucket[arm]["total_reward"] += reward
    _save_state(state)


def get_arm_stats(bandit_name: str, session_id: str, arms: list[str]) -> dict:
    state = _load_state()
    bucket = _get_bucket(state, bandit_name, session_id, arms)
    result = {}
    for a in arms:
        count = bucket[a]["count"]
        total = bucket[a]["total_reward"]
        result[a] = {"count": count, "avg_reward": round(total / count, 3) if count > 0 else None}
    return result


def reset_bandit(bandit_name: str, session_id: str):
    state = _load_state()
    if bandit_name in state and session_id in state[bandit_name]:
        del state[bandit_name][session_id]
        _save_state(state)


# ---------------------------------------------------------------------------
# Explanation-style bandit (Day 2) - backward-compatible wrapper functions
# ---------------------------------------------------------------------------

def select_style(session_id: str, epsilon: float = BANDIT_EPSILON) -> str:
    return select_arm("explanation_style", session_id, STYLES, epsilon)


def update_reward(session_id: str, style: str, reward: float):
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style}")
    update_arm_reward("explanation_style", session_id, style, reward)


def get_stats(session_id: str) -> dict:
    return get_arm_stats("explanation_style", session_id, STYLES)


def reset_session(session_id: str):
    reset_bandit("explanation_style", session_id)


# ---------------------------------------------------------------------------
# Tutoring-mode bandit (Day 3) - "Explain instead of simply answering"
# hint_first: Socratic hint முதல்ல தரும் | direct_answer: நேரடி பதில் தரும்
# ---------------------------------------------------------------------------

def select_tutoring_mode(session_id: str, epsilon: float = BANDIT_EPSILON) -> str:
    return select_arm("tutoring_mode", session_id, TUTORING_MODES, epsilon)


def update_tutoring_reward(session_id: str, mode: str, reward: float):
    if mode not in TUTORING_MODES:
        raise ValueError(f"Unknown tutoring mode: {mode}")
    update_arm_reward("tutoring_mode", session_id, mode, reward)


def get_tutoring_stats(session_id: str) -> dict:
    return get_arm_stats("tutoring_mode", session_id, TUTORING_MODES)
