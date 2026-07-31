"""
Adaptive AI Learning Assistant - Day 3 Frontend
Multi-Agent Tutor UI: Chat (Hybrid RAG + Socratic hints) + Progress Dashboard
(Knowledge Tracing) + Recommendations (Knowledge Graph) + Proactive Mentor + Goals

Run (FastAPI backend must already be running - `uvicorn app.main:app --reload`):
    streamlit run streamlit_app.py
"""

import uuid

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "https://app-spuf.onrender.com"

st.set_page_config(page_title="LUMINO AI", page_icon="", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "student_id" not in st.session_state:
    st.session_state.student_id = "student_1"
if "practice_quiz" not in st.session_state:
    st.session_state.practice_quiz = None  # {"concept": ..., "questions": [...]}


def api_get(path):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend. Check that `uvicorn app.main:app --reload` is running.\n\n{e}")
        return None


def api_post(path, json=None, files=None):
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=json, files=files, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {e}")
        return None


def api_delete(path):
    try:
        r = requests.delete(f"{API_BASE_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {e}")
        return None


STYLE_LABELS = {
    "auto": "🤖 Auto", "analogy": "🎯 Analogy", "summary": "📝 Summary",
    "detailed": "📖 Detailed", "default": "⚪ Default", "memory_summary": "🧠 Memory",
}
MODE_LABELS = {"auto": "🤖 Auto", "hint_first": "💡 Hint First (Socratic)", "direct_answer": "✅ Direct Answer"}
ROUTE_LABELS = {"rag": "📄 From your document", "general": "🌐 General knowledge", "memory": "🧠 Your learning history"}
TIER_COLOR = {"beginner": "🔴", "intermediate": "🟡", "advanced": "🟢"}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Adaptive AI Learning Assistant")
    st.caption("Hybrid RAG + Knowledge Tracing + RL + Multi-Agent Tutor")

    st.session_state.student_id = st.text_input(
        "🧑‍🎓 Student ID / Name", value=st.session_state.student_id,
        help="Each student's memory, mastery, and bandit history is tracked separately."
    )

    st.divider()
    st.subheader("📤 Upload Textbook / Syllabus")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Upload & Process", width='stretch'):
            with st.spinner("Processing PDF... (chunking + embedding)"):
                result = api_post("/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")})
            if result:
                st.success(f"✅ {result['chunks_created']} chunks created!")
                st.session_state.active_doc_id = result["doc_id"]

    st.divider()
    st.subheader("📄 Active Document")
    docs_data = api_get("/documents")
    doc_list = docs_data.get("documents", []) if docs_data else []
    if doc_list:
        options = ["(none - general knowledge only)"] + doc_list
        default_idx = options.index(st.session_state.active_doc_id) if st.session_state.active_doc_id in options else 0
        chosen = st.selectbox("Current documents", options, index=default_idx)
        st.session_state.active_doc_id = None if chosen.startswith("(none") else chosen
    else:
        st.info("No PDFs have been uploaded yet. You can still ask general knowledge questions without a document.")

    st.divider()
    st.subheader("🎛️ Overrides (optional)")
    style_choice = st.radio("Explanation Style", options=["auto", "analogy", "summary", "detailed"],
                             format_func=lambda x: STYLE_LABELS[x], index=0)
    mode_choice = st.radio("Tutoring Mode", options=["auto", "hint_first", "direct_answer"],
                            format_func=lambda x: MODE_LABELS[x], index=0)
    st.caption("🤖 When set to Auto, the RL bandit will automatically choose the best option based on your feedback.")

    st.divider()
    st.subheader("🧹 Reset / Testing Tools")
    reset_col1, reset_col2 = st.columns(2)
    with reset_col1:
        if st.button("🗑️ Clear Chat", width='stretch'):
            st.session_state.messages = []
            st.rerun()
    with reset_col2:
        if st.button("Reset Progress"):
    # API_URL சரியாக இருக்கிறதா என்று உறுதி செய்து கொள்ளவும் (பொதுவாக http://127.0.0.1:8000)
            res = requests.delete(f"http://127.0.0.1:8000/kt/mastery/{st.session_state.student_id}")
    
            if res.status_code == 200:
                st.success("✅ Progress Reset Successfully!")
                st.rerun() # Page-ஐ refresh செய்ய
            else:
                st.error(f"❌ Backend Error: {res.status_code} - {res.text}")

# ---------------------------------------------------------------------------
# Main area - Tabs
# ---------------------------------------------------------------------------
st.title("💬 Your AI Tutor")

tab_chat, tab_progress, tab_recommend, tab_mentor, tab_goals = st.tabs(
    ["💬 Tutor Chat", "📊 My Progress", "🎯 Recommendations", "🧑‍🏫 Mentor Check-in", "🥅 My Goals"]
)

# ---------------- Tab 1: Tutor Chat ----------------
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])

            badges = [ROUTE_LABELS.get(msg["route"], msg["route"])]
            if msg.get("concept"):
                badges.append(f"📌 {msg['concept']}")
            if msg.get("style_used"):
                badges.append(STYLE_LABELS.get(msg["style_used"], msg["style_used"]))
            if msg.get("tutoring_mode_used"):
                badges.append(MODE_LABELS.get(msg["tutoring_mode_used"], msg["tutoring_mode_used"]))
            if msg.get("difficulty"):
                badges.append(f"{TIER_COLOR.get(msg['difficulty'],'')} {msg['difficulty']}")
            if msg.get("sources"):
                badges.append(f"📄 pg {', '.join(map(str, msg['sources']))}")
            st.caption(" · ".join(badges))

            if msg["route"] != "memory":
                fb = msg["feedback"]
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                if c1.button("👍", key=f"up_{msg['id']}", disabled=fb.get("thumbs") is not None):
                    api_post("/agent/feedback", json={"session_id": st.session_state.student_id,
                              "style": msg["style_used"], "thumbs_reward": 1.0})
                    fb["thumbs"] = "up"; st.rerun()
                if c2.button("👎", key=f"down_{msg['id']}", disabled=fb.get("thumbs") is not None):
                    api_post("/agent/feedback", json={"session_id": st.session_state.student_id,
                              "style": msg["style_used"], "thumbs_reward": 0.0})
                    fb["thumbs"] = "down"; st.rerun()
                if c3.button("💪 Solved myself", key=f"solved_{msg['id']}", disabled=fb.get("tutoring") is not None):
                    api_post("/agent/feedback", json={"session_id": st.session_state.student_id,
                              "tutoring_mode": msg["tutoring_mode_used"], "tutoring_reward": 1.0})
                    fb["tutoring"] = "solved"; st.rerun()
                if c4.button("🙋 Just tell me", key=f"tellme_{msg['id']}", disabled=fb.get("tutoring") is not None):
                    api_post("/agent/feedback", json={"session_id": st.session_state.student_id,
                              "tutoring_mode": msg["tutoring_mode_used"], "tutoring_reward": 0.0})
                    fb["tutoring"] = "tellme"; st.rerun()
                if msg.get("concept"):
                    if c5.button("✅ Understood", key=f"got_{msg['id']}", disabled=fb.get("kt") is not None):
                        api_post("/agent/feedback", json={"session_id": st.session_state.student_id,
                                  "concept": msg["concept"], "correct": True})
                        fb["kt"] = "correct"; st.rerun()
                    if c6.button("❌ Didn't understand", key=f"notgot_{msg['id']}", disabled=fb.get("kt") is not None):
                        api_post("/agent/feedback", json={"session_id": st.session_state.student_id,
                                  "concept": msg["concept"], "correct": False})
                        fb["kt"] = "incorrect"; st.rerun()

                if any(fb.values()):
                    st.caption("✅ Feedback recorded — bandit/knowledge-tracing updated.")

    question = st.chat_input("Ask anything — it can be from the syllabus, or outside of it...")
    if question:
        with st.spinner("Router Agent -> RAG/LLM/Memory Agent -> Tutoring Engine thinking..."):
            result = api_post("/agent/ask", json={
                "question": question, "doc_id": st.session_state.active_doc_id,
                "session_id": st.session_state.student_id, "style": style_choice, "tutoring_mode": mode_choice,
            })
        if result:
            st.session_state.messages.append({
                "id": str(uuid.uuid4()), "question": question, "answer": result["answer"],
                "sources": result.get("sources", []), "route": result.get("route"),
                "concept": result.get("concept"), "style_used": result.get("style_used"),
                "tutoring_mode_used": result.get("tutoring_mode_used"), "difficulty": result.get("difficulty"),
                "feedback": {},
            })
            st.rerun()

# ---------------- Tab 2: My Progress ----------------
with tab_progress:
    st.subheader("📊 Concept Mastery (Knowledge Tracing - Bayesian)")
    mastery_data = api_get(f"/kt/mastery/{st.session_state.student_id}")
    mastery = mastery_data.get("mastery", {}) if mastery_data else {}
    if not mastery:
        st.info("No concepts attempted yet. Ask a question in Tutor Chat and give '✅ Understood' feedback.")
    else:
        for concept, data in sorted(mastery.items(), key=lambda x: -x[1]["mastery"]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(data["mastery"], text=f"{TIER_COLOR.get(data['tier'],'')} **{concept}** — {data['mastery']*100:.0f}% ({data['tier']})")
            with col2:
                st.caption(f"{data['attempts']} attempts, {data['correct_count']} correct")

# ---------------- Tab 3: Recommendations ----------------
with tab_recommend:
    st.subheader("🎯 Recommendation Agent")
    rec_data = api_get(f"/recommend/{st.session_state.student_id}")
    if rec_data:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔁 Revision Topics (weak mastery)")
            if rec_data["revision_topics"]:
                for t in rec_data["revision_topics"]:
                    st.warning(f"**{t['concept']}** — {t['mastery']*100:.0f}% mastery ({t['attempts']} attempts)")
                    if st.button(f"🎯 Practice '{t['concept']}'", key=f"practice_{t['concept']}"):
                        with st.spinner("Generating practice questions..."):
                            quiz = api_post("/practice/generate", json={
                                "student_id": st.session_state.student_id,
                                "concept": t["concept"],
                                "num_questions": 3,
                            })
                        if quiz:
                            st.session_state.practice_quiz = quiz
                            st.rerun()
            else:
                st.caption("No weak topics need revision right now 👍")
        with col2:
            st.markdown("#### ➡️ Ready to Learn Next")
            if rec_data["next_topics"]:
                for t in rec_data["next_topics"]:
                    st.success(f"**{t['concept']}** — prerequisites {t['readiness']*100:.0f}% ready")
            else:
                st.caption("Next topics aren't known until prerequisites are mastered.")

    # ---- Practice quiz (MCQ) UI ----
    if st.session_state.practice_quiz:
        st.divider()
        quiz = st.session_state.practice_quiz
        st.markdown(f"#### 📝 Practice Quiz: {quiz['concept']}")

        answers = {}
        with st.form("practice_quiz_form"):
            for i, q in enumerate(quiz["questions"]):
                answers[i] = st.radio(q["question"], options=q["options"], key=f"quiz_q_{i}", index=None)
            submitted = st.form_submit_button("✅ Submit Answers")

        if submitted:
            correct_count = 0
            for i, q in enumerate(quiz["questions"]):
                chosen = answers.get(i)
                is_correct = chosen is not None and q["options"].index(chosen) == q["correct_index"]
                if is_correct:
                    correct_count += 1
                api_post("/agent/feedback", json={
                    "session_id": st.session_state.student_id,
                    "concept": quiz["concept"],
                    "correct": is_correct,
                })
            st.success(f"Score: {correct_count}/{len(quiz['questions'])} — mastery updated!")
            if st.button("Close quiz"):
                st.session_state.practice_quiz = None
                st.rerun()

# ---------------- Tab 4: Proactive Mentor ----------------
with tab_mentor:
    st.subheader("🧑‍🏫 Proactive Mentor Agent")
    st.caption("Without the student asking, the AI mentor checks in on its own based on mastery patterns.")
    if st.button("🔔 Check my progress now"):
        mentor_data = api_get(f"/mentor/check/{st.session_state.student_id}")
        messages = mentor_data.get("messages", []) if mentor_data else []
        if not messages:
            st.success("No special alerts right now. Keep going! 🚀")
        for m in messages:
            if m["type"] == "struggle":
                st.error(f"⚠️ {m['message']}")
            elif m["type"] == "ready":
                st.success(f"🎉 {m['message']}")
            elif m["type"] == "progress":
                st.info(f"📈 {m['message']}")
            elif m["type"] == "stale":
                st.info(f"⏰ {m['message']}")

    st.divider()
    st.markdown("#### 💡 Mini Project Ideas")
    st.caption("Based on the concepts you've mastered, your mentor suggests small projects to build hands-on skill.")
    if st.button("💡 Get Project Ideas"):
        with st.spinner("Your mentor is thinking of good project ideas..."):
            idea_data = api_get(f"/mentor/project-ideas/{st.session_state.student_id}")
        if idea_data:
            st.info(idea_data.get("ideas", f"Error: {idea_data}"))

# ---------------- Tab 5: My Goals ----------------
with tab_goals:
    st.subheader("🥅 My Learning Goals")
    goal_text = st.text_input("Add a new goal", placeholder="e.g. Master dynamic programming by next week")
    if st.button("➕ Add Goal") and goal_text.strip():
        api_post("/memory/goal", json={"session_id": st.session_state.student_id, "goal_text": goal_text})
        st.rerun()

    goals_data = api_get(f"/memory/goals/{st.session_state.student_id}")
    goals = goals_data.get("goals", []) if goals_data else []
    if goals:
        for g in goals:
            status = "✅" if g["achieved"] else "⏳"
            st.write(f"{status} {g['goal_text']}  \n*({g['created_at'][:10]})*")
    else:
        st.caption("No goals set yet.")
