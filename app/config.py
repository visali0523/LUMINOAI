import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# Chunking settings - PDF-ஐ இந்த அளவுக்கு துண்டு துண்டா பிரிக்கும்
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # chunks-க்கு இடையே overlap (context இழக்காம இருக்க)

# Retrieval settings
TOP_K = 4  # ஒரு question-க்கு எத்தனை chunks retrieve பண்ணனும்

# Embedding model - local-ஆ run ஆகும், API தேவையில்ல (privacy-க்கு நல்லது)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Bandit (RL) settings - style adaptation-க்கு
BANDIT_STATE_PATH = os.getenv("BANDIT_STATE_PATH", "./bandit_state.json")
BANDIT_EPSILON = float(os.getenv("BANDIT_EPSILON", "0.2"))  # 20% explore, 80% exploit

# ============ Day 3: Multi-Agent / Knowledge Tracing / KG settings ============

# SQLite - user profiles, mastery history, interactions, feedback
# (production-ல் PostgreSQL-க்கு swap பண்ணலாம், அதே schema logic வேலை செய்யும்)
DB_PATH = os.getenv("DB_PATH", "./learning_engine.db")

# Knowledge Graph - networkx graph, JSON-ஆ persist ஆகும்
# (production-ல் Neo4j-க்கு swap பண்ணலாம்)
KG_PATH = os.getenv("KG_PATH", "./knowledge_graph.json")

# Bayesian Knowledge Tracing (BKT) default parameters
# ஒவ்வொரு concept-க்கும் இதே defaults-ஆ start ஆகும், per-concept override பண்ணலாம்
BKT_DEFAULTS = {
    "p_init": 0.3,      # ஆரம்பத்தில் concept தெரிஞ்சிருக்கும் probability
    "p_transit": 0.15,  # ஒரு attempt-க்கு அப்புறம் தெரியாததில் இருந்து தெரிஞ்சதுக்கு போகும் probability
    "p_slip": 0.1,       # தெரிஞ்சும் தப்பா சொல்லும் probability
    "p_guess": 0.2,       # தெரியாமலேயே சரியா guess பண்ணும் probability
}
MASTERY_THRESHOLD = 0.85   # இந்த P(know) மேல போனா "mastered"-ஆ கருதப்படும்
WEAK_THRESHOLD = 0.4       # இதுக்கு கீழ போனா "weak concept"

# RAG relevance threshold - இந்த score-க்கு கீழ போனா, RAG-ல் "நல்ல match இல்ல"-ன்னு
# Router Agent முடிவு பண்ணி LLM Agent (general knowledge)-க்கு fallback பண்ணும்
RAG_RELEVANCE_THRESHOLD = 0.35

# Proactive Mentor settings
STRUGGLE_ATTEMPT_THRESHOLD = 3     # ஒரு concept-ல இத்தனை தடவை தப்பு பண்ணா "struggling"
STALE_REVISION_DAYS = 14           # இத்தனை நாள் revise பண்ணலனா "revision reminder"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
