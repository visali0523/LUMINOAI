# LUMINOAI
# AI-Optimized Learning Engine — Day 1

**Goal:** Student upload PDF (textbook / syllabus), accurate  RAG (Retrieval-Augmented Generation) system.

**Stack:** FastAPI + ChromaDB (local vector DB) + sentence-transformers (local embeddings — privacy-friendly) + Groq API (Llama-3)

---

## 📁 Project Structure

```
ai-learning-engine/
├── app/
│   ├── __init__.py
│   ├── config.py          # settings, chunk size, top_k etc.
│   ├── pdf_processor.py   # PDF -> text -> chunks
│   ├── vector_store.py    # ChromaDB wrapper (add/query/list/delete)
│   ├── groq_client.py     # Groq Llama-3 call + style-based prompting
│   └── main.py            # FastAPI app & endpoints
├── demo/
│   └── sample_syllabus.pdf   # test  sample PDF
├── uploads/                  # uploaded PDFs 
├── chroma_db/                # ChromaDB local persistent storage
├── requirements.txt
├── .env.example
├── test_api.py             # end-to-end test script
└── README.md
```

---

## 🚀 Setup 

### 1. Virtual environment 
```bash
cd ai-learning-engine
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 2. Dependencies install 
```bash
pip install -r requirements.txt
```
> Note: `sentence-transformers` run , embedding model (`all-MiniLM-L6-v2`, ~90MB) HuggingFace-ல் ு auto-download . Internet  local- cache .

### 3. Groq API Key 
1. https://console.groq.com/keys 
2. Sign up  API key generate 
3. `.env.example`- `.env` - rename,  key:
```bash
cp .env.example .env
```
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
```

### 4. Server run
```bash
uvicorn app.main:app --reload
```
Server: `http://127.0.0.1:8000`
Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`

### 5. Test 
 terminal window-:
```bash
python test_api.py
```
 sample PDF upload, 3  styles-

---

## 🔌 API Endpoints

| Method | Endpoint | வேலை |
|--------|----------|------|
| GET | `/health` | Server running- check |
| POST | `/upload` | PDF upload  chunk+embed+store  |
| POST | `/ask` |  (question, doc_id, style) |
| GET | `/documents` | Upload  list |
| DELETE | `/documents/{doc_id}` |

### `/ask` request example:
```json
{
  "question": "What is photosynthesis?",
  "doc_id": "abc123_sample_syllabus",
  "style": "analogy"
}
```
`style` values: `"analogy"` , `"summary"` , `"detailed"`  `"default"`

> **Day 2-
---



- **Embeddings local- (`sentence-transformers`) generate  PDF content embedding API-ு.
- **ChromaDB local- disk- persist ஆகுது — cloud vector DB .
- **Answer strictly document context-ல்  — hallucination  syllabus-specific.
- Groq API-ஐ மட்டும் final answer generation (that too without storing student PDFs on their servers).

---

## ✅ Day 1 Status: 

- [x] PDF upload & text extraction (page-wise)
- [x] Chunking with overlap (context loss )
- [x] Local embeddings + ChromaDB storage
- [x] doc_id-based filtering 
- [x] Groq Llama-3 call, strict "context-only" system prompt
- [x] Style-adaptive prompting (analogy/summary/detailed) — manual trigger
- [x] Page number citation in answers
- [x] "Not in document" fallback (hallucination prevent)

## ➡️ Day 2 Preview
- Streamlit UI + 👍/👎 feedback buttons
- Simple multi-armed bandit (e.g., epsilon-greedy) to auto-pick best `style` per student based on feedback history
- 
