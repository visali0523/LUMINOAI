"""
ChromaDB வைத்து embeddings-ஐ store பண்ணி, search பண்ண உதவும் module.
Local embedding model (sentence-transformers) பயன்படுத்துறதால்
மாணவனோட data எதுவும் வெளியில (privacy) போகாது - இதுவே இந்த project-ஓட
முக்கிய selling point (Khanmigo மாதிரி cloud-only அல்ல).
"""
import chromadb
from chromadb.utils import embedding_functions
from app.config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

COLLECTION_NAME = "syllabus_documents"

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(records: list[dict]):
    """PDF chunks-ஐ ChromaDB-ல் insert பண்ணும்."""
    if not records:
        return

    ids = [r["id"] for r in records]
    documents = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]

    _collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def query_chunks(question: str, doc_id: str | None = None, top_k: int = TOP_K) -> list[dict]:
    """
    கேள்விக்கு பொருத்தமான chunks-ஐ retrieve பண்ணும்.
    doc_id கொடுத்தா, அந்த ஒரு document-ல் மட்டும் தேடும்
    (ஒரு மாணவன் upload பண்ண syllabus-ல் மட்டும் பதில் வர).
    """
    where_filter = {"doc_id": doc_id} if doc_id else None

    results = _collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where_filter,
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": text,
            "page": meta.get("page"),
            "doc_id": meta.get("doc_id"),
            "score": 1 - dist,  # cosine distance -> similarity
        })

    return chunks


def list_documents() -> list[str]:
    """இப்போ store-ல இருக்கிற எல்லா unique doc_ids-ஐயும் திருப்பும்."""
    all_data = _collection.get(include=["metadatas"])
    doc_ids = {m["doc_id"] for m in all_data.get("metadatas", []) if m}
    return sorted(doc_ids)


def delete_document(doc_id: str):
    """ஒரு document-ஓட எல்லா chunks-ஐயும் நீக்கும்."""
    _collection.delete(where={"doc_id": doc_id})
