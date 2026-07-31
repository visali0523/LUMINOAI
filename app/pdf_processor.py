"""
PDF-ஐ படிச்சு, text extract பண்ணி, சிறு chunks-ஆ பிரிக்கும் module.
"""
from pypdf import PdfReader
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """
    PDF-ல் இருந்து page-வாரியா text எடுக்கும்.
    Return: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    ஒரு பெரிய text-ஐ chunk_size அளவுள்ள சிறு chunks-ஆ பிரிக்கும்.
    Overlap வச்சிருக்கிறதால, chunk boundary-ல் context இழக்காது.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # வார்த்தை நடுவுல முறிஞ்சு போகாம, கடைசி space வரைக்கும் trim பண்ணுவோம்
        if end < text_len:
            last_space = chunk.rfind(" ")
            if last_space != -1 and last_space > chunk_size * 0.5:
                chunk = chunk[:last_space]
                end = start + last_space

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start <= 0:
            start = end

    return chunks


def process_pdf(pdf_path: str, doc_id: str) -> list[dict]:
    """
    PDF முழுசா process பண்ணி, ChromaDB-ல் store பண்ண தயார் ஆன
    chunk records-ஐ திருப்பி தரும்.

    Return: [
        {
            "id": "docid_p1_c0",
            "text": "...",
            "metadata": {"doc_id": ..., "page": 1, "chunk_index": 0}
        },
        ...
    ]
    """
    pages = extract_text_by_page(pdf_path)
    records = []

    for page in pages:
        page_chunks = chunk_text(page["text"])
        for idx, chunk in enumerate(page_chunks):
            record_id = f"{doc_id}_p{page['page']}_c{idx}"
            records.append({
                "id": record_id,
                "text": chunk,
                "metadata": {
                    "doc_id": doc_id,
                    "page": page["page"],
                    "chunk_index": idx,
                }
            })

    return records
