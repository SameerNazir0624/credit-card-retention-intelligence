"""
Lets users upload arbitrary documents (PDF, DOCX, TXT) instead of only
structured CSVs. Text is extracted, split into chunks, and made
searchable via the same free TF-IDF technique as vector_store.py -- no
embeddings model or internet needed. Retrieved chunks get stuffed into
the LLM prompt so the chat can answer questions grounded in whatever
was uploaded.

Chunks live in Streamlit session state (per-session, not persisted to
disk) -- documents are usually one-off reference material for a
conversation, unlike customer records which need to persist and retrain
the model.
"""
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return _extract_pdf(file_bytes)
    if ext == "docx":
        return _extract_docx(file_bytes)
    if ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: .{ext}")


def _extract_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(file_bytes: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class DocumentStore:
    """In-memory (per Streamlit session) store of uploaded document chunks."""

    def __init__(self):
        self.chunks = []
        self._vectorizer = None
        self._matrix = None

    def add_document(self, file_bytes: bytes, filename: str) -> int:
        text = extract_text(file_bytes, filename)
        new_chunks = chunk_text(text)
        for c in new_chunks:
            self.chunks.append({"source": filename, "text": c})
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self):
        if not self.chunks:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int = 3):
        if not self.chunks or self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in ranked_idx if sims[i] > 0]

    def list_sources(self):
        seen = []
        for c in self.chunks:
            if c["source"] not in seen:
                seen.append(c["source"])
        return seen

    def is_empty(self):
        return len(self.chunks) == 0