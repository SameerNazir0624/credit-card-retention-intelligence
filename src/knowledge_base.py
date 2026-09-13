"""
A persistent, company-wide knowledge base -- unlike doc_store.py (which is
session-only and resets on every browser refresh), everything added here
lives in a local SQLite database and accumulates over time. New data
always ADDS to it, never overwrites what's already there, so it keeps
growing as more is uploaded over time.

Two kinds of content:
  - "document": unstructured text extracted from PDF/DOCX/TXT/MD, chunked.
  - "tabular":  structured rows from any CSV/Excel file with ANY schema
                (not just the churn dataset's fixed columns) -- each row
                (or a small batch of rows) is serialized to text so it's
                searchable the same way documents are.

Retrieval uses the same free TF-IDF technique as vector_store.py and
memory_store.py -- no embeddings model, no internet, no paid service --
rebuilt fresh on each search call, which is fine at the (thousands-of-
chunks) scale this is designed for.
"""
import io
import os
import sqlite3
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.db")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
ROWS_PER_TABULAR_CHUNK = 8  # how many spreadsheet rows get grouped into one searchable chunk


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            batch_label TEXT,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---------- text extraction (documents) ----------

def _extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == "docx":
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: .{ext}")


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def add_document(file_bytes: bytes, filename: str, batch_label: str = None) -> int:
    """Extracts and persists text from a PDF/DOCX/TXT/MD file. Returns chunk count."""
    text = _extract_text(file_bytes, filename)
    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("No extractable text found in this file.")
    _save_chunks(filename, "document", batch_label, chunks)
    return len(chunks)


# ---------- tabular ingestion (any CSV/Excel, any schema) ----------

def read_tabular(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError(f"Unsupported tabular type: .{ext}")


def _row_to_text(row: pd.Series) -> str:
    return "; ".join(f"{col}={row[col]}" for col in row.index)


def add_tabular(df: pd.DataFrame, filename: str, batch_label: str = None,
                rows_per_chunk: int = ROWS_PER_TABULAR_CHUNK) -> int:
    """Serializes an arbitrary-schema dataframe into searchable text chunks
    (grouped rows), independent of the fixed churn-model schema. Returns
    chunk count."""
    if df.empty:
        raise ValueError("This file has no rows.")
    row_texts = [_row_to_text(row) for _, row in df.iterrows()]
    chunks = []
    for i in range(0, len(row_texts), rows_per_chunk):
        group = row_texts[i:i + rows_per_chunk]
        chunks.append(f"Columns: {', '.join(df.columns.astype(str))}\n" + "\n".join(group))
    _save_chunks(filename, "tabular", batch_label, chunks)
    return len(chunks)


# ---------- storage / retrieval ----------

def _save_chunks(source: str, kind: str, batch_label: str, chunks: list):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    now = datetime.utcnow().isoformat()
    for i, chunk in enumerate(chunks):
        conn.execute(
            "INSERT INTO kb_chunks (source, kind, batch_label, chunk_index, content, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source, kind, batch_label, i, chunk, now),
        )
    conn.commit()
    conn.close()


def list_sources():
    """Returns one row per uploaded file: source, kind, chunk count, batch label, latest upload time."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT source, kind, batch_label, COUNT(*), MAX(uploaded_at)
        FROM kb_chunks GROUP BY source, kind, batch_label
        ORDER BY MAX(uploaded_at) DESC
    """).fetchall()
    conn.close()
    return [
        {"source": r[0], "kind": r[1], "batch_label": r[2], "chunks": r[3], "uploaded_at": r[4]}
        for r in rows
    ]


def delete_source(source: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM kb_chunks WHERE source = ?", (source,))
    conn.commit()
    conn.close()


def total_chunks() -> int:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
    conn.close()
    return n


def search(query: str, k: int = 4):
    """TF-IDF similarity search across every chunk ever added -- documents
    and tabular data together, across every batch/upload ever made."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT source, kind, content FROM kb_chunks").fetchall()
    conn.close()
    if not rows:
        return []
    texts = [r[2] for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts)
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
    except ValueError:
        return [{"source": r[0], "kind": r[1], "content": r[2]} for r in rows[:k]]
    ranked_idx = sims.argsort()[::-1][:k]
    return [
        {"source": rows[i][0], "kind": rows[i][1], "content": rows[i][2]}
        for i in ranked_idx if sims[i] > 0
    ]