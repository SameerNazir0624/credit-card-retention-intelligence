"""
A small "taught facts" memory system. When the user says something like
"remember this" or "remember: X", we save it to a local SQLite database.
On future freeform questions, we retrieve any memories relevant to the
current question (via TF-IDF similarity, same technique as vector_store.py)
and inject them into the LLM's prompt -- so corrections/facts you've taught
it actually get used going forward, without retraining anything.

This is NOT the same as the LLM "learning" in the machine-learning sense --
the model's weights never change. It's closer to giving the model sticky
notes it re-reads on every relevant question.
"""
import os
import sqlite3
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "memories.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_memory(content: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO memories (content, created_at) VALUES (?, ?)",
        (content.strip(), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def load_all_memories():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, content, created_at FROM memories ORDER BY id DESC").fetchall()
    conn.close()
    return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]


def delete_memory(memory_id: int):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()


def search_memories(query: str, k: int = 3):
    memories = load_all_memories()
    if not memories:
        return []
    texts = [m["content"] for m in memories]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts)
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
    except ValueError:
        return memories[:k]
    ranked_idx = sims.argsort()[::-1][:k]
    return [memories[i] for i in ranked_idx if sims[i] > 0] or memories[:1]