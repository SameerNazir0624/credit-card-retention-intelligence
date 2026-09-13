"""
A tiny 'vector store' of past retention campaigns/policies, retrieved by
similarity to a customer's churn-driver profile.

We use TF-IDF + cosine similarity here instead of sentence-transformer
embeddings + FAISS, purely because this environment has no internet access
to download a model or install faiss. The interface (`retrieve(query, k)`)
is identical to what you'd get from a real vector DB, so swapping in
FAISS/Chroma + sentence-transformers later is a drop-in replacement --
see the commented alternative at the bottom of this file.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PLAYBOOK = [
    {
        "id": "PB001",
        "title": "Spend-decline win-back",
        "text": ("Customer showing declining monthly spend after a period of "
                  "high engagement. Successful play: targeted category bonus "
                  "(e.g. 5x points on travel or dining for 60 days) timed to "
                  "their historical peak spend category. Worked well for "
                  "customers with tenure over 12 months."),
    },
    {
        "id": "PB002",
        "title": "Service-friction recovery",
        "text": ("Customer with multiple recent customer service calls or "
                  "complaints. Successful play: proactive outreach from a "
                  "retention specialist (not a bot), fee waiver on next "
                  "statement, and a direct callback within 48 hours. "
                  "Avoid generic discount offers here -- the friction is "
                  "about experience, not price."),
    },
    {
        "id": "PB003",
        "title": "Missed-payment risk stabilization",
        "text": ("Customer with missed payments in the last 12 months. "
                  "Successful play: flexible payment plan offer, temporary "
                  "APR reduction, and financial wellness check-in rather than "
                  "a rewards-based offer, which has historically not moved "
                  "this segment."),
    },
    {
        "id": "PB004",
        "title": "Low-engagement reactivation",
        "text": ("Customer with long gaps since last login/app usage and low "
                  "rewards redemption. Successful play: simplified digital "
                  "onboarding nudge, reminder of unused rewards balance, and "
                  "a low-friction one-tap redemption offer."),
    },
    {
        "id": "PB005",
        "title": "Single-product deepening",
        "text": ("Customer holding only one product with the bank. Successful "
                  "play: cross-sell a complementary product (e.g. savings or "
                  "a second card tier) bundled with a fee waiver, which "
                  "correlates strongly with reduced attrition in this "
                  "dataset."),
    },
    {
        "id": "PB006",
        "title": "Long-tenure loyalty risk",
        "text": ("Long-tenured customer showing early risk signals despite "
                  "loyalty. Successful play: loyalty-tier upgrade or "
                  "anniversary recognition offer -- these customers respond "
                  "better to recognition than to transactional discounts."),
    },
]


class PlaybookStore:
    def __init__(self, docs=PLAYBOOK):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in docs])

    def retrieve(self, query: str, k: int = 2):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in top_idx]


# --- Drop-in upgrade path (needs internet + `pip install faiss-cpu
#     sentence-transformers`) ---
# from sentence_transformers import SentenceTransformer
# import faiss
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode([d["text"] for d in PLAYBOOK])
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)
# def retrieve(query, k=2):
#     q_emb = model.encode([query])
#     distances, idx = index.search(q_emb, k)
#     return [PLAYBOOK[i] for i in idx[0]]

if __name__ == "__main__":
    store = PlaybookStore()
    results = store.retrieve("customer service calls complaints friction", k=2)
    for doc, score in results:
        print(f"[{score:.2f}] {doc['title']}")
