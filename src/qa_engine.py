"""
A Q&A engine over the customer dataset, uploaded documents, taught
memories, and (when an image is attached) a vision model.

Routing order for a given message:
  1. An image is attached -> vision call, bypassing everything else.
  2. "Remember this" / "Remember: X" -> saves to memory_store.py (SQLite).
  3. Precise intent matching (regex) for exact lookups -- specific
     customer, exact counts, exact group rates. Deterministic, always
     correct, no LLM needed.
  4. Everything else -> the configured LLM with a data digest, relevant
     taught memories, AND relevant entries from the persistent knowledge
     base (any CSV/Excel/PDF/Word a company has ever uploaded), all
     stuffed into the prompt.
"""
import re
import pandas as pd

from churn_model import explain_customer, ALL_FEATURES
from llm_playbook import FEATURE_LABELS, generate_recommendation, BACKEND, _call_llm, _call_huggingface_vision
from fairness_check import run_full_audit
import memory_store
import knowledge_base


CUSTOMER_ID_RE = re.compile(r"\b([Cc]\d{5,6})\b")
MEMORY_TRIGGER_RE = re.compile(
    r"\b(remember|save this|note this|keep in mind|don'?t forget)\b", re.I
)


def _find_customer(df, question):
    m = CUSTOMER_ID_RE.search(question)
    if not m:
        return None
    cid = m.group(1).upper()
    match = df[df["customer_id"].str.upper() == cid]
    if match.empty:
        return None
    return match.iloc[0]


def _why_at_risk(df, pipe, store, row):
    prob, reasons = explain_customer(pipe, row, df)
    query = " ".join(FEATURE_LABELS.get(f, f) for f, c in reasons if c > 0) or \
            " ".join(FEATURE_LABELS.get(f, f) for f, _ in reasons)
    retrieved = store.retrieve(query, k=2)
    reason_text = "; ".join(
        f"{FEATURE_LABELS.get(f, f)} ({'raises' if c > 0 else 'lowers'} risk, impact {c:+.3f})"
        for f, c in reasons
    )
    return (
        f"Customer {row['customer_id']} has a {prob:.1%} predicted churn probability. "
        f"Main drivers: {reason_text}."
    ), prob, reasons, retrieved


def _recommend(df, pipe, store, row, backend=None, model=None):
    text, prob, reasons, retrieved = _why_at_risk(df, pipe, store, row)
    rec, _ = generate_recommendation(prob, reasons, retrieved, backend=backend, model=model)
    return f"{text}\n\n**Recommendation:** {rec}"


def _count_flagged(df):
    n = int(df["flagged"].sum())
    pct = df["flagged"].mean()
    return f"{n:,} customers are currently flagged as at-risk ({pct:.1%} of the base)."


def _group_rate(df, group_col, group_val, metric_col="churned"):
    subset = df[df[group_col].astype(str).str.lower() == group_val.lower()]
    if subset.empty:
        return None
    rate = subset[metric_col].mean()
    label = "churn rate" if metric_col == "churned" else "flag rate"
    return f"{label.capitalize()} for {group_col}='{group_val}': {rate:.1%} (n={len(subset)})."


def _top_drivers_overall(df, pipe, sample_n=200):
    sample = df[df["flagged"] == 1].sample(min(sample_n, df["flagged"].sum()), random_state=0) \
        if df["flagged"].sum() > 0 else df.sample(min(sample_n, len(df)), random_state=0)
    totals, counts = {}, {}
    for _, row in sample.iterrows():
        _, reasons = explain_customer(pipe, row, df, top_k=len(ALL_FEATURES) - 2)
        for feat, contrib in reasons:
            totals[feat] = totals.get(feat, 0) + contrib
            counts[feat] = counts.get(feat, 0) + 1
    avg = {f: totals[f] / counts[f] for f in totals}
    ranked = sorted(avg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    return ranked


def _build_data_context(df, pipe, sample_size=15):
    lines = []
    lines.append(f"Total customers: {len(df)}")
    lines.append(f"Historical churn rate: {df['churned'].mean():.1%}")
    lines.append(f"Currently flagged at-risk: {df['flagged'].mean():.1%}")
    lines.append(f"Average predicted churn probability: {df['churn_prob'].mean():.1%}")

    lines.append("\nAverages across all customers:")
    for feat in ["tenure_months", "num_products", "avg_monthly_spend_now",
                 "missed_payments_12m", "complaints_12m", "days_since_last_login",
                 "rewards_redeemed_12m", "credit_utilization", "customer_service_calls_12m"]:
        lines.append(f"- {FEATURE_LABELS.get(feat, feat)}: {df[feat].mean():.2f}")

    lines.append("\nTop overall churn drivers (avg impact across at-risk customers):")
    for feat, impact in _top_drivers_overall(df, pipe, sample_n=100):
        lines.append(f"- {FEATURE_LABELS.get(feat, feat)}: {impact:+.3f}")

    lines.append("\nFlag rate by income band:")
    for band, rate in df.groupby("income_band")["flagged"].mean().items():
        lines.append(f"- {band}: {rate:.1%}")

    lines.append("\nFlag rate by region:")
    for region, rate in df.groupby("region")["flagged"].mean().items():
        lines.append(f"- {region}: {rate:.1%}")

    audit = run_full_audit(df)
    for group_col, result in audit.items():
        if result["groups_needing_review"]:
            lines.append(f"\nFairness note: groups in '{group_col}' below the 80% "
                         f"parity threshold: {list(result['groups_needing_review'].keys())}")

    lines.append(f"\nSample of top {sample_size} at-risk customers "
                  f"(customer_id, churn_prob, tenure_months, missed_payments_12m, "
                  f"complaints_12m, customer_service_calls_12m, num_products):")
    top_risk = df.sort_values("churn_prob", ascending=False).head(sample_size)
    for _, r in top_risk.iterrows():
        lines.append(
            f"- {r['customer_id']}: {r['churn_prob']:.1%}, {r['tenure_months']}mo, "
            f"{r['missed_payments_12m']} missed pmts, {r['complaints_12m']} complaints, "
            f"{r['customer_service_calls_12m']} service calls, {r['num_products']} products"
        )

    return "\n".join(lines)


def _save_memory_from_request(question: str, last_exchange):
    m = re.search(
        r"(?:remember|save this|note this|keep in mind|don'?t forget)"
        r"\s*(?:that|this is|:)?\s*(.+)",
        question, re.I,
    )
    explicit_text = m.group(1).strip() if m else ""

    generic_refs = {"this", "that", "this.", "that.", ""}
    if explicit_text.lower().strip(".") in generic_refs:
        if last_exchange and last_exchange[1]:
            prev_q, prev_a = last_exchange
            text_to_save = f"Q: {prev_q}\nA: {prev_a}"
        else:
            return ("There's nothing from earlier in this conversation to remember yet -- "
                    "ask something first, then say 'remember this'.")
    else:
        text_to_save = explicit_text

    memory_store.save_memory(text_to_save)
    return f"Got it, I'll remember that:\n\n> {text_to_save}"


def _llm_freeform(question, df, pipe, doc_store=None, backend=None, model=None):
    backend = backend or BACKEND
    if backend not in ("huggingface", "groq", "ollama"):
        return (
            "This needs a real LLM to answer freely, but none is configured yet.\n\n"
            "Easiest free option: get a token at https://huggingface.co/settings/tokens, "
            "paste it into a .env file as HF_TOKEN=..., and set `BACKEND = \"huggingface\"` "
            "in llm_playbook.py.\n\n"
            "In the meantime, try one of these known question types:\n"
            "- \"Why is customer C102629 at risk?\"\n"
            "- \"What should we do for customer C101541?\"\n"
            "- \"How many customers are flagged?\"\n"
            "- \"What's the churn rate for region west?\"\n"
            "- \"What are the top drivers of churn?\""
        )

    context = _build_data_context(df, pipe)

    relevant_memories = memory_store.search_memories(question, k=3)
    memory_block = ""
    if relevant_memories:
        memory_block = (
            "\n\n=== THINGS YOU'VE BEEN TAUGHT TO REMEMBER "
            "(treat as ground truth, prioritize over general reasoning) ===\n"
            + "\n".join(f"- {m['content']}" for m in relevant_memories)
            + "\n=== END TAUGHT FACTS ==="
        )

    doc_block = ""
    if doc_store is not None and not doc_store.is_empty():
        chunks = doc_store.retrieve(question, k=3)
        if chunks:
            doc_block = (
                "\n\n=== RELEVANT EXCERPTS FROM UPLOADED DOCUMENTS (this session) ===\n"
                + "\n".join(f"- [{c['source']}] {c['text']}" for c in chunks)
                + "\n=== END DOCUMENT EXCERPTS ==="
            )

    kb_block = ""
    kb_hits = knowledge_base.search(question, k=4)
    if kb_hits:
        kb_block = (
            "\n\n=== RELEVANT ENTRIES FROM THE PERSISTENT KNOWLEDGE BASE "
            "(accumulated company data across all uploads to date) ===\n"
            + "\n".join(f"- [{h['source']} / {h['kind']}] {h['content']}" for h in kb_hits)
            + "\n=== END KNOWLEDGE BASE ENTRIES ==="
        )

    prompt = (
        "You are a data analyst assistant answering questions about a credit-card "
        "customer churn/retention dataset, plus a persistent company knowledge base "
        "that may contain other documents and datasets uploaded over time. Use ONLY "
        "the information below to answer -- if it doesn't cover the question, say so "
        "plainly instead of guessing. Be concise (3-6 sentences unless the question "
        "asks for a list).\n\n"
        f"=== DATASET SUMMARY ===\n{context}\n=== END DATASET SUMMARY ==="
        f"{memory_block}{doc_block}{kb_block}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        return _call_llm(prompt, backend, model=model)
    except Exception as e:
        hint = {
            "huggingface": "Check that HF_TOKEN is set correctly in .env, and that you "
                           "have internet access. Free-tier models can take ~20s to 'wake up' "
                           "on first use -- try again if it just timed out.",
            "groq": "Check that your GROQ_API_KEY is set correctly in .env.",
            "ollama": "Make sure Ollama is running (`ollama serve` or `ollama run phi3`).",
        }.get(backend, "")
        return f"Couldn't reach the {backend} LLM ({e}). {hint}"


def answer_image(question: str, image_bytes: bytes, mime_type: str, model: str = None):
    """Routes a message with an attached image to a vision-capable model.
    Only Hugging Face is wired for vision currently -- Groq/Ollama text
    models can't see images."""
    prompt = question.strip() or "Describe what's in this image and note anything relevant to a credit-card retention/customer-service context."
    try:
        return _call_huggingface_vision(prompt, image_bytes, mime_type, model=model)
    except Exception as e:
        return (
            f"Couldn't reach the vision model ({e}). Make sure HF_TOKEN is set in .env "
            f"and you have internet access -- vision requires the Hugging Face backend "
            f"specifically (Groq/Ollama text models can't process images)."
        )


def answer_question(question: str, df: pd.DataFrame, pipe, store, last_exchange=None,
                     doc_store=None, backend=None, model=None):
    """last_exchange, if provided, should be a (previous_question, previous_answer)
    tuple -- used so 'remember this' can save the prior answer.
    backend/model, if provided, override the default from llm_playbook.py
    for this call only (used by the Playground's model picker)."""
    q = question.strip().lower()

    if MEMORY_TRIGGER_RE.search(question):
        return _save_memory_from_request(question, last_exchange)

    row = _find_customer(df, question)

    if row is not None and any(w in q for w in ["why", "risk", "reason", "explain"]):
        text, *_ = _why_at_risk(df, pipe, store, row)
        return text

    if row is not None and any(w in q for w in ["recommend", "should", "do about", "action", "playbook"]):
        return _recommend(df, pipe, store, row, backend=backend, model=model)

    if row is not None:
        text, *_ = _why_at_risk(df, pipe, store, row)
        return text

    if any(w in q for w in ["how many", "count", "number of"]) and "flagged" in q:
        return _count_flagged(df)

    m = re.search(r"(?:for|in)\s+(region|income_band|income band)\s*[:=]?\s*(\w+)", q)
    if m:
        col = "region" if "region" in m.group(1) else "income_band"
        val = m.group(2)
        metric = "flagged" if "flag" in q else "churned"
        result = _group_rate(df, col, val, metric)
        if result:
            return result

    return _llm_freeform(question, df, pipe, doc_store=doc_store, backend=backend, model=model)