"""
Turns (churn probability + top risk-driver reasons + retrieved playbook docs)
into a personalized, human-readable retention recommendation, and also
powers the freeform Q&A + vision fallback in qa_engine.py.

Backend is pluggable -- pick a DEFAULT below, but the Playground UI can
override backend+model per-request at runtime via AVAILABLE_MODELS.

  - "huggingface": Free hosted router (https://router.huggingface.co).
                    Text AND vision models available. Needs HF_TOKEN in .env.
  - "groq":         Free hosted API (https://console.groq.com). Text only.
  - "ollama":       Free, fully local -- needs Ollama installed and running.
  - "template":     No LLM at all -- deterministic template filling.

Setup for Hugging Face (recommended):
  1. pip install python-dotenv
  2. Create a .env file in this same folder (src/) containing:
         HF_TOKEN=hf_your_actual_token_here
     (free token at https://huggingface.co/settings/tokens)
  3. Add ".env" to your .gitignore so it never gets pushed to GitHub.
"""
import base64
import json
import os
import time
import urllib.request
import urllib.error

from dotenv import load_dotenv

load_dotenv()

BACKEND = "huggingface"  # default backend; UI can override per-request

BROWSER_HEADERS_EXTRA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# --- Hugging Face settings ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "deepseek-ai/DeepSeek-R1:fastest"
HF_VISION_MODEL = "zai-org/GLM-5.3-Flash:baseten"
HF_URL = "https://router.huggingface.co/v1/chat/completions"

# --- Groq settings ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Ollama settings ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

# Models the Playground UI lets the user pick between. "vision": True marks
# models that can accept an image in the same request.
AVAILABLE_MODELS = {
        "huggingface": [
        {"label": "DeepSeek R1 (text, fastest available)", "model": "deepseek-ai/DeepSeek-R1:fastest", "vision": False},
        {"label": "Qwen 2.5 7B Instruct (text, fastest available)", "model": "Qwen/Qwen2.5-7B-Instruct:fastest", "vision": False},
        {"label": "GLM-5.3 Flash (vision + text)", "model": "zai-org/GLM-5.3-Flash:baseten", "vision": True},
    ],
    "groq": [
        {"label": "Llama 3.3 70B Versatile (text)", "model": "llama-3.3-70b-versatile", "vision": False},
    ],
    "ollama": [
        {"label": "phi3 (local, small)", "model": "phi3", "vision": False},
        {"label": "llama3.1:8b (local, larger)", "model": "llama3.1:8b", "vision": False},
    ],
    "template": [
        {"label": "No LLM (deterministic template)", "model": None, "vision": False},
    ],
}

FEATURE_LABELS = {
    "tenure_months": "how long they've been a customer",
    "num_products": "number of products they hold with us",
    "avg_monthly_spend_now": "recent monthly spend level",
    "spend_decline_pct": "recent decline in spending",
    "missed_payments_12m": "missed payments in the last year",
    "complaints_12m": "complaints filed in the last year",
    "days_since_last_login": "days since they last used the app",
    "rewards_redeemed_12m": "rewards redeemed in the last year",
    "credit_utilization": "credit utilization level",
    "customer_service_calls_12m": "customer service calls in the last year",
    "age": "age",
}

PROMPT_TEMPLATE = """You are a retention strategy assistant for a credit card business.
A model has flagged a customer as at-risk of churn. Write a SHORT (3-4 sentence),
specific, actionable recommendation for the relationship manager. Ground your
recommendation in the retrieved playbook guidance below -- do not invent offers
that aren't supported by it. Be concrete about WHY (cite the top risk drivers)
and WHAT to do (cite the playbook).

Customer risk drivers (top signals from the model, most important first):
{reasons_block}

Relevant past-playbook guidance (retrieved by similarity):
{playbook_block}

Write the recommendation now:"""


def _format_reasons(reasons):
    lines = []
    for feat, contrib in reasons:
        label = FEATURE_LABELS.get(feat, feat)
        direction = "raising" if contrib > 0 else "lowering"
        lines.append(f"- {label} is {direction} their churn risk (impact {contrib:+.3f})")
    return "\n".join(lines)


def _format_playbook(retrieved):
    lines = []
    for doc, score in retrieved:
        lines.append(f"- [{doc['title']}] {doc['text']}")
    return "\n".join(lines)


def _call_huggingface(prompt: str, model: str = None, max_retries: int = 3) -> str:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a .env file in the src/ folder containing "
            "HF_TOKEN=hf_your_actual_token_here (free token at "
            "https://huggingface.co/settings/tokens)."
        )
    use_model = model or HF_MODEL
    payload = json.dumps({
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(
        HF_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
            **BROWSER_HEADERS_EXTRA,
        },
    )
    last_error = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            last_error = f"HTTP {e.code}: {body[:300]}"
            if e.code in (503, 429):
                time.sleep(5)
                continue
            raise RuntimeError(last_error)
        except Exception as e:
            last_error = str(e)
            time.sleep(3)
    raise RuntimeError(f"Hugging Face API failed after {max_retries} attempts: {last_error}")


def _call_huggingface_vision(prompt: str, image_bytes: bytes, mime_type: str, model: str = None) -> str:
    """Vision-capable call: sends the image alongside the text prompt using
    the OpenAI-compatible content-list format the HF router expects."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a .env file in the src/ folder containing "
            "HF_TOKEN=hf_your_actual_token_here."
        )
    use_model = model or HF_VISION_MODEL
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime_type};base64,{b64}"
    payload = json.dumps({
        "model": use_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": 0.3,
        "max_tokens": 500,
    }).encode()
    req = urllib.request.Request(
        HF_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
            **BROWSER_HEADERS_EXTRA,
        },
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            last_error = f"HTTP {e.code}: {body[:300]}"
            if e.code in (503, 429):
                time.sleep(5)
                continue
            raise RuntimeError(last_error)
        except Exception as e:
            last_error = str(e)
            time.sleep(3)
    raise RuntimeError(f"Hugging Face vision API failed: {last_error}")


def _call_groq(prompt: str, model: str = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env.")
    use_model = model or GROQ_MODEL
    payload = json.dumps({
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            **BROWSER_HEADERS_EXTRA,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str, model: str = None) -> str:
    use_model = model or OLLAMA_MODEL
    payload = json.dumps({"model": use_model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result.get("response", "").strip()


def _template_fallback(prob, reasons, retrieved):
    top_feat, top_contrib = reasons[0]
    top_label = FEATURE_LABELS.get(top_feat, top_feat)
    best_doc, best_score = retrieved[0]
    return (
        f"This customer has a {prob:.0%} predicted churn probability, driven "
        f"mainly by {top_label} (and secondarily by "
        f"{', '.join(FEATURE_LABELS.get(f, f) for f, _ in reasons[1:])}). "
        f"Based on the '{best_doc['title']}' playbook (similarity {best_score:.2f}), "
        f"recommended action: {best_doc['text'].split('. ')[1] if '. ' in best_doc['text'] else best_doc['text']} "
        f"Prioritize outreach within the next billing cycle given the risk level."
    )


def _call_llm(prompt: str, backend: str, model: str = None) -> str:
    if backend == "huggingface":
        return _call_huggingface(prompt, model=model)
    if backend == "groq":
        return _call_groq(prompt, model=model)
    if backend == "ollama":
        return _call_ollama(prompt, model=model)
    raise ValueError(f"No LLM call for backend={backend}")


def generate_recommendation(prob, reasons, retrieved, backend=None, model=None):
    backend = backend or BACKEND
    prompt = PROMPT_TEMPLATE.format(
        reasons_block=_format_reasons(reasons),
        playbook_block=_format_playbook(retrieved),
    )
    if backend in ("huggingface", "groq", "ollama"):
        try:
            return _call_llm(prompt, backend, model=model), prompt
        except Exception as e:
            return f"[{backend} unavailable ({e}), falling back to template]\n" + \
                   _template_fallback(prob, reasons, retrieved), prompt
    return _template_fallback(prob, reasons, retrieved), prompt


if __name__ == "__main__":
    reasons = [("customer_service_calls_12m", 0.14), ("complaints_12m", 0.09), ("num_products", -0.11)]
    from vector_store import PlaybookStore
    store = PlaybookStore()
    retrieved = store.retrieve("customer service calls complaints friction", k=2)
    rec, prompt = generate_recommendation(0.70, reasons, retrieved)
    print("--- PROMPT SENT ---")
    print(prompt)
    print("\n--- RECOMMENDATION ---")
    print(rec)