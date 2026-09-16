https://credit-card-retention-intelligence-q6qyhlvsztetblae9ttqf4.streamlit.app/ 
# Explainable Churn Early-Warning + GenAI Retention Playbook Generator

A credit-card customer retention system: predict -> explain -> recommend -> govern,
now with a real chat Playground, free-LLM backends, and a local database so the
model retrains on data you upload.

## Setup

```bash
cd src
pip install -r ../requirements.txt
```

### Pick an LLM backend (in `llm_playbook.py`, set `BACKEND = ...`)

- **"groq"** (recommended) -- free hosted API, no local process to keep running.
  1. Get a free key at https://console.groq.com -> API Keys
  2. Copy `.env.example` to `.env` (same `src/` folder) and paste your key in
  3. `pip install python-dotenv` (already in requirements.txt)
- **"ollama"** -- fully free and local.
  1. Install from https://ollama.com/download
  2. `ollama pull llama3.1:8b` (or `phi3` for a smaller model)
  3. Set `OLLAMA_MODEL` in `llm_playbook.py` to match
  4. Set `BACKEND = "ollama"`
- **"template"** -- no LLM, deterministic text. Works with nothing installed.

## Run it

```bash
streamlit run app.py
```

This opens the main dashboard. The sidebar (and a header button) link to a
**Playground** page for chatting with the model, uploading more customer
records, and uploading images.

## Pages

- **app.py** -- main dashboard: top at-risk customers, customer deep-dive,
  fairness/governance audit, what-if simulator.
- **pages/1_Playground.py** -- standalone chat page:
  - **Chat tab**: ask free-form questions; precise lookups (specific
    customer, exact counts) are answered deterministically, everything
    else goes to your configured LLM with a real data digest.
  - **Upload customer data tab**: upload a CSV of new customer records.
    They're validated, saved to a local SQLite database
    (`src/data/uploaded_customers.db`), and folded into the training
    data on the next retrain -- this is how the model "learns" from
    data you share, with zero cloud dependency.
  - **Upload an image tab**: preview an uploaded image inline.

## Why SQLite for the database

It's free, needs no signup, no server, no internet, and ships built into
Python. `src/db.py` is a thin wrapper around it -- if you later want a
hosted free database (e.g. Supabase's free Postgres tier) instead, only
`db.py` needs to change; nothing else in the app calls SQLite directly.

## Files

| File | What it does |
|---|---|
| `data_gen.py` | Synthetic customer generator |
| `churn_model.py` | GradientBoosting churn model + explainability |
| `vector_store.py` | TF-IDF retrieval over a retention playbook (RAG) |
| `llm_playbook.py` | Groq / Ollama / template LLM backends + prompt engineering |
| `qa_engine.py` | Routes questions to precise lookups or the LLM with data context |
| `fairness_check.py` | Four-fifths rule governance audit |
| `db.py` | SQLite persistence for uploaded customer data |
| `common.py` | Shared cached data/model loaders (used by all pages) |
| `app.py` | Main Streamlit dashboard |
| `pages/1_Playground.py` | Chat + upload page |
| `main.py` | Non-UI CLI pipeline runner (optional, for quick testing) |

## Security note

`.env` (your real API key) is in `.gitignore` and will never be committed.
Only `.env.example` (a placeholder) is tracked. Never paste a real key
directly into any `.py` file if this repo might go public.
