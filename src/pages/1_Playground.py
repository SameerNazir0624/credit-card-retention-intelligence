"""
One unified chat page. A "📎 Attach a file" control next to the chat input
accepts CSV, Excel, PDF, Word, text, or images and routes each intelligently:

  - .csv matching the churn dataset's schema -> validated, saved to
    SQLite, folds into the next model retrain (see db.py / common.py).
    This is specific to THIS app's built-in churn model.

  - .csv / .xlsx / .xls with ANY OTHER schema -> added to the persistent
    knowledge base (see knowledge_base.py). This is the general-purpose
    path: a company's own data, in whatever shape it's in, accumulates
    here permanently -- upload more later, and the chat draws on
    everything uploaded to date, every session, forever (until you
    delete it), independent of the churn model.

  - .pdf / .docx / .txt / .md -> text extracted and added to the same
    persistent knowledge base.

  - .png / .jpg / .webp -> sent to a vision-capable model (only the
    Hugging Face backend currently has one).

A sidebar picker lets the user choose which backend + model answers
their questions, sourced from AVAILABLE_MODELS in llm_playbook.py, and
a sidebar panel shows/manages everything currently in the knowledge base.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from common import get_model, get_scored_data, get_store, current_uploaded_count
from qa_engine import answer_question, answer_image
from llm_playbook import AVAILABLE_MODELS, BACKEND
from db import validate_upload, save_uploaded_df, REQUIRED_COLUMNS
import knowledge_base as kb

st.set_page_config(page_title="Playground — Churn Retention AI", layout="wide")

# ---------------- Sidebar: data + model picker ----------------
st.sidebar.title("⚙️ Playground settings")
n_customers = st.sidebar.slider("Number of synthetic customers", 500, 10000, 5000, step=500)
st.sidebar.page_link("app.py", label="⬅ Back to Dashboard", icon="📊")

st.sidebar.divider()
st.sidebar.subheader("🧠 Model")
backend_names = list(AVAILABLE_MODELS.keys())
default_idx = backend_names.index(BACKEND) if BACKEND in backend_names else 0
selected_backend = st.sidebar.selectbox("Backend", backend_names, index=default_idx)

model_options = AVAILABLE_MODELS[selected_backend]
model_labels = [m["label"] for m in model_options]
selected_label = st.sidebar.selectbox("Model", model_labels)
selected_model_entry = next(m for m in model_options if m["label"] == selected_label)
selected_model = selected_model_entry["model"]
selected_vision_capable = selected_model_entry["vision"]

if selected_vision_capable:
    st.sidebar.caption("✅ This model can read attached images.")
else:
    st.sidebar.caption(
        "ℹ️ This model is text-only. Attaching an image will automatically "
        "use the Hugging Face vision model instead, regardless of this selection."
    )

st.sidebar.divider()
uploaded_count = current_uploaded_count()
if uploaded_count > 0:
    st.sidebar.success(f"📥 {uploaded_count} customer record(s) in the churn model's training data")

st.sidebar.divider()
st.sidebar.subheader("📚 Knowledge base")
kb_sources = kb.list_sources()
kb_total = kb.total_chunks()
if kb_sources:
    st.sidebar.caption(f"{len(kb_sources)} source(s), {kb_total} chunk(s) total -- persists across restarts")
    for s in kb_sources:
        label = f"{s['source']}"
        if s["batch_label"]:
            label += f"  ·  {s['batch_label']}"
        with st.sidebar.expander(label, expanded=False):
            st.caption(f"{s['kind']}, {s['chunks']} chunk(s), added {s['uploaded_at'][:10]}")
            if st.button("🗑️ Remove", key=f"del_{s['source']}_{s['batch_label']}"):
                kb.delete_source(s["source"])
                st.rerun()
else:
    st.sidebar.caption("Empty. Attach a document, CSV, or Excel file to start building it.")

pipe = get_model(n_customers, uploaded_count)
store = get_store()
df = get_scored_data(n_customers, uploaded_count)

# ---------------- Header ----------------
st.title("🎮 Playground")
st.caption(
    "Chat about the churn data, or about anything you've added to the knowledge base. "
    "Use 📎 to attach files -- CSV/Excel/PDF/Word data accumulates permanently so it "
    "keeps growing as you add more over time."
)

with st.expander("💡 Example questions", expanded=False):
    st.markdown(
        "- Why is customer C102629 at risk?\n"
        "- What should we do for customer C101541?\n"
        "- How many customers are flagged?\n"
        "- What's the churn rate for region west?\n"
        "- What are the top drivers of churn?\n"
        "- Remember: prioritize fee waivers over discounts for high-income customers\n"
        "- (after attaching your own data) Ask anything about it -- it's searched automatically."
    )

if "playground_history" not in st.session_state:
    st.session_state.playground_history = []
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

for role, msg in st.session_state.playground_history:
    with st.chat_message(role):
        st.markdown(msg)


def _last_exchange():
    history = st.session_state.playground_history
    for i in range(len(history) - 1, 0, -1):
        if history[i][0] == "assistant" and history[i - 1][0] == "user":
            return (history[i - 1][1], history[i][1])
    return None


def _append_and_show(role, content):
    st.session_state.playground_history.append((role, content))


# ---------------- Attach control ----------------
attach_col, clear_col = st.columns([1, 1])
with attach_col:
    with st.popover("📎 Attach a file"):
        st.caption(
            "Customer-schema CSV → retrains the churn model. Any other CSV/Excel, "
            "or a PDF/Word/text file → added to the persistent knowledge base. "
            "Image → vision model."
        )
        batch_label = st.text_input(
            "Optional label for this upload (e.g. 'support tickets - March')",
            key="batch_label_input",
            help="Helps you track which upload a piece of data came from. Leave blank if not needed.",
        )
        uploaded = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"],
            key="attach_uploader",
        )

        if uploaded is not None:
            file_id = f"{uploaded.name}:{uploaded.size}"
            already_processed = st.session_state.last_processed_file == file_id
            ext = uploaded.name.lower().rsplit(".", 1)[-1]
            file_bytes = uploaded.getvalue()

            # ---- CSV: try churn schema first, else knowledge base ----
            if ext == "csv":
                try:
                    new_df = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f"Couldn't read that CSV: {e}")
                    new_df = None

                if new_df is not None:
                    ok, msg = validate_upload(new_df)
                    if ok:
                        st.success(f"This matches the churn-model schema -- {len(new_df)} row(s).")
                        st.dataframe(new_df.head(5), use_container_width=True, hide_index=True)
                        if st.button("➕ Add to churn model training data and retrain"):
                            save_uploaded_df(new_df)
                            st.cache_resource.clear()
                            st.cache_data.clear()
                            _append_and_show(
                                "assistant",
                                f"📥 Added {len(new_df)} customer record(s) from `{uploaded.name}` "
                                f"to the churn model's training data and retrained it.",
                            )
                            st.rerun()
                    else:
                        st.info(
                            f"This doesn't match the churn-model schema ({msg}) -- "
                            f"that's fine, it'll go into the general knowledge base instead."
                        )
                        st.dataframe(new_df.head(5), use_container_width=True, hide_index=True)
                        if not already_processed and st.button("➕ Add to knowledge base"):
                            n_chunks = kb.add_tabular(new_df, uploaded.name, batch_label or None)
                            st.session_state.last_processed_file = file_id
                            _append_and_show(
                                "assistant",
                                f"📚 Added `{uploaded.name}` ({len(new_df)} rows, {n_chunks} chunk(s)) "
                                f"to the knowledge base"
                                + (f" under '{batch_label}'." if batch_label else ".")
                                + " Ask me anything about it.",
                            )
                            st.rerun()

            # ---- Excel: always goes to the knowledge base ----
            elif ext in ("xlsx", "xls"):
                if not already_processed:
                    try:
                        new_df = kb.read_tabular(file_bytes, uploaded.name)
                        st.success(f"Read {len(new_df)} row(s).")
                        st.dataframe(new_df.head(5), use_container_width=True, hide_index=True)
                        if st.button("➕ Add to knowledge base", key="add_xlsx"):
                            n_chunks = kb.add_tabular(new_df, uploaded.name, batch_label or None)
                            st.session_state.last_processed_file = file_id
                            _append_and_show(
                                "assistant",
                                f"📚 Added `{uploaded.name}` ({len(new_df)} rows, {n_chunks} chunk(s)) "
                                f"to the knowledge base"
                                + (f" under '{batch_label}'." if batch_label else ".")
                                + " Ask me anything about it.",
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't read that Excel file: {e}")
                else:
                    st.info(f"`{uploaded.name}` was already added. Attach a different file to add more.")

            # ---- PDF/Word/text: knowledge base ----
            elif ext in ("pdf", "docx", "txt", "md"):
                if not already_processed:
                    try:
                        n_chunks = kb.add_document(file_bytes, uploaded.name, batch_label or None)
                        st.session_state.last_processed_file = file_id
                        st.success(f"Read `{uploaded.name}` -- {n_chunks} chunk(s) indexed.")
                        _append_and_show(
                            "assistant",
                            f"📄 Added `{uploaded.name}` ({n_chunks} chunk(s)) to the knowledge base"
                            + (f" under '{batch_label}'." if batch_label else ".")
                            + " Ask me anything about it.",
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't extract text from `{uploaded.name}`: {e}")
                else:
                    st.info(f"`{uploaded.name}` was already added. Attach a different file to add more.")

            # ---- Image: vision model (not persisted to the knowledge base) ----
            elif ext in ("png", "jpg", "jpeg", "webp"):
                st.image(file_bytes, caption=uploaded.name, use_container_width=True)
                img_question = st.text_input(
                    "Ask something about this image (optional)",
                    placeholder="e.g. What does this chart show?",
                    key="img_question",
                )
                if st.button("🔍 Analyze image"):
                    mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
                    vision_model = (
                        selected_model if selected_vision_capable
                        else next(m["model"] for m in AVAILABLE_MODELS["huggingface"] if m["vision"])
                    )
                    with st.spinner("Analyzing image..."):
                        answer = answer_image(img_question, file_bytes, mime, model=vision_model)
                    q_display = img_question or "(no question given -- general description)"
                    _append_and_show("user", f"🖼️ [Attached `{uploaded.name}`] {q_display}")
                    _append_and_show("assistant", answer)
                    st.rerun()

with clear_col:
    if st.session_state.playground_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.playground_history = []
            st.rerun()

# ---------------- Chat input ----------------
user_q = st.chat_input("Ask a question about the customers, churn drivers, or your uploaded data...")
if user_q:
    last_exchange = _last_exchange()
    _append_and_show("user", user_q)
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(
                user_q, df, pipe, store,
                last_exchange=last_exchange,
                backend=selected_backend,
                model=selected_model,
            )
        st.markdown(answer)
    _append_and_show("assistant", answer)