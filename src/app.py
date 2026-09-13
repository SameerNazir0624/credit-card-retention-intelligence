"""
Streamlit dashboard for the Explainable Churn + GenAI Retention Playbook system.

This is the main/home page. The chat playground lives on its own page at
pages/1_Playground.py -- Streamlit auto-detects the "pages" folder and adds
sidebar navigation for it automatically.

Run with:
    streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from churn_model import explain_customer, ALL_FEATURES
from llm_playbook import generate_recommendation, FEATURE_LABELS
from fairness_check import run_full_audit
from common import get_model, get_scored_data, get_store, current_uploaded_count

st.set_page_config(page_title="Churn Retention AI", layout="wide")


def build_recommendation(pipe, store, row, background):
    prob, reasons = explain_customer(pipe, row, background)
    query = " ".join(FEATURE_LABELS.get(f, f) for f, c in reasons if c > 0)
    if not query:
        query = " ".join(FEATURE_LABELS.get(f, f) for f, _ in reasons)
    retrieved = store.retrieve(query, k=2)
    rec, _ = generate_recommendation(prob, reasons, retrieved)
    return prob, reasons, retrieved, rec


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Controls")
n_customers = st.sidebar.slider("Number of synthetic customers", 500, 10000, 5000, step=500)
top_n = st.sidebar.slider("How many at-risk customers to show", 5, 50, 10)
st.sidebar.caption(
    "Data is 100% synthetic. Model = GradientBoostingClassifier. "
    "Explanations = perturbation-based local attribution. "
    "Retrieval = TF-IDF over a retention playbook."
)
st.sidebar.divider()
st.sidebar.page_link("pages/1_Playground.py", label="🎮 Open the Playground", icon="🎮")

# ---------- Data + model (cached, shared with the Playground page) ----------
uploaded_count = current_uploaded_count()
pipe = get_model(n_customers, uploaded_count)
store = get_store()
df = get_scored_data(n_customers, uploaded_count)
if uploaded_count > 0:
    st.sidebar.success(f"📥 {uploaded_count} uploaded customer(s) included in training")

# ---------- Header ----------
header_left, header_right = st.columns([5, 1])
with header_left:
    st.title("💳 Explainable Churn Early-Warning + GenAI Retention Playbooks")
    st.caption("Predict → Explain → Recommend → Govern — an end-to-end retention pipeline")
with header_right:
    st.write("")
    st.page_link("pages/1_Playground.py", label="🎮 Playground", icon="🎮", use_container_width=True)

# ---------- KPI row ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(df):,}")
col2.metric("Historical churn rate", f"{df['churned'].mean():.1%}")
col3.metric("Currently flagged at-risk", f"{df['flagged'].mean():.1%}")
col4.metric("Avg. predicted churn prob.", f"{df['churn_prob'].mean():.1%}")

st.divider()

# ---------- Tabs: Dashboard / What-if ----------
tab_dashboard, tab_whatif = st.tabs(["📊 Dashboard", "🎛️ What-if simulator"])

with tab_dashboard:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("🚨 Top at-risk customers")
        top_risk = df.sort_values("churn_prob", ascending=False).head(top_n)
        st.dataframe(
            top_risk[["customer_id", "churn_prob", "tenure_months", "missed_payments_12m",
                      "complaints_12m", "customer_service_calls_12m", "num_products"]]
            .rename(columns={"churn_prob": "Churn probability"})
            .style.format({"Churn probability": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("🔍 Customer deep-dive")
        selected_id = st.selectbox("Pick a customer to inspect", top_risk["customer_id"])
        row = df[df["customer_id"] == selected_id].iloc[0]
        prob, reasons, retrieved, rec = build_recommendation(pipe, store, row, df)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted churn probability", f"{prob:.1%}")
            st.write("**Top risk drivers:**")
            for feat, contrib in reasons:
                label = FEATURE_LABELS.get(feat, feat)
                direction = "🔺 raises risk" if contrib > 0 else "🔻 lowers risk"
                st.write(f"- {label}: {direction} (impact `{contrib:+.3f}`)")

        with c2:
            st.write("**Retrieved retention playbook(s):**")
            for doc, score in retrieved:
                st.info(f"**{doc['title']}** (similarity {score:.2f})\n\n{doc['text']}")
            st.write("**Generated recommendation:**")
            st.success(rec)

    with right:
        st.subheader("⚖️ Fairness / governance audit")
        st.caption("Four-fifths rule check on who gets flagged as at-risk")
        audit = run_full_audit(df)
        for group_col, result in audit.items():
            st.write(f"**By {group_col}**")
            rates_df = pd.DataFrame.from_dict(result["rates"], orient="index", columns=["Flag rate"])
            st.bar_chart(rates_df)
            if result["groups_needing_review"]:
                st.warning(f"⚠ Below 80% threshold: {list(result['groups_needing_review'].keys())}")
            else:
                st.success("✓ Passes 80% rule")

    st.divider()
    st.subheader("📊 Churn probability distribution")
    st.bar_chart(df["churn_prob"].round(1).value_counts().sort_index())

with tab_whatif:
    st.subheader("🎛️ Build a hypothetical customer and see the model react live")
    st.caption("Move the sliders — churn probability and the recommendation update instantly.")

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        w_tenure = st.slider("Tenure (months)", 1, 120, 24)
        w_products = st.slider("Number of products", 1, 5, 1)
        w_spend_decline = st.slider("Spend decline (%)", -100, 100, 20) / 100.0
    with wc2:
        w_missed = st.slider("Missed payments (12m)", 0, 10, 1)
        w_complaints = st.slider("Complaints (12m)", 0, 10, 1)
        w_calls = st.slider("Customer service calls (12m)", 0, 15, 2)
    with wc3:
        w_login_gap = st.slider("Days since last login", 0, 365, 20)
        w_rewards = st.slider("Rewards redeemed (12m)", 0, 20, 1)
        w_utilization = st.slider("Credit utilization", 0.0, 1.0, 0.4)

    hypothetical = pd.DataFrame([{
        "age": 40,
        "income_band": "mid",
        "region": "north",
        "tenure_months": w_tenure,
        "num_products": w_products,
        "avg_monthly_spend_now": 500 * (1 - w_spend_decline),
        "spend_decline_pct": w_spend_decline,
        "missed_payments_12m": w_missed,
        "complaints_12m": w_complaints,
        "days_since_last_login": w_login_gap,
        "rewards_redeemed_12m": w_rewards,
        "credit_utilization": w_utilization,
        "customer_service_calls_12m": w_calls,
    }])

    hyp_prob = pipe.predict_proba(hypothetical[ALL_FEATURES])[0, 1]
    st.metric("Predicted churn probability", f"{hyp_prob:.1%}")
    st.progress(min(int(hyp_prob * 100), 100))

    hyp_prob2, hyp_reasons = explain_customer(pipe, hypothetical.iloc[0], df)
    q = " ".join(FEATURE_LABELS.get(f, f) for f, c in hyp_reasons if c > 0) or \
        " ".join(FEATURE_LABELS.get(f, f) for f, _ in hyp_reasons)
    hyp_retrieved = store.retrieve(q, k=2)
    hyp_rec, _ = generate_recommendation(hyp_prob2, hyp_reasons, hyp_retrieved)

    st.write("**What's driving this prediction:**")
    for feat, contrib in hyp_reasons:
        label = FEATURE_LABELS.get(feat, feat)
        direction = "🔺 raises risk" if contrib > 0 else "🔻 lowers risk"
        st.write(f"- {label}: {direction} (impact `{contrib:+.3f}`)")

    st.write("**Live-generated recommendation:**")
    st.success(hyp_rec)
