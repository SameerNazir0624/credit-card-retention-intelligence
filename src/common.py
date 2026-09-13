"""
Shared, cached resources used by both app.py (main dashboard) and
pages/1_Playground.py. Streamlit's cache is shared across all pages of
the same running app, so defining these once here means the model only
trains once no matter which page the user is on.

get_model / get_scored_data now fold in any customer rows the user has
uploaded through the Playground (persisted in SQLite via db.py), so the
model actually retrains on shared data instead of only ever seeing the
synthetic baseline.
"""
import streamlit as st
import pandas as pd

from data_gen import generate_customers
from churn_model import train, ALL_FEATURES
from vector_store import PlaybookStore
from db import load_uploaded_df, count_uploaded


def _combined_dataset(n_customers: int) -> pd.DataFrame:
    base = generate_customers(n=n_customers)
    uploaded = load_uploaded_df()
    if len(uploaded) > 0:
        combined = pd.concat([base, uploaded], ignore_index=True)
        combined = combined.drop_duplicates(subset="customer_id", keep="last")
        return combined
    return base


@st.cache_resource
def get_model(n_customers: int, uploaded_count: int = 0):
    # uploaded_count is part of the cache key on purpose: when it changes
    # (a new upload happened), Streamlit invalidates this cache entry and
    # retrains -- that's the actual "learning from shared data" mechanism.
    df = _combined_dataset(n_customers)
    pipe, X_test, y_test = train(df)
    return pipe


@st.cache_data
def get_scored_data(n_customers: int, uploaded_count: int = 0):
    df = _combined_dataset(n_customers)
    pipe = get_model(n_customers, uploaded_count)
    df["churn_prob"] = pipe.predict_proba(df[ALL_FEATURES])[:, 1]
    df["flagged"] = (df["churn_prob"] > 0.5).astype(int)
    return df


@st.cache_resource
def get_store():
    return PlaybookStore()


def current_uploaded_count() -> int:
    return count_uploaded()
