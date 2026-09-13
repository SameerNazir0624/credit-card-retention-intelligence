"""
Free, local, zero-setup persistence layer using SQLite (built into Python,
no server, no signup, no internet needed). Every customer record a user
uploads through the Playground's file uploader gets saved here, so it
persists across app restarts and gets folded into model training next
time the app scores/trains -- this is the "model learns from data you
share" piece.

If you'd rather use a hosted free database later (e.g. Supabase's free
Postgres tier), swap the functions below for equivalent SQL calls -- the
rest of the app only calls these three functions, so nothing else needs
to change.
"""
import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploaded_customers.db")

REQUIRED_COLUMNS = [
    "customer_id", "age", "income_band", "region", "tenure_months",
    "num_products", "avg_monthly_spend_6m_ago", "avg_monthly_spend_now",
    "spend_decline_pct", "missed_payments_12m", "complaints_12m",
    "days_since_last_login", "rewards_redeemed_12m", "credit_utilization",
    "customer_service_calls_12m", "churned",
]


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_customers (
            customer_id TEXT PRIMARY KEY,
            age INTEGER, income_band TEXT, region TEXT, tenure_months INTEGER,
            num_products INTEGER, avg_monthly_spend_6m_ago REAL, avg_monthly_spend_now REAL,
            spend_decline_pct REAL, missed_payments_12m INTEGER, complaints_12m INTEGER,
            days_since_last_login INTEGER, rewards_redeemed_12m INTEGER,
            credit_utilization REAL, customer_service_calls_12m INTEGER, churned INTEGER
        )
    """)
    conn.commit()
    conn.close()


def validate_upload(df: pd.DataFrame):
    """Returns (ok: bool, message: str). Checks required columns exist."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "OK"


def save_uploaded_df(df: pd.DataFrame):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = df[REQUIRED_COLUMNS].copy()
    placeholders = ", ".join(["?"] * len(REQUIRED_COLUMNS))
    cols = ", ".join(REQUIRED_COLUMNS)
    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
    conn.executemany(
        f"INSERT OR REPLACE INTO uploaded_customers ({cols}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    conn.close()


def load_uploaded_df() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM uploaded_customers", conn)
    except Exception:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    conn.close()
    return df


def count_uploaded() -> int:
    return len(load_uploaded_df())
