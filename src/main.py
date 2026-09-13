"""
End-to-end pipeline:
  1. Generate/load synthetic customer data
  2. Train churn model + get per-customer explanations
  3. Retrieve relevant retention playbook via TF-IDF similarity search
  4. Generate a personalized retention recommendation (GenAI layer)
  5. Run a fairness/governance audit across the flagged population
  6. Save a report of the top-N highest-risk customers with recommendations
"""
import pandas as pd
import joblib
import os

from data_gen import generate_customers
from churn_model import train, explain_customer, ALL_FEATURES
from vector_store import PlaybookStore
from llm_playbook import generate_recommendation, FEATURE_LABELS
from fairness_check import run_full_audit

DATA_DIR = "/home/claude/churn_retention_ai/data"
OUT_DIR = "/home/claude/churn_retention_ai/outputs"
TOP_N = 10


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = f"{DATA_DIR}/customers.csv"
    if not os.path.exists(csv_path):
        df = generate_customers()
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} customers ({df['churned'].mean():.1%} historical churn rate)")

    pipe, X_test, y_test = train(df)
    joblib.dump(pipe, f"{OUT_DIR}/churn_pipeline.joblib")

    df["churn_prob"] = pipe.predict_proba(df[ALL_FEATURES])[:, 1]
    df["flagged"] = (df["churn_prob"] > 0.5).astype(int)

    print("\n=== Fairness / governance audit ===")
    audit = run_full_audit(df)
    for group_col, result in audit.items():
        print(f"\nFlag rate by {group_col}:")
        for g, r in result["rates"].items():
            print(f"  {g}: {r:.1%}")
        if result["groups_needing_review"]:
            print(f"  ⚠ Review needed: {result['groups_needing_review']}")
        else:
            print("  ✓ Passes 80% rule")

    print(f"\n=== Generating recommendations for top {TOP_N} at-risk customers ===")
    store = PlaybookStore()
    top_risk = df.sort_values("churn_prob", ascending=False).head(TOP_N)

    rows = []
    for idx, row in top_risk.iterrows():
        prob, reasons = explain_customer(pipe, row, df)
        # Build a natural-language query from the top risk drivers so
        # TF-IDF similarity actually has overlapping vocabulary to match on.
        query = " ".join(FEATURE_LABELS.get(feat, feat) for feat, contrib in reasons if contrib > 0)
        if not query:
            query = " ".join(FEATURE_LABELS.get(feat, feat) for feat, _ in reasons)
        retrieved = store.retrieve(query, k=2)
        rec, _ = generate_recommendation(prob, reasons, retrieved)
        rows.append({
            "customer_id": row["customer_id"],
            "churn_prob": round(prob, 3),
            "top_reasons": "; ".join(f"{f} ({c:+.3f})" for f, c in reasons),
            "playbook_used": retrieved[0][0]["title"],
            "recommendation": rec,
        })
        print(f"\n[{row['customer_id']}] churn_prob={prob:.2f}")
        print(f"  Recommendation: {rec}")

    report_df = pd.DataFrame(rows)
    report_path = f"{OUT_DIR}/retention_recommendations.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
