"""
Post-hoc fairness audit: checks whether the churn model's flag rate differs
substantially across income_band / region groups, even though those fields
are NOT causal drivers we designed into the risk logic. This mirrors the
kind of disparate-impact check a fair-lending / model-governance team runs
before a model is allowed into production.

Rule of thumb used here: the "80% rule" (four-fifths rule) common in
adverse-impact analysis -- if any group's flag rate is less than 80% of the
highest group's flag rate, that's flagged for review.
"""
import pandas as pd


def audit_group_flag_rates(df: pd.DataFrame, group_col: str, flag_col: str = "flagged"):
    rates = df.groupby(group_col)[flag_col].mean().sort_values(ascending=False)
    max_rate = rates.max()
    review_needed = rates[rates < 0.8 * max_rate]
    return rates, review_needed


def run_full_audit(df: pd.DataFrame, flag_col: str = "flagged"):
    report = {}
    for col in ["income_band", "region"]:
        rates, flagged_groups = audit_group_flag_rates(df, col, flag_col)
        report[col] = {
            "rates": rates.to_dict(),
            "groups_needing_review": flagged_groups.to_dict(),
        }
    return report


if __name__ == "__main__":
    import joblib
    from churn_model import ALL_FEATURES

    df = pd.read_csv("/home/claude/churn_retention_ai/data/customers.csv")
    pipe = joblib.load("/home/claude/churn_retention_ai/outputs/churn_pipeline.joblib")
    df["churn_prob"] = pipe.predict_proba(df[ALL_FEATURES])[:, 1]
    df["flagged"] = (df["churn_prob"] > 0.5).astype(int)

    report = run_full_audit(df)
    for group_col, result in report.items():
        print(f"\n=== Flag rate by {group_col} ===")
        for g, r in result["rates"].items():
            print(f"  {g}: {r:.1%}")
        if result["groups_needing_review"]:
            print(f"  ⚠ Needs review (below 80% of max group rate): {result['groups_needing_review']}")
        else:
            print("  ✓ No group falls below the 80% threshold.")
