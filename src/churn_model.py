"""
Trains a churn model (GradientBoostingClassifier -- a free, no-install
substitute for XGBoost) and produces PER-CUSTOMER explanations.

Explainability approach:
  We don't have SHAP available offline, so we implement a lightweight
  per-instance explanation using signed marginal contributions: for each
  feature, we perturb it to the population median and measure how much
  the predicted probability moves. This is a simplified, model-agnostic
  stand-in for SHAP values -- same spirit (local, additive-ish attribution),
  cheaper to compute, zero dependencies.

  If you later `pip install shap xgboost`, swap `explain_customer()` for
  `shap.TreeExplainer(model).shap_values(row)` -- the rest of the pipeline
  (playbook generation, fairness check) doesn't need to change.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "age", "tenure_months", "num_products", "avg_monthly_spend_now",
    "spend_decline_pct", "missed_payments_12m", "complaints_12m",
    "days_since_last_login", "rewards_redeemed_12m", "credit_utilization",
    "customer_service_calls_12m",
]
CATEGORICAL_FEATURES = ["income_band", "region"]  # kept OUT of the model's
# risk score inputs deliberately is not required here since these aren't
# protected attributes, but region/income are still audited post-hoc in
# fairness_check.py to mirror real fair-lending review practice.
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")
    model = GradientBoostingClassifier(random_state=42)
    return Pipeline([("pre", pre), ("model", model)])


def train(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    # Class-imbalance fix: up-weight the minority (churn) class instead of
    # resampling, so probabilities stay well-calibrated.
    pos_rate = y_train.mean()
    sample_weight = np.where(y_train == 1, (1 - pos_rate) / pos_rate, 1.0)
    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Test AUC: {auc:.3f}")
    print(classification_report(y_test, (proba > 0.5).astype(int)))
    return pipe, X_test, y_test


def explain_customer(pipe, row: pd.Series, background: pd.DataFrame, top_k=3):
    """Simplified local explanation: perturb each numeric feature to the
    population median one at a time, measure the drop/rise in predicted
    churn probability caused by that feature's actual value vs. 'typical'."""
    base_row = row[ALL_FEATURES].to_frame().T
    base_prob = pipe.predict_proba(base_row)[0, 1]

    contributions = {}
    for feat in NUMERIC_FEATURES:
        median_val = background[feat].median()
        perturbed = base_row.copy()
        perturbed[feat] = median_val
        perturbed_prob = pipe.predict_proba(perturbed)[0, 1]
        contributions[feat] = base_prob - perturbed_prob  # positive = raises risk

    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_reasons = ranked[:top_k]
    return base_prob, top_reasons


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/churn_retention_ai/data/customers.csv")
    pipe, X_test, y_test = train(df)
    import joblib
    joblib.dump(pipe, "/home/claude/churn_retention_ai/outputs/churn_pipeline.joblib")

    # Show a worked example
    sample_idx = X_test.index[0]
    prob, reasons = explain_customer(pipe, df.loc[sample_idx], df)
    print(f"\nExample customer {df.loc[sample_idx, 'customer_id']}: churn prob = {prob:.2f}")
    for feat, contrib in reasons:
        direction = "increases" if contrib > 0 else "decreases"
        print(f"  - {feat} {direction} risk (impact {contrib:+.3f})")
