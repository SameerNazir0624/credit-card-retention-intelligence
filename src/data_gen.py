"""
Generates a synthetic credit-card customer dataset for churn modeling.
100% synthetic -- no real customer data, so it's safe to share/showcase.
"""
import numpy as np
import pandas as pd

def generate_customers(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(3, 120, n)
    avg_monthly_spend_6m_ago = rng.gamma(shape=3.0, scale=400, size=n)
    spend_decline_pct = rng.normal(0, 0.25, n).clip(-1, 1)
    avg_monthly_spend_now = (avg_monthly_spend_6m_ago * (1 + spend_decline_pct)).clip(0)

    num_products = rng.integers(1, 5, n)
    missed_payments_12m = rng.poisson(0.4, n)
    complaints_12m = rng.poisson(0.15, n)
    days_since_last_login = rng.exponential(15, n).clip(0, 400).astype(int)
    rewards_redeemed_12m = rng.poisson(2, n)
    credit_utilization = rng.beta(2, 5, n)
    customer_service_calls_12m = rng.poisson(1.2, n)
    age = rng.integers(21, 75, n)
    income_band = rng.choice(["low", "mid", "high"], size=n, p=[0.3, 0.5, 0.2])
    region = rng.choice(["north", "south", "east", "west"], size=n)

    # Latent churn propensity, built from realistic drivers (not the demographic fields)
    logit = (
        -1.5
        - 0.015 * tenure_months
        - 2.2 * spend_decline_pct
        + 0.55 * missed_payments_12m
        + 0.9 * complaints_12m
        + 0.02 * days_since_last_login
        - 0.25 * rewards_redeemed_12m
        + 0.4 * customer_service_calls_12m
        - 0.3 * num_products
        + rng.normal(0, 0.6, n)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churned = (rng.random(n) < prob_churn).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "age": age,
        "income_band": income_band,
        "region": region,
        "tenure_months": tenure_months,
        "num_products": num_products,
        "avg_monthly_spend_6m_ago": avg_monthly_spend_6m_ago.round(2),
        "avg_monthly_spend_now": avg_monthly_spend_now.round(2),
        "spend_decline_pct": spend_decline_pct.round(3),
        "missed_payments_12m": missed_payments_12m,
        "complaints_12m": complaints_12m,
        "days_since_last_login": days_since_last_login,
        "rewards_redeemed_12m": rewards_redeemed_12m,
        "credit_utilization": credit_utilization.round(3),
        "customer_service_calls_12m": customer_service_calls_12m,
        "churned": churned,
    })
    return df

if __name__ == "__main__":
    df = generate_customers()
    df.to_csv("/home/claude/churn_retention_ai/data/customers.csv", index=False)
    print(df.shape)
    print(df["churned"].mean())
