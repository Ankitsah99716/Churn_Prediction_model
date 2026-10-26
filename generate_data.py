"""Generate a synthetic telecom churn dataset (data/churn.csv)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

df = pd.DataFrame({
    "customer_id": [f"CUST-{i:05d}" for i in range(N)],
    "gender": rng.choice(["Male", "Female"], N),
    "senior_citizen": rng.choice([0, 1], N, p=[0.84, 0.16]),
    "partner": rng.choice(["Yes", "No"], N),
    "dependents": rng.choice(["Yes", "No"], N, p=[0.3, 0.7]),
    "tenure": rng.integers(0, 73, N),
    "phone_service": rng.choice(["Yes", "No"], N, p=[0.9, 0.1]),
    "internet_service": rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.2]),
    "contract": rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.2]),
    "paperless_billing": rng.choice(["Yes", "No"], N, p=[0.6, 0.4]),
    "payment_method": rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N),
    "tech_support": rng.choice(["Yes", "No"], N, p=[0.35, 0.65]),
    "online_security": rng.choice(["Yes", "No"], N, p=[0.35, 0.65]),
})

base = {"DSL": 45, "Fiber optic": 80, "No": 20}
df["monthly_charges"] = (df["internet_service"].map(base)
                         + rng.normal(0, 8, N)
                         + np.where(df["phone_service"] == "Yes", 10, 0)).clip(18, 120).round(2)
df["total_charges"] = (df["monthly_charges"] * df["tenure"]
                       * rng.uniform(0.95, 1.05, N)).round(2)

# Churn probability from a logistic model of realistic drivers
z = (-1.0
     + 1.2 * (df["contract"] == "Month-to-month")
     - 1.0 * (df["contract"] == "Two year")
     + 0.8 * (df["internet_service"] == "Fiber optic")
     + 0.6 * (df["payment_method"] == "Electronic check")
     + 0.3 * df["senior_citizen"]
     + 0.3 * (df["paperless_billing"] == "Yes")
     - 0.6 * (df["tech_support"] == "Yes")
     - 0.5 * (df["online_security"] == "Yes")
     - 0.04 * df["tenure"]
     + 0.012 * (df["monthly_charges"] - 60))
p = 1 / (1 + np.exp(-z))
df["churn"] = np.where(rng.random(N) < p, "Yes", "No")

df.to_csv("data/churn.csv", index=False)
print(df.shape, "churn rate:", (df["churn"] == "Yes").mean().round(3))
