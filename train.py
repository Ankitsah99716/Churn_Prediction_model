"""Train a churn prediction model and save it to models/churn_model.joblib."""
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/churn_model.joblib"
META_PATH = "models/meta.json"

NUMERIC = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL = ["gender", "partner", "dependents", "phone_service", "internet_service",
               "contract", "paperless_billing", "payment_method",
               "tech_support", "online_security"]


def main():
    df = pd.read_csv("data/churn.csv")
    X = df[NUMERIC + CATEGORICAL]
    y = (df["churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([
        ("prep", preprocess),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print(f"ROC-AUC: {auc:.3f}")

    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"roc_auc": round(auc, 4)}, f)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
