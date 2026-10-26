"""Streamlit app for churn prediction. Run: streamlit run app.py"""
import json
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/churn_model.joblib"
META_PATH = "models/meta.json"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load("models\churn_model.joblib")


model = load_model()
if os.path.exists("models\meta.json"):
    st.caption(f"Model ROC-AUC on held-out data: {json.load(open(META_PATH))['roc_auc']}")

FEATURES = ["tenure", "monthly_charges", "total_charges", "senior_citizen", "gender",
            "partner", "dependents", "phone_service", "internet_service", "contract",
            "paperless_billing", "payment_method", "tech_support", "online_security"]


def risk_label(p):
    return "🔴 High" if p >= 0.6 else "🟠 Medium" if p >= 0.3 else "🟢 Low"


tab1, tab2 = st.tabs(["Single customer", "Batch (CSV)"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
    with c2:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with c3:
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method",
                               ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        tech = st.selectbox("Tech support", ["Yes", "No"])
        security = st.selectbox("Online security", ["Yes", "No"])

    if st.button("Predict churn", type="primary"):
        row = pd.DataFrame([{
            "tenure": tenure, "monthly_charges": monthly,
            "total_charges": round(monthly * tenure, 2), "senior_citizen": senior,
            "gender": gender, "partner": partner, "dependents": dependents,
            "phone_service": phone, "internet_service": internet, "contract": contract,
            "paperless_billing": paperless, "payment_method": payment,
            "tech_support": tech, "online_security": security}])[FEATURES]
        p = model.predict_proba(row)[0, 1]
        st.metric("Churn probability", f"{p:.1%}")
        st.progress(float(p))
        st.subheader(f"Risk level: {risk_label(p)}")

with tab2:
    st.write("Upload a CSV with columns: " + ", ".join(FEATURES))
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["churn_probability"] = model.predict_proba(data[FEATURES])[:, 1].round(4)
            data["risk"] = data["churn_probability"].apply(risk_label)
            st.dataframe(data)
            st.download_button("Download predictions", data.to_csv(index=False),
                               "predictions.csv", "text/csv")
