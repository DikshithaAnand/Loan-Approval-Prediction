import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_approval_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please train the model first by running: python train_model.py"
        )
    model = joblib.load(MODEL_PATH)
    return model

def main():
    st.title("🏦 Loan Approval Predictor")
    st.write("Fill in the applicant details below to estimate loan approval probability.")
    st.markdown("---")

    try:
        model = load_model()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.sidebar.title("📌 Applicant Details")

    gender = st.sidebar.selectbox(
        "Gender", 
        ["Male", "Female"], 
        help="Select applicant gender"
    )

    married = st.sidebar.selectbox(
        "Married", 
        ["Yes", "No"], 
        help="Marital status"
    )

    dependents = st.sidebar.selectbox(
        "Dependents", 
        ["0", "1", "2", "3+"], 
        help="Number of people financially dependent on the applicant"
    )

    education = st.sidebar.selectbox(
        "Education", 
        ["Graduate", "Not Graduate"],
        help="Education qualification"
    )

    self_employed = st.sidebar.selectbox(
        "Self Employed", 
        ["Yes", "No"], 
        help="Business owner or freelancer?"
    )

    applicant_income = st.sidebar.number_input(
        "Monthly Income of Applicant (₹)",
        min_value=0,
        max_value=100000,
        value=5000,
        step=500,
        help="Typical range: 1,500 to 50,000"
    )

    coapplicant_income = st.sidebar.number_input(
        "Monthly Income of Co-applicant (₹)",
        min_value=0,
        max_value=100000,
        value=0,
        step=500,
        help="Usually 0 to 40,000"
    )

    loan_amount = st.sidebar.number_input(
        "Loan Amount (in ₹ Thousands)",
        min_value=10,
        max_value=700,
        value=150,
        step=10,
        help="Enter amount in thousands. Example: 200 = ₹2,00,000"
    )

    loan_amount_term = st.sidebar.number_input(
        "Loan Duration (Days)",
        min_value=60,
        max_value=480,
        value=360,
        step=12,
        help="Valid terms: 120, 180, 240, 300, 360"
    )

    credit_history = st.sidebar.selectbox(
        "Credit History", 
        [1.0, 0.0],
        help="1 = Good Credit History, 0 = Bad Credit History"
    )

    property_area = st.sidebar.selectbox(
        "Property Area", 
        ["Urban", "Semiurban", "Rural"],
        help="Loan region"
    )

    st.markdown("---")
    st.subheader("📊 Input Summary")
    
    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": np.nan if loan_amount == 0 else loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    st.write(pd.DataFrame([input_dict]))

    if st.button("🚀 Predict Loan Approval"):
        input_df = pd.DataFrame([input_dict])

        try:
            proba = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return

        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        if pred == 1:
            st.success(f"🎉 Loan is likely to be **APPROVED**!\n📈 Approval Probability: **{proba:.2f}**")
        else:
            st.error(f"❌ Loan is likely to be **REJECTED**.\n📉 Approval Probability: **{proba:.2f}**")

if __name__ == "__main__":
    main()
