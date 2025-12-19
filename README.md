# 🏦 Loan Approval Prediction using Machine Learning & Streamlit  
**Status:** Production-ready • Fully Tested • Explainable ML • Deployable  

This repository contains an end-to-end ML solution that predicts whether a home loan application should be **Approved or Rejected** using applicant demographic and financial details. The project follows a complete lifecycle: data preparation, modeling, evaluation, explainability, saving the best pipeline, and deployment through Streamlit.

This project is ideal for:
- Internship submission
- Academic ML demonstration
- GitHub portfolio project
- Deployable real-world prototype  

---

## 💡 Business Motivation  

Banks face two major risks:

- **Approving loans for financially weak customers → money lost**
- **Rejecting customers who can actually repay → lost business**

Traditional manual assessment is slow and subjective.  
This project uses Machine Learning to automate loan decisions in a **fast, unbiased, data-driven** manner.

---

## 🎯 ML Problem Statement  

- **Task:** Binary Classification  
- **Objective:** Predict `Loan_Status (Y = Approved, N = Rejected)`
- **Approach:** Supervised Learning  
- **Primary Metric:** F1-Score (risk-sensitive)  
- **Secondary Metric:** Accuracy  

> F1 matters more because loan datasets are slightly imbalanced.

---

## 📊 Dataset Overview  

Total Observations: **615**  
Features: **13 input features**  
Target: **Loan_Status**

### 🔹 Target Column  
| Column       | Meaning |
|--------------|---------|
| Loan_Status  | Y = Approved, N = Rejected |

### 🔹 Feature Categories  

#### 👤 Applicant Profile  
- Gender  
- Married  
- Dependents  
- Education  
- Self_Employed  

#### 💰 Financial Attributes  
- ApplicantIncome  
- CoapplicantIncome  

#### 🧾 Loan Parameters  
- LoanAmount (in thousands ₹)  
- Loan_Amount_Term (days)  
- Credit_History (1 = Good, 0 = Bad)  

#### 🌍 Applicant Context  
- Property_Area (Urban / Semiurban / Rural)

Missing values handled using `SimpleImputer`.

---

## 🔍 EDA Highlights  

Key insights discovered:

- **Credit History is the strongest predictor of approval.**
- Semi-Urban applicants show the highest approval distribution.
- Applicant income is positively skewed.
- LoanAmount is right-skewed → median imputation preferred.
- Property_Area impacts approval likelihood.
- Rural applicants face slightly lower approval rates.

These patterns shaped model development.

---

## 🤖 Model Engineering  

A unified **Scikit-Learn Pipeline** was used:

### 🧩 Preprocessing  
- Numerical → Median Imputation + Standard Scaling  
- Categorical → Mode Imputation + One-Hot Encoding  

### 🧪 Trained Models  
The following models were trained and benchmarked:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| ⭐ Logistic Regression | **0.8618** | **0.9081** |
| Random Forest | 0.8211 | 0.8764 |
| Gradient Boosting | 0.8049 | 0.8667 |
| Decision Tree | 0.7561 | 0.8235 |

---

## 🥇 Selected Model: Logistic Regression  

**Reason for Selection:**
- Best F1-Score (handles imbalance)
- High accuracy
- Strong generalization
- Coefficient-based explainability
- No overfitting indicators  

Serialized model stored in:

