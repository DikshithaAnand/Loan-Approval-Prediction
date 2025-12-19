import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---- Load Dataset ----
df = pd.read_csv("data/loan_prediction.csv")

# Drop rows where target missing
df = df.dropna(subset=["Loan_Status"])

# Drop ID column if present
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Convert target to numeric
y = df["Loan_Status"].map(lambda v: 1 if str(v).upper() == "Y" else 0)
X = df.drop(columns=["Loan_Status"])

# Split categorical vs numerical
categorical = [c for c in X.columns if X[c].dtype == "object"]
numerical = [c for c in X.columns if c not in categorical]

# Preprocessing pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical),
    ("cat", categorical_transformer, categorical)
])

# ---- Define Models ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = []

# ---- Train & Evaluate ----
for name, model in models.items():
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n===== {name} =====")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print(classification_report(y_test, y_pred))
    
    results.append((name, pipe, acc, f1))

# ---- Pick Best Model by F1 first, then Accuracy ----
results_sorted = sorted(results, key=lambda x: (x[3], x[2]), reverse=True)
best_name, best_model, best_acc, best_f1 = results_sorted[0]

print("\n\n🎉 Best Model Selected:", best_name)
print("Best Accuracy:", round(best_acc, 4))
print("Best F1 Score:", round(best_f1, 4))

# ---- Save the Best Model ----
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/loan_approval_model.pkl")
print("\n💾 Model saved to: models/loan_approval_model.pkl")

# ----- FEATURE IMPORTANCE FOR BEST MODEL (if Logistic Regression) -----
if best_name == "Logistic Regression":
    print("\n📊 Generating Feature Importance for Logistic Regression...\n")
    
    # Extract preprocessing & LR classifier
    pre = best_model.named_steps["pre"]
    clf = best_model.named_steps["clf"]
    
    # OneHot categories
    ohe = pre.named_transformers_["cat"].named_steps["encoder"]
    cat_features = ohe.get_feature_names_out(categorical)
    
    # Combine numeric + OHE names
    feature_names = np.concatenate([numerical, cat_features])
    
    # Extract coefficients
    coefs = clf.coef_[0]
    
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    
    print("\nTop Influential Features:")
    print(coef_df[["feature", "coef"]].head(10))
    
    # Plot
    plt.figure(figsize=(10,6))
    sns.barplot(data=coef_df[:15], x="coef", y="feature", palette="coolwarm")
    plt.title("Feature Influence on Loan Approval (Logistic Regression)")
    plt.xlabel("Coefficient (Impact)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ Feature importance visualization only available for Logistic Regression.")
