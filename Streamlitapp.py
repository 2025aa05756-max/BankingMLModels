# -----------------------------
# Import section
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# Load all saved models using pickle from the other program where the models were trained
# -----------------------------------------------------------------------------------
models = {
    "Logistic Regression": pickle.load(open("logistic_regression.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree.pkl", "rb")),
    "KNN": pickle.load(open("knn.pkl", "rb")),
    "Naive Bayes": pickle.load(open("naive_bayes.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest.pkl", "rb")),
    "XGBoost": pickle.load(open("xgboost.pkl", "rb"))
}

# Load scaler and training columns
scaler = pickle.load(open("scaler.pkl", "rb"))
training_columns = pickle.load(open("columns.pkl", "rb"))

# ---------------------------------------------------------
# Show Title and labels for the UI
# ---------------------------------------------------------
st.title("Bank Term Deposit Prediction App")
st.write("Upload test data, choose a model, and view evaluation metrics.")

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data")
    st.dataframe(df.head())

    # ---------------------------------------------------------
    # Validate target column which we used as Deposit
    # ---------------------------------------------------------
    if "Deposit" not in df.columns:
        st.error("Your CSV must contain the target column 'Deposit'.")
        st.stop()

    # Convert target to numeric BEFORE extracting y_true
    df["Deposit"] = df["Deposit"].map({"yes": 1, "no": 0}).astype(int)
    y_true = df["Deposit"]

    # Remove target column before encoding/scaling
    df_features = df.drop(columns=["Deposit"])

    # ---------------------------------------------------------
    # One-hot encode and align columns
    # ---------------------------------------------------------
    df_encoded = pd.get_dummies(df_features)
    df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)

    # Scale numerical features
    df_scaled = scaler.transform(df_encoded)

    # ---------------------------------------------------------
    # Model Selection
    # ---------------------------------------------------------
    model_name = st.selectbox("Select a Model", list(models.keys()))
    model = models[model_name]  # User selected model will load here

    # ---------------------------------------------------------
    # Predict & Evaluate based on the selected model and uploaded file
    # ---------------------------------------------------------
    if st.button("Run Evaluation"):
        y_pred = model.predict(df_scaled)
        y_prob = model.predict_proba(df_scaled)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        st.write("### Evaluation Metrics")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**AUC Score:** {auc:.4f}")

        # ---------------------------------------------------------
        # Print Confusion Matrix. 1 means yes and 0 means no in the confusion matrix
        # ---------------------------------------------------------
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ---------------------------------------------------------
        # Classification Report
        # ---------------------------------------------------------
        st.write("### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)     
        df_report = pd.DataFrame(report).transpose()
        # change 1 and 0 to yes and no to display in the classification report
        df_report.rename(index={"0": "no", "1": "yes"}, inplace=True)
        st.dataframe(df_report)