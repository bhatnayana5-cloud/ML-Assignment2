
import os, joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

TARGET = "HeartDisease"   # <- must match TARGET in train_models.py

# Fallback to current directory if running in an interactive Jupyter/Colab environment
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

# The trained models are saved in the same directory as the script (or '/content' in Colab)
MODEL_DIR = HERE
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

def compute_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    multi = y.nunique() > 2
    auc = roc_auc_score(y, proba, multi_class="ovr") if multi else roc_auc_score(y, proba[:, 1])
    avg = "weighted" if multi else "binary"
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average=avg, zero_division=0),
        "Recall": recall_score(y, pred, average=avg, zero_division=0),
        "F1": f1_score(y, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

st.set_page_config(page_title="Classifier Explorer", layout="wide")
st.title("Classifier Explorer")
st.caption("Upload test data, pick a model, inspect the metrics.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Model", list(MODELS))

if uploaded is None:
    st.info("Upload test_data.csv from the repo to begin.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"CSV must contain the label column '{TARGET}'.")
    st.stop()

cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
X = df.reindex(columns=cols, fill_value=0)
y = df[TARGET]
model = joblib.load(os.path.join(MODEL_DIR, MODELS[choice]))

st.subheader(f"Metrics — {choice}")
m = compute_metrics(model, X, y)
for c, (k, v) in zip(st.columns(len(m)), m.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, model.predict(X)), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(classification_report(y, model.predict(X), output_dict=True)).T.round(3))

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20))