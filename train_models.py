"""Train 5 classifiers on ONE dataset, save models + test_data.csv + metrics table.

Edit the CONFIG block for your dataset, then run:  python model/train_models.py
"""
import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# ---------------- CONFIG: change these for YOUR dataset ----------------
DATA_PATH = "heart.csv"   # raw CSV downloaded from Kaggle / UCI
TARGET = "HeartDisease"                     # name of the label column
DROP_COLS = []                        # id columns etc. to drop
TEST_SIZE = 0.2
SEED = 42
# -----------------------------------------------------------------------

# Fallback to current directory if running in an interactive Jupyter/Colab environment
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
ROOT = os.path.dirname(HERE) if "/model" in HERE else HERE


def load_and_clean(path, target, drop_cols):
    df = pd.read_csv(path).drop(columns=drop_cols, errors="ignore")
    df = df.dropna()
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)  # encode categoricals
    y = df[target]
    if y.dtype == object:
        y = pd.Series(LabelEncoder().fit_transform(y), name=target)
    return X, y


def build_models():
    # scaling matters for LR, kNN, SVM; harmless for the others
    return { 
        "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]),
        "Decision Tree": Pipeline([("sc", StandardScaler()), ("m", DecisionTreeClassifier(max_depth=6, random_state=SEED))]),
        "kNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=7))]),
        "Naive Bayes": Pipeline([("sc", StandardScaler()), ("m", GaussianNB())]),
        "Support Vector Machine": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=SEED))]),
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


if __name__ == "__main__":
    X, y = load_and_clean(os.path.join(ROOT, DATA_PATH), TARGET, DROP_COLS)
    print(f"Rows: {X.shape[0]}, Features: {X.shape[1]}, Classes: {y.nunique()}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # test_data.csv is what you upload in the Streamlit app
    pd.concat([X_te, y_te], axis=1).to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
    joblib.dump(list(X.columns), os.path.join(HERE, "feature_columns.pkl"))

    rows = {}
    for name, mdl in build_models().items():
        mdl.fit(X_tr, y_tr)
        joblib.dump(mdl, os.path.join(HERE, name.lower().replace(" ", "_") + ".pkl"))
        rows[name] = compute_metrics(mdl, X_te, y_te)

    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "ML Model Name"
    print(table)
    table.to_csv(os.path.join(ROOT, "metrics_table.csv"))
    print(table.to_markdown())  # paste into README (needs: pip install tabulate)