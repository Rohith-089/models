import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from style import ACCENT, INK, MUTED


def load_table(file):
    """Read an uploaded CSV or Excel file (first sheet) into a DataFrame."""
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)


def build_pipeline(X, model):
    """Auto preprocessing (impute, scale, one-hot) + the chosen model."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def compute_metrics(y_true, y_pred):
    kw = {"average": "weighted", "zero_division": 0}
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, **kw),
        "Recall": recall_score(y_true, y_pred, **kw),
        "F1 score": f1_score(y_true, y_pred, **kw),
    }


def plot_cm(y_true, y_pred, labels):
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    size = min(3.3 + 0.3 * max(0, n - 4), 6)
    cmap = LinearSegmentedColormap.from_list("accent", ["#FFFFFF", ACCENT])

    fig, ax = plt.subplots(figsize=(size, size * 0.9), dpi=110)
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, color=MUTED)
    ax.set_yticklabels(labels, fontsize=8, color=MUTED)
    ax.set_xlabel("Predicted", fontsize=8.5, color=MUTED, labelpad=6)
    ax.set_ylabel("Actual", fontsize=8.5, color=MUTED, labelpad=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,
                fontweight="600", color="white" if cm[i, j] > thresh else INK,
            )
    fig.tight_layout()
    return fig


def report_table(y_true, y_pred, classes):
    """Same layout as sklearn's classification_report, as a DataFrame."""
    names = [str(c) for c in classes]
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(names))), target_names=names,
        output_dict=True, zero_division=0,
    )
    f2 = lambda v: f"{v:.2f}"
    rows = [
        (n, f2(rep[n]["precision"]), f2(rep[n]["recall"]), f2(rep[n]["f1-score"]), f"{int(rep[n]['support']):,}")
        for n in names
    ]
    rows.append(("accuracy", "", "", f2(accuracy_score(y_true, y_pred)), f"{len(y_true):,}"))
    for k in ("macro avg", "weighted avg"):
        r = rep[k]
        rows.append((k, f2(r["precision"]), f2(r["recall"]), f2(r["f1-score"]), f"{int(r['support']):,}"))
    return pd.DataFrame(
        rows, columns=["", "precision", "recall", "f1-score", "support"]
    ).set_index("")


def show_results(title, y_true, y_pred, classes):
    st.markdown(f"#### {title}")

    cols = st.columns(4)
    for col, (k, v) in zip(cols, compute_metrics(y_true, y_pred).items()):
        col.metric(k, f"{v:.2%}")
    if len(classes) > 2:
        st.caption("Precision, recall and F1 are weighted averages across classes.")

    st.write("")
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.caption("Confusion matrix")
        fig = plot_cm(y_true, y_pred, [str(c) for c in classes])
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    with right:
        st.caption("Classification report")
        st.dataframe(report_table(y_true, y_pred, classes), use_container_width=True)

    with st.expander("View report as text"):
        names = [str(c) for c in classes]
        st.code(
            classification_report(
                y_true, y_pred, labels=list(range(len(names))),
                target_names=names, zero_division=0,
            )
        )