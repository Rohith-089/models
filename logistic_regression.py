import streamlit as st
from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"
DESCRIPTION = "Linear baseline for classification. Fast and interpretable."


def get_params():
    """Render hyperparameter inputs and return them as a dict."""
    c1, c2, c3 = st.columns(3)
    return {
        "C": c1.number_input("C (regularization)", 0.001, 100.0, 1.0),
        "max_iter": c2.number_input("Max iterations", 100, 5000, 1000),
        "balanced": c3.checkbox("Balance classes", value=False),
    }


def build_model(params):
    return LogisticRegression(
        C=params["C"],
        max_iter=int(params["max_iter"]),
        class_weight="balanced" if params["balanced"] else None,
    )
