import streamlit as st
from sklearn.ensemble import RandomForestClassifier

NAME = "Random Forest"
DESCRIPTION = "Ensemble of trees. Robust, handles mixed features, little tuning."


def get_params():
    """Render hyperparameter inputs and return them as a dict."""
    c1, c2, c3 = st.columns(3)
    return {
        "trees": c1.number_input("Trees", 10, 1000, 200),
        "depth": c2.number_input("Max depth (0 = none)", 0, 100, 0),
        "balanced": c3.checkbox("Balance classes", value=False),
    }


def build_model(params):
    return RandomForestClassifier(
        n_estimators=int(params["trees"]),
        max_depth=int(params["depth"]) or None,
        class_weight="balanced" if params["balanced"] else None,
        random_state=42,
        n_jobs=-1,
    )
