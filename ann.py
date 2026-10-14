import streamlit as st
from sklearn.neural_network import MLPClassifier

NAME = "ANN"
DESCRIPTION = "Neural network (MLP). Good for non-linear patterns, needs scaled data."


def get_params():
    """Render hyperparameter inputs and return them as a dict."""
    c1, c2, c3 = st.columns(3)
    return {
        "layers": c1.text_input("Hidden layers (comma-separated)", "64,32"),
        "epochs": c2.number_input("Epochs", 10, 2000, 200),
        "lr": c3.number_input("Learning rate", 0.0001, 1.0, 0.001, format="%.4f"),
    }


def build_model(params):
    layers = tuple(int(x) for x in params["layers"].split(",") if x.strip())
    return MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=int(params["epochs"]),
        learning_rate_init=params["lr"],
        random_state=42,
    )
