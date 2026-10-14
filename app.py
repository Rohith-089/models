import io
import pandas as pd

import joblib
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import ann
import logistic_regression
import random_forest
from style import inject_css, step
from utils import build_pipeline, load_table, show_results

# Registry: add a new model by creating a module and listing it here
MODELS = {m.NAME: m for m in (ann, random_forest, logistic_regression)}

st.set_page_config(page_title="Model Trainer", layout="wide")
inject_css()

for key in ("pipe", "le", "target", "features", "train_eval", "split_eval", "model_name"):
    st.session_state.setdefault(key, None)

st.title("Model Trainer")
st.caption("Train a classifier on your data and see how well it performs.")



def reset_model_state():
    # Clear the previously trained model and its results
    for key in (
        "pipe",
        "le",
        "target",
        "features",
        "train_eval",
        "split_eval",
        "model_name",
    ):
        st.session_state[key] = None

    # Create a fresh test-upload widget
    st.session_state.test_upload_version += 1


st.session_state.setdefault("test_upload_version", 0)


# ---------- Step 1: model ----------
step(1, "Choose a model")

model_name = st.radio(
    "Model",
    list(MODELS.keys()),
    horizontal=True,
    label_visibility="collapsed",
    key="model_selector",
    on_change=reset_model_state,
)

module = MODELS[model_name]
st.caption(module.DESCRIPTION)

with st.expander("Hyperparameters"):
    params = module.get_params()

# ---------- Step 2: training file ----------
step(2, "Upload training data", "CSV or Excel. Pick the column you want to predict.")
train_file = st.file_uploader(
    "Training file", type=["csv", "xlsx"], key="train_file", label_visibility="collapsed"
)


if train_file is not None:
    df = load_table(train_file)

    # Show original dataset information
    st.caption(f"{df.shape[0]:,} rows, {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    # Remove unnecessary columns
    remove_columns = st.multiselect(
        "Remove columns (optional)",
        options=df.columns.tolist(),
        default=[],
        help="Select columns you don't need, such as Date or Time."
    )

    # Apply column removal
    df = df.drop(columns=remove_columns)

    if df.empty or len(df.columns) < 2:
        st.warning("Keep at least two columns: features and a target column.")
    else:
        c1, c2 = st.columns(2)

        target = c1.selectbox(
            "Target column",
            df.columns,
            index=len(df.columns) - 1
        )

        test_mode = c2.radio(
            "Test data",
            ["Upload separately", "Hold out from this file"],
            horizontal=True,
        )

        test_size = 0.2

        if test_mode == "Hold out from this file":
            test_size = st.slider(
                "Test split", 0.1, 0.5, 0.2, 0.05
            )

        if st.button("Train model", type="primary"):
            data = df.dropna(subset=[target]).copy()
            X = data.drop(columns=[target])

            le = LabelEncoder()

            if pd.api.types.is_numeric_dtype(data[target]):
                 y = le.fit_transform(data[target].astype(int).astype(str))
            else:
                y = le.fit_transform(data[target].astype(str))



           

      

            with st.spinner("Training..."):
                pipe = build_pipeline(X, module.build_model(params))

                if test_mode == "Hold out from this file":
                    strat = (
                        y if np.min(np.bincount(y)) >= 2 else None
                    )

                    Xtr, Xte, ytr, yte = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=42,
                        stratify=strat
                    )

                    pipe.fit(Xtr, ytr)
                    st.session_state.split_eval = (
                        yte, pipe.predict(Xte)
                    )
                else:
                    Xtr, ytr = X, y
                    pipe.fit(Xtr, ytr)
                    st.session_state.split_eval = None

                st.session_state.train_eval = (
                    ytr, pipe.predict(Xtr)
                )

            st.session_state.pipe = pipe
            st.session_state.le = le
            st.session_state.target = target
            st.session_state.features = X.columns.tolist()
            st.session_state.model_name = model_name

            st.success(f"{model_name} trained.")


# ---------- Step 3: results ----------
test_file = None
if st.session_state.pipe is not None:
    classes = st.session_state.le.classes_
    step(
        3,
        "Results",
        f"{st.session_state.model_name} predicting {st.session_state.target}",
    )

    tabs = ["Training"]
    if st.session_state.split_eval is not None:
        tabs.append("Hold-out test")
    tabs.append("Test file")
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        yt, yp = st.session_state.train_eval
        show_results("Training performance", yt, yp, classes)

    idx = 1
    if st.session_state.split_eval is not None:
        with tab_objs[idx]:
            yt, yp = st.session_state.split_eval
            show_results("Hold-out test performance", yt, yp, classes)
        idx += 1

    
    # ---------- Test file ----------
    with tab_objs[idx]:
        version = st.session_state.test_upload_version

        test_file = st.file_uploader(
            "Test file (CSV or Excel)",
            type=["csv", "xlsx"],
            key=f"test_file_{version}",
        )

        if test_file is not None:
            if st.button(
                "Test model",
                type="primary",
                key=f"test_button_{version}",
            ):
                tdf = load_table(test_file)
                tgt = st.session_state.target

                # Check required columns
                required = st.session_state.features + [tgt]
                missing = [
                    c for c in required
                    if c not in tdf.columns
                ]

                if missing:
                    st.error(
                        f"Test file is missing columns: {missing}"
                    )

                else:
                    # Remove rows with missing labels
                    tdf = tdf.dropna(subset=[tgt]).copy()

                    # Normalize test labels
                    le = st.session_state.le
                    classes = le.classes_.astype(str)

                    labels = tdf[tgt].astype(str).str.strip()

                    # Match text labels or numeric class IDs
                    if labels.isin(classes).all():
                        normalized = labels

                    elif labels.str.fullmatch(r"\d+").all():
                        numeric_map = {
                            str(i): label
                            for i, label in enumerate(classes)
                        }
                        normalized = labels.map(numeric_map)

                    else:
                        normalized = labels

                    # Keep known labels
                    known = normalized.isin(classes)

                    if (~known).any():
                        st.warning(
                            f"Dropped {(~known).sum()} rows "
                            "with labels not seen in training."
                        )

                    tdf = tdf.loc[known].copy()
                    normalized = normalized.loc[known]

                    if tdf.empty:
                        st.error(
                            "No matching test labels found. "
                            "Check your training and test files."
                        )

                    else:
                        # Actual labels and predictions
                        yt = le.transform(normalized)

                        yp = st.session_state.pipe.predict(
                            tdf[st.session_state.features]
                        )

                        show_results(
                            "Test file performance",
                            yt,
                            yp,
                            classes,
                        )





    st.write("")
    buf = io.BytesIO()
    joblib.dump(
        {
            "pipeline": st.session_state.pipe,
            "label_encoder": st.session_state.le,
            "features": st.session_state.features,
        },
        buf,
    )
    st.download_button("Download trained model", buf.getvalue(), "model.joblib")