from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

folder = Path(__file__).parent
data_file = folder / "dataset" / "creditcard.csv"
model_file = folder / "model.joblib"

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #f3f7fc;
        color: #17243b;
    }
    [data-testid="stHeader"] {
        background: rgba(243, 247, 252, 0.9);
    }
    [data-testid="stMain"] .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #132544 0%, #102035 100%);
    }
    [data-testid="stSidebar"] * {
        color: #edf5ff;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b7c8df;
    }
    [data-testid="stSidebar"] input[role="combobox"] {
        color: #17243b !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 9px;
        padding: 0.45rem 0.55rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] input[type="radio"] {
        accent-color: #2dd4bf;
    }
    h1, h2, h3 {
        color: #172b4d;
        letter-spacing: -0.025em;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce7f3;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 5px 18px rgba(27, 55, 90, 0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #60728b;
    }
    [data-testid="stMetricValue"] {
        color: #172b4d;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dce7f3;
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 5px 18px rgba(27, 55, 90, 0.05);
    }
    div.stButton > button,
    div.stFormSubmitButton > button,
    div.stDownloadButton > button {
        background: #087f8c;
        border: 1px solid #087f8c;
        border-radius: 9px;
        color: #ffffff;
        font-weight: 600;
        transition: background-color 150ms ease, border-color 150ms ease;
    }
    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #066874;
        border-color: #066874;
        color: #ffffff;
    }
    [data-testid="stFileUploader"] section {
        background: #f8fbff;
        border: 1px dashed #9cb6cf;
        border-radius: 12px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #dce7f3;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not data_file.exists() or not model_file.exists():
    st.error("Run `python train.py` first.")
    st.stop()

data = pd.read_csv(data_file)
saved = joblib.load(model_file)
model = saved["model"]
features = saved["features"]

page = st.sidebar.radio(
    "Navigate",
    ["Home / Dashboard", "Model Training", "Batch Prediction", "Single Prediction"],
)

if page == "Home / Dashboard":
    st.title("Credit Card Fraud Detection")
    counts = data["Class"].value_counts()
    columns = st.columns(3)
    columns[0].metric("Total Transactions", len(data))
    columns[1].metric("Genuine Transactions", counts[0])
    columns[2].metric("Fraudulent Transactions", counts[1])
    with st.container(border=True):
        st.subheader("Transaction mix")
        st.bar_chart(
            pd.Series({"Genuine": counts[0], "Fraudulent": counts[1]}),
            color="#0b9a9a",
            height=190,
        )

elif page == "Model Training":
    st.header("Model Training")
    with st.container(border=True):
        st.subheader("Dataset preview")
        st.caption(f"{len(data):,} rows · {len(data.columns)} columns")
        st.dataframe(data.head(), use_container_width=True)
        st.write(data["Class"].value_counts().rename({0: "Genuine", 1: "Fraud"}))

    st.subheader("Model performance")
    for column, (name, score) in zip(st.columns(4), saved["scores"].items()):
        column.metric(name, f"{score:.3f}")
    with st.container(border=True):
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(
            saved["matrix"],
            index=["Actual genuine", "Actual fraud"],
            columns=["Predicted genuine", "Predicted fraud"],
        ), use_container_width=True)

elif page == "Single Prediction":
    st.header("Single Prediction")
    st.caption("Enter the transaction feature values to classify one transaction.")
    with st.form("transaction"):
        columns = st.columns(3)
        values = {}
        for index, name in enumerate(features):
            values[name] = columns[index % 3].number_input(name, value=0.0)
        predict = st.form_submit_button("Predict")

    if predict:
        transaction = pd.DataFrame([values], columns=features)
        result = model.predict(transaction)[0]
        if result == 1:
            st.error("Potentially fraudulent transaction")
        else:
            st.success("Genuine transaction")

else:
    st.header("Batch Prediction")
    st.caption("Upload a CSV containing the transaction feature columns to classify multiple transactions.")
    file = st.file_uploader("Upload a CSV file", type="csv")
    if file is not None:
        try:
            rows = pd.read_csv(file)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            st.error(f"Could not read CSV: {error}")
            st.stop()

        missing = [name for name in features if name not in rows.columns]
        if rows.empty or missing:
            st.error("CSV is empty or missing required feature columns.")
            st.stop()

        transactions = rows[features].apply(pd.to_numeric, errors="coerce")
        if transactions.isna().any().any() or not np.isfinite(transactions.to_numpy()).all():
            st.error("CSV values must be finite numbers.")
            st.stop()

        predictions = model.predict(transactions)
        rows["Predicted_Class"] = predictions
        rows["Prediction"] = np.where(
            predictions == 1, "Potentially fraudulent", "Genuine"
        )
        st.dataframe(rows)
        st.download_button(
            "Download results",
            rows.to_csv(index=False).encode("utf-8"),
            "fraud_predictions.csv",
            "text/csv",
        )
