from feature_engineering import extract_features
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import time
from io import BytesIO

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Propensity-To-Pay Recovery Decision Engine",
    layout="wide"
)

st.title("📞 Propensity-To-Pay Recovery Decision Engine")
st.caption("Predict whether a loan should be handled by ICC or referred out")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
LOAN_ID_COL = "loan_id"

FEATURE_COLS = [
    "emi_amount",
    "pos",
    "bucket_numeric",
    "calling_attempts",
    "connected",
    "ptp_count"
]

THRESHOLD = 0.5
DECISION_LABELS = {1: "ICC RECOVERABLE", 0: "REFER OUT"}
DECISION_COLORS = {1: "green", 0: "red"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "outputs", "tables"), exist_ok=True)

# --------------------------------------------------
# SESSION STATE (PERFORMANCE CRITICAL)
# --------------------------------------------------
if "features_ready" not in st.session_state:
    st.session_state.features_ready = False

if "df_features" not in st.session_state:
    st.session_state.df_features = None

# --------------------------------------------------
# LOAD MODEL (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "models", "final_model.pkl")
    return joblib.load(model_path)

model = load_model()

# --------------------------------------------------
# CACHE RAW FILE LOAD (🔥 FAST)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------
st.sidebar.header("📂 Upload Monthly Loan File")

uploaded_file = st.sidebar.file_uploader(
    "Upload Raw Monthly File (Excel or CSV)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👈 Upload a monthly file to begin")
    st.stop()

# --------------------------------------------------
# LOAD + FEATURE EXTRACTION (🔥 RUNS ONLY ONCE)
# --------------------------------------------------
if not st.session_state.features_ready:

    with st.spinner("⚙️ Processing raw data (one-time)…"):
        raw_df = load_raw_file(uploaded_file)

        # 🔍 DEBUG (ADD HERE)
        st.write("Raw columns:", raw_df.columns.tolist())

        df = extract_features(raw_df)

        st.write(f"⏱ Feature extraction time: {round(time.time()-start, 2)} sec")

        # Safety check
        missing = set(FEATURE_COLS + [LOAN_ID_COL]) - set(df.columns)
        if missing:
            st.error(f"Missing required columns after feature extraction: {missing}")
            st.stop()

        # Prediction
        X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["icc_recovery_probability"] = model.predict_proba(X)[:, 1]
        df["decision"] = (df["icc_recovery_probability"] >= THRESHOLD).astype(int)
        df["decision_label"] = df["decision"].map(DECISION_LABELS)

        st.session_state.df_features = df
        st.session_state.features_ready = True

else:
    df = st.session_state.df_features

st.success(f"Processed {len(df)} loans successfully")

# --------------------------------------------------
# SIDEBAR — SELECTION MODE
# --------------------------------------------------
st.sidebar.markdown("---")
selection_mode = st.sidebar.radio(
    "Selection Mode",
    ["Drilldown by Decision", "Single Loan Search", "Bulk Loan Search"]
)

# --------------------------------------------------
# VISUAL HELPERS
# --------------------------------------------------
def plot_distribution_pie(data, title):
    summary = (
        data.groupby("decision_label")
        .size()
        .reindex(["ICC RECOVERABLE", "REFER OUT"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(summary, labels=summary.index, autopct="%1.1f%%",
           colors=[DECISION_COLORS[1], DECISION_COLORS[0]])
    ax.axis("equal")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def plot_single_decision(label):
    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    ax.pie([1], colors=[DECISION_COLORS[1] if label=="ICC RECOVERABLE" else DECISION_COLORS[0]])
    ax.set_title(label)
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# DRILLDOWN MODE
# --------------------------------------------------
if selection_mode == "Drilldown by Decision":

    k1, k2 = st.columns(2)
    k1.metric("🟢 ICC Recoverable", (df["decision_label"]=="ICC RECOVERABLE").sum())
    k2.metric("🔴 Refer Out", (df["decision_label"]=="REFER OUT").sum())

    plot_distribution_pie(df, "Overall Portfolio")
    st.dataframe(df)

# --------------------------------------------------
# SINGLE LOAN SEARCH
# --------------------------------------------------
elif selection_mode == "Single Loan Search":
    loan_input = st.sidebar.text_input("Enter Loan Number")
    if loan_input:
        result = df[df[LOAN_ID_COL].astype(str).str.contains(loan_input, case=False)]
        if result.empty:
            st.error("No matching loan found")
        else:
            row = result.iloc[0]
            plot_single_decision(row["decision_label"])
            st.metric("ICC Recovery Probability", f"{row['icc_recovery_probability']*100:.1f}%")
            st.dataframe(result)

# --------------------------------------------------
# BULK SEARCH
# --------------------------------------------------
else:
    loan_input = st.sidebar.text_area("Enter Loan Numbers")
    if loan_input:
        pattern = "|".join(x.strip() for x in loan_input.replace(",", "\n").split("\n"))
        result = df[df[LOAN_ID_COL].astype(str).str.contains(pattern, case=False)]
        plot_distribution_pie(result, "Selected Loans")
        st.dataframe(result)

# --------------------------------------------------
# EXPORT (FAST + SAFE)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_excel(dataframe):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

st.sidebar.markdown("---")
if st.sidebar.button("⬇️ Download Excel"):
    st.sidebar.download_button(
        "Download",
        generate_excel(df),
        "icc_decisions.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

