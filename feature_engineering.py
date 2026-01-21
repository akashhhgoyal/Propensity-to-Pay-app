import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def extract_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -------------------------------
    # SAFE BUCKET HANDLING
    # -------------------------------
    bucket_map = {
        "0-30": 1,
        "31-60": 2,
        "61-90": 3,
        "90+": 4
    }

    if "bucket" in df.columns:
        df["bucket_numeric"] = df["bucket"].map(bucket_map)

    elif "dpd_bucket" in df.columns:
        df["bucket_numeric"] = df["dpd_bucket"].map(bucket_map)

    elif "days_past_due" in df.columns:
        df["bucket_numeric"] = pd.cut(
            df["days_past_due"],
            bins=[-1, 30, 60, 90, 10_000],
            labels=[1, 2, 3, 4]
        ).astype(float)

    else:
        # FINAL FALLBACK (prevents crash)
        df["bucket_numeric"] = 0

    # -------------------------------
    # ENSURE REQUIRED COLUMNS EXIST
    # -------------------------------
    required_defaults = {
        "emi_amount": 0,
        "pos": 0,
        "calling_attempts": 0,
        "connected": 0,
        "ptp_count": 0,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
    return df[
        [
            "loan_id",
            "emi_amount",
            "pos",
            "bucket_numeric",
            "calling_attempts",
            "connected",
            "ptp_count"
        ]
    ]
