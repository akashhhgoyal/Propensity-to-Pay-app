import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def extract_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # --------------------------------------------------
    # STANDARDIZE COLUMN NAMES (VERY IMPORTANT)
    # --------------------------------------------------
    df.columns = df.columns.str.strip().str.lower()

    column_aliases = {
        "loan_id": ["loan_id", "loan number", "loan_no", "loanid"],
        "emi_amount": ["emi_amount", "emi", "emi amt", "emi_amt"],
        "pos": ["pos", "outstanding", "principal_outstanding"],
        "calling_attempts": ["calling_attempts", "call_attempts", "total_calls"],
        "connected": ["connected", "connected_calls", "calls_connected"],
        "ptp_count": ["ptp_count", "ptp", "promise_to_pay"]
    }

    for final_col, aliases in column_aliases.items():
        for col in aliases:
            if col in df.columns:
                df[final_col] = df[col]
                break
        else:
            # Column not found → safe default
            df[final_col] = 0

    # --------------------------------------------------
    # BUCKET HANDLING (ALREADY FIXED, KEEP IT)
    # --------------------------------------------------
    bucket_map = {"0-30": 1, "31-60": 2, "61-90": 3, "90+": 4}

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
        df["bucket_numeric"] = 0

    # --------------------------------------------------
    # FINAL GUARANTEE (NO KEYERROR POSSIBLE)
    # --------------------------------------------------
    final_cols = [
        "loan_id",
        "emi_amount",
        "pos",
        "bucket_numeric",
        "calling_attempts",
        "connected",
        "ptp_count"
    ]

    return df[final_cols]
