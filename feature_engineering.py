import pandas as pd

def extract_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw monthly loan data
    Returns dataframe with model-ready features
    """

    df = raw_df.copy()

    # ---- Example mappings (use your notebook logic here) ----
    df["bucket_numeric"] = df["bucket"].map({
        "0-30": 1,
        "31-60": 2,
        "61-90": 3,
        "90+": 4
    })

    df["connected"] = df["connected"].astype(int)
    df["ptp_count"] = df["ptp_count"].fillna(0)

    feature_df = df[
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

    return feature_df
