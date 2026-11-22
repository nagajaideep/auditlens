"""Cleaning: dedupe, fix types, impute balances, derive time columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

BAL_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, log) where log records what each step changed."""
    log: dict = {"rows_in": len(df)}
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = int(before - len(df))

    df["type"] = df["type"].astype(str).str.strip().str.upper()
    for c in ["amount", *BAL_COLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    neg = (df[["amount", *BAL_COLS]] < 0).sum().to_dict()
    log["negative_values_clipped"] = {k: int(v) for k, v in neg.items() if v}
    df[["amount", *BAL_COLS]] = df[["amount", *BAL_COLS]].clip(lower=0)

    nulls = df[BAL_COLS].isna().sum().to_dict()
    log["nulls_imputed"] = {k: int(v) for k, v in nulls.items() if v}
    # impute missing balances from the accounting identity where possible, else 0
    out = df["type"].isin(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
    m = df["oldbalanceDest"].isna() & df["newbalanceDest"].notna()
    df.loc[m & out, "oldbalanceDest"] = (df["newbalanceDest"] - df["amount"]).clip(lower=0)
    m = df["newbalanceDest"].isna() & df["oldbalanceDest"].notna()
    df.loc[m & out, "newbalanceDest"] = df["oldbalanceDest"] + df["amount"]
    df[BAL_COLS] = df[BAL_COLS].fillna(0.0)

    bad_amount = df["amount"].isna()
    log["rows_dropped_missing_amount"] = int(bad_amount.sum())
    df = df[~bad_amount]

    df["step"] = df["step"].astype(int)
    df["isFraud"] = df["isFraud"].astype(int)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(int)
    df["day"] = ((df["step"] - 1) // 24).astype(int)
    df["hour"] = ((df["step"] - 1) % 24).astype(int)
    df = df.reset_index(drop=True)
    df.insert(0, "tx_id", np.arange(len(df), dtype=np.int64))
    log["rows_out"] = len(df)
    return df, log
