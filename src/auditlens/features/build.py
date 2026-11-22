"""Feature engineering for transaction-level fraud models.

Features only use information available at scoring time for that transaction
(no future leakage); account-history features use expanding counts ordered by step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPE_DUMMIES = ["type_CASH_OUT", "type_TRANSFER", "type_PAYMENT", "type_CASH_IN", "type_DEBIT"]

FEATURES = [
    "amount", "log_amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "err_balance_orig", "err_balance_dest", "orig_emptied", "dest_zero_balances",
    "amount_to_balance", "hour", "is_night", "dest_is_merchant",
    "orig_tx_count", "dest_tx_count", *TYPE_DUMMIES,
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature frame (same index as df) with columns FEATURES."""
    X = pd.DataFrame(index=df.index)
    amt = df["amount"].astype(float)
    X["amount"] = amt
    X["log_amount"] = np.log1p(amt)
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        X[c] = df[c].astype(float)
    X["err_balance_orig"] = df["newbalanceOrig"] + amt - df["oldbalanceOrg"]
    X["err_balance_dest"] = df["oldbalanceDest"] + amt - df["newbalanceDest"]
    X["orig_emptied"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
    X["dest_zero_balances"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)
    X["amount_to_balance"] = amt / (df["oldbalanceOrg"] + 1.0)
    hour = df["hour"] if "hour" in df else (df["step"] - 1) % 24
    X["hour"] = hour.astype(int)
    X["is_night"] = hour.between(0, 5).astype(int)
    X["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

    order = df.sort_values("step", kind="stable").index
    s = df.loc[order]
    X.loc[order, "orig_tx_count"] = s.groupby("nameOrig").cumcount().to_numpy()
    X.loc[order, "dest_tx_count"] = s.groupby("nameDest").cumcount().to_numpy()

    for t in TYPE_DUMMIES:
        X[t] = (df["type"] == t.removeprefix("type_")).astype(int)
    return X[FEATURES].astype(float)


def segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Segments used for the fairness / slice analysis."""
    amt_band = pd.cut(df["amount"], [-1, 1e4, 1e5, 1e6, np.inf], labels=["<10k", "10k-100k", "100k-1M", ">1M"])
    bal_band = pd.cut(df["oldbalanceOrg"], [-1, 0, 1e4, 1e5, np.inf], labels=["zero", "<10k", "10k-100k", ">100k"])
    return pd.DataFrame({
        "type": df["type"].values,
        "amount_band": amt_band.astype(str).values,
        "origin_balance_band": bal_band.astype(str).values,
        "time_of_day": np.where(((df["step"] - 1) % 24) < 6, "night", "day"),
    }, index=df.index)
