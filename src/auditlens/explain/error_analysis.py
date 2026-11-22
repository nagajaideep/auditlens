"""Error analysis and slice (fairness) analysis for the fraud classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def outcome_labels(y_true, pred) -> np.ndarray:
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    return np.select([(y_true == 1) & (pred == 1), (y_true == 0) & (pred == 1), (y_true == 1) & (pred == 0)],
                     ["TP", "FP", "FN"], "TN")


def compare_outcomes(X: pd.DataFrame, y_true, pred, cols: list[str] | None = None) -> pd.DataFrame:
    """Median feature values for TP / FP / FN / TN — shows what false positives look like."""
    cols = cols or ["amount", "oldbalanceOrg", "newbalanceOrig", "err_balance_orig", "orig_emptied",
                    "dest_zero_balances", "is_night", "type_TRANSFER", "type_CASH_OUT"]
    lab = outcome_labels(y_true, pred)
    t = X[cols].assign(outcome=lab).groupby("outcome").median().T
    t.loc["count"] = pd.Series(lab).value_counts()
    return t[[c for c in ["TP", "FP", "FN", "TN"] if c in t.columns]]


def slice_metrics(segments: pd.DataFrame, y_true, proba, threshold: float) -> pd.DataFrame:
    """Per-segment recall, precision, false-positive rate and flag rate."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    rows = []
    for col in segments.columns:
        for val, idx in segments.groupby(col).indices.items():
            yt, yp = y_true[idx], pred[idx]
            neg = (yt == 0).sum()
            rows.append({
                "segment": col, "value": val, "n": len(idx), "fraud": int(yt.sum()),
                "flag_rate": float(yp.mean()),
                "recall": recall_score(yt, yp, zero_division=0) if yt.sum() else np.nan,
                "precision": precision_score(yt, yp, zero_division=0) if yp.sum() else np.nan,
                "fpr": float(((yp == 1) & (yt == 0)).sum() / neg) if neg else np.nan,
            })
    return pd.DataFrame(rows)


def disparity(slices: pd.DataFrame, metric: str = "fpr", min_n: int = 200) -> pd.DataFrame:
    """Max/min ratio of a metric across values of each segment (flags uneven treatment)."""
    s = slices[(slices["n"] >= min_n) & slices[metric].notna()]
    g = s.groupby("segment")[metric]
    out = pd.DataFrame({"min": g.min(), "max": g.max()})
    out["ratio"] = out["max"] / out["min"].replace(0, np.nan)
    return out
