"""Shared scoring logic used by the API and the nightly batch job."""
from __future__ import annotations

import numpy as np
import pandas as pd

from auditlens.features.build import build_features


def score_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df:
        df = df.assign(hour=((df["step"] - 1) % 24).astype(int))
    X = build_features(df)
    proba = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    anom = bundle["anomaly"].score(X)
    return pd.DataFrame({
        "fraud_score": proba,
        "flagged": proba >= bundle["threshold"],
        "anomaly_score": anom,
        "anomaly_flag": bundle["anomaly"].is_anomaly(X),
    }, index=df.index), X


def risk_band(score: float, threshold: float) -> str:
    if score >= max(0.95, threshold):
        return "high"
    if score >= threshold:
        return "elevated"
    if score >= threshold / 2:
        return "watch"
    return "low"


__all__ = ["np", "risk_band", "score_frame"]
