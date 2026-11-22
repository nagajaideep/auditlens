"""Isolation Forest: unsupervised anomaly scores that don't depend on fraud labels.
Useful for new fraud patterns the classifier never saw."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

ANOMALY_FEATURES = ["log_amount", "err_balance_orig", "err_balance_dest", "amount_to_balance",
                    "orig_emptied", "dest_zero_balances", "is_night", "type_TRANSFER", "type_CASH_OUT"]


class AnomalyDetector:
    def __init__(self, contamination: float = 0.01, seed: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(n_estimators=300, contamination=contamination, max_samples=min(4096, 256 * 16),
                                     random_state=seed, n_jobs=-1)
        self.features = ANOMALY_FEATURES

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X[self.features].copy()
        for c in ["err_balance_orig", "err_balance_dest", "amount_to_balance"]:
            Z[c] = np.sign(Z[c]) * np.log1p(Z[c].abs())
        return Z

    def fit(self, X: pd.DataFrame) -> AnomalyDetector:
        self.model.fit(self._prep(X))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous (sign-flipped sklearn score_samples)."""
        return -self.model.score_samples(self._prep(X))

    def is_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prep(X)) == -1

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        s = self.score(X)
        flags = self.is_anomaly(X)
        return {
            "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "flag_rate": float(flags.mean()),
            "fraud_captured": float(flags[y.to_numpy() == 1].mean()) if y.sum() else float("nan"),
            "precision_of_flags": float(y.to_numpy()[flags].mean()) if flags.any() else 0.0,
        }
