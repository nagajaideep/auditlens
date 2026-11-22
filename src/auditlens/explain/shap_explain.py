"""SHAP explanations for the fraud model: global importance + per-transaction reasons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_TEXT = {
    "amount": "transaction amount", "log_amount": "transaction amount",
    "oldbalanceOrg": "sender's balance before", "newbalanceOrig": "sender's balance after",
    "oldbalanceDest": "receiver's balance before", "newbalanceDest": "receiver's balance after",
    "err_balance_orig": "sender balance doesn't reconcile", "err_balance_dest": "receiver balance doesn't reconcile",
    "orig_emptied": "sender account emptied", "dest_zero_balances": "receiver shows zero balances",
    "amount_to_balance": "amount vs. sender balance", "hour": "hour of day", "is_night": "night-time transaction",
    "dest_is_merchant": "receiver is a merchant", "orig_tx_count": "sender's prior activity",
    "dest_tx_count": "receiver's prior activity", "type_CASH_OUT": "type is CASH_OUT",
    "type_TRANSFER": "type is TRANSFER", "type_PAYMENT": "type is PAYMENT", "type_CASH_IN": "type is CASH_IN",
    "type_DEBIT": "type is DEBIT",
}


def _unwrap(model):
    return model.named_steps["clf"] if hasattr(model, "named_steps") else model


class Explainer:
    def __init__(self, model, background: pd.DataFrame | None = None):
        import shap
        self.model = _unwrap(model)
        self.is_tree = hasattr(self.model, "get_booster") or hasattr(self.model, "estimators_")
        if self.is_tree:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            bg = background.sample(min(200, len(background)), random_state=0)
            self.explainer = shap.Explainer(model.predict_proba, bg)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        sv = self.explainer.shap_values(X) if self.is_tree else self.explainer(X).values
        if isinstance(sv, list):  # older RF API: one array per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[..., 1]
        return sv

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        sv = self.shap_values(X)
        return pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)

    def reasons(self, X: pd.DataFrame, top_k: int = 4) -> list[list[dict]]:
        """Top contributing features per row, signed (positive pushes towards fraud)."""
        sv = self.shap_values(X)
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(sv[i]))[:top_k]
            out.append([{"feature": X.columns[j], "description": FEATURE_TEXT.get(X.columns[j], X.columns[j]),
                         "value": float(X.iloc[i, j]), "shap": float(sv[i, j]),
                         "direction": "raises risk" if sv[i, j] > 0 else "lowers risk"} for j in order])
        return out

    def summary_plot(self, X: pd.DataFrame, out: str | Path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        shap.summary_plot(self.shap_values(X), X, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
        return Path(out)
