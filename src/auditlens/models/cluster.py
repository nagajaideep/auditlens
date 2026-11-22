"""K-Means vendor segmentation. A 'vendor' is a receiving account (nameDest);
each gets a behavioural profile, and clusters are labelled in plain English."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

VENDOR_FEATURES = ["log_n_tx", "log_mean_amount", "cv_amount", "log_max_amount", "share_transfer",
                   "share_cash_out", "share_night", "n_unique_payers_ratio", "active_days"]


def vendor_profiles(df: pd.DataFrame, min_tx: int = 3) -> pd.DataFrame:
    d = df[df["nameDest"].str.startswith("C")].copy()
    d["night"] = ((d["step"] - 1) % 24) < 6
    d["day"] = (d["step"] - 1) // 24
    g = d.groupby("nameDest")
    prof = pd.DataFrame({
        "n_tx": g.size(),
        "mean_amount": g["amount"].mean(),
        "std_amount": g["amount"].std().fillna(0),
        "max_amount": g["amount"].max(),
        "share_transfer": g["type"].apply(lambda s: (s == "TRANSFER").mean()),
        "share_cash_out": g["type"].apply(lambda s: (s == "CASH_OUT").mean()),
        "share_night": g["night"].mean(),
        "n_unique_payers": g["nameOrig"].nunique(),
        "active_days": g["day"].nunique(),
        "fraud_in": g["isFraud"].sum() if "isFraud" in d else 0,
    })
    prof = prof[prof["n_tx"] >= min_tx]
    prof["log_n_tx"] = np.log1p(prof["n_tx"])
    prof["log_mean_amount"] = np.log1p(prof["mean_amount"])
    prof["log_max_amount"] = np.log1p(prof["max_amount"])
    prof["cv_amount"] = prof["std_amount"] / (prof["mean_amount"] + 1)
    prof["n_unique_payers_ratio"] = prof["n_unique_payers"] / prof["n_tx"]
    return prof


class VendorSegmenter:
    def __init__(self, k: int | None = None, k_range=range(3, 9), seed: int = 42):
        self.k, self.k_range, self.seed = k, k_range, seed
        self.scaler = StandardScaler()
        self.km: KMeans | None = None
        self.silhouettes: dict[int, float] = {}
        self.labels_: dict[int, str] = {}

    def fit(self, prof: pd.DataFrame) -> VendorSegmenter:
        Z = self.scaler.fit_transform(prof[VENDOR_FEATURES])
        sample = Z if len(Z) <= 5000 else Z[np.random.default_rng(self.seed).choice(len(Z), 5000, replace=False)]
        if self.k is None:
            for k in self.k_range:
                if k >= len(sample):
                    break
                lab = KMeans(k, n_init=10, random_state=self.seed).fit_predict(sample)
                self.silhouettes[k] = float(silhouette_score(sample, lab))
            self.k = max(self.silhouettes, key=self.silhouettes.get)
        self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(Z)
        self._name_clusters(prof, self.km.labels_)
        return self

    def predict(self, prof: pd.DataFrame) -> np.ndarray:
        return self.km.predict(self.scaler.transform(prof[VENDOR_FEATURES]))

    def _name_clusters(self, prof: pd.DataFrame, labels: np.ndarray) -> None:
        overall = prof[VENDOR_FEATURES].mean()
        spread = prof[VENDOR_FEATURES].std().replace(0, 1)
        nice = {"log_n_tx": "high-volume", "log_mean_amount": "large-ticket", "cv_amount": "erratic-amounts",
                "log_max_amount": "big-spikes", "share_transfer": "transfer-heavy", "share_cash_out": "cash-out-heavy",
                "share_night": "night-active", "n_unique_payers_ratio": "many-payers", "active_days": "always-active"}
        low = {"log_n_tx": "low-volume", "log_mean_amount": "small-ticket", "cv_amount": "steady-amounts",
               "log_max_amount": "no-spikes", "share_transfer": "few-transfers", "share_cash_out": "few-cash-outs",
               "share_night": "daytime-only", "n_unique_payers_ratio": "repeat-payers", "active_days": "sporadic"}
        for c in np.unique(labels):
            z = (prof.loc[labels == c, VENDOR_FEATURES].mean() - overall) / spread
            top = z.abs().sort_values(ascending=False).index[:2]
            parts = [(nice[f] if z[f] > 0 else low[f]) for f in top]
            self.labels_[int(c)] = " / ".join(parts)

    def summary(self, prof: pd.DataFrame) -> pd.DataFrame:
        p = prof.assign(cluster=self.km.labels_)
        agg = p.groupby("cluster").agg(vendors=("n_tx", "size"), avg_tx=("n_tx", "mean"),
                                       avg_amount=("mean_amount", "mean"), night_share=("share_night", "mean"),
                                       fraud_rate=("fraud_in", lambda s: (s > 0).mean()))
        agg["label"] = [self.labels_[int(c)] for c in agg.index]
        return agg
