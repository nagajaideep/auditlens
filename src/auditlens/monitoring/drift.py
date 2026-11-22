"""Data / prediction drift. A dependency-free PSI + KS check always runs (results go to
JSON for the dashboard and CI); an Evidently HTML report is produced when Evidently is installed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: compare category shares instead
        vals = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == v).mean() for v in vals]); c = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref); c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, psi_threshold: float = 0.2) -> pd.DataFrame:
    rows = []
    for col in [c for c in reference.columns if c in current.columns]:
        r, c = reference[col].dropna().to_numpy(), current[col].dropna().to_numpy()
        if len(r) == 0 or len(c) == 0:
            continue
        p = psi(r, c)
        ks = ks_2samp(r, c)
        rows.append({"feature": col, "psi": p, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                     "drifted": p > psi_threshold})
    return pd.DataFrame(rows)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, html_out: str | Path | None = None) -> dict:
    table = drift_table(reference, current)
    summary = {
        "n_features": len(table), "n_drifted": int(table.drifted.sum()),
        "share_drifted": float(table.drifted.mean()) if len(table) else 0.0,
        "prediction_psi": float(table.loc[table.feature == "score", "psi"].iloc[0]) if "score" in set(table.feature) else None,
        "reference_alert_rate": None, "current_rows": len(current),
        "features": table.round(4).to_dict("records"),
    }
    summary["dataset_drift"] = summary["share_drifted"] > 0.3  # policy threshold (model risk policy s.7)
    if html_out:
        html_out = Path(html_out); html_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
            snap = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
            snap.save_html(str(html_out))
            summary["evidently_html"] = str(html_out)
        except Exception as e:  # Evidently optional / API changes
            summary["evidently_error"] = str(e)[:200]
        html_out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary
