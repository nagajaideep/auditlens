"""Profiling + EDA figures + a Markdown data-quality report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from auditlens.data.schema import failure_summary


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c, "dtype": str(s.dtype), "nulls": int(s.isna().sum()),
            "null_pct": round(100 * s.isna().mean(), 3), "unique": int(s.nunique()),
            "min": s.min() if pd.api.types.is_numeric_dtype(s) else "",
            "max": s.max() if pd.api.types.is_numeric_dtype(s) else "",
        })
    return pd.DataFrame(rows)


def balance_violations(df: pd.DataFrame) -> pd.DataFrame:
    out = df["type"].isin(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
    err = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).abs() > 0.01
    return (df[out].assign(violation=err[out])
            .groupby("type")["violation"].agg(["sum", "mean"]).rename(columns={"sum": "n", "mean": "rate"}))


def make_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df["type"].value_counts().plot.bar(ax=ax[0], color="#4C72B0", title="Transactions by type")
    df.groupby("type")["isFraud"].mean().mul(100).plot.bar(ax=ax[1], color="#C44E52", title="Fraud rate by type (%)")
    fig.tight_layout(); p = out_dir / "type_mix.png"; fig.savefig(p, dpi=110); plt.close(fig); paths.append(p.name)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, g in df.groupby("isFraud"):
        ax.hist(np.log10(g["amount"] + 1), bins=60, alpha=0.6, density=True,
                label="fraud" if label else "normal")
    ax.set(xlabel="log10(amount + 1)", ylabel="density", title="Amount distribution"); ax.legend()
    fig.tight_layout(); p = out_dir / "amount_dist.png"; fig.savefig(p, dpi=110); plt.close(fig); paths.append(p.name)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    df.groupby("hour")["isFraud"].mean().mul(100).plot(ax=ax, marker="o", color="#C44E52")
    ax.set(xlabel="hour of day", ylabel="fraud %", title="Fraud rate by hour")
    fig.tight_layout(); p = out_dir / "fraud_by_hour.png"; fig.savefig(p, dpi=110); plt.close(fig); paths.append(p.name)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    df.groupby("day")["amount"].sum().plot(ax=ax)
    ax.set(xlabel="day", ylabel="total amount", title="Daily transaction value")
    fig.tight_layout(); p = out_dir / "daily_value.png"; fig.savefig(p, dpi=110); plt.close(fig); paths.append(p.name)

    num = df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(num.corr(method="spearman"), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Spearman correlation")
    fig.tight_layout(); p = out_dir / "correlation.png"; fig.savefig(p, dpi=110); plt.close(fig); paths.append(p.name)
    return paths


def write_report(raw: pd.DataFrame, clean_df: pd.DataFrame, clean_log: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    figs = make_figures(clean_df, out_dir / "figures")
    fails = failure_summary(raw)
    prof = profile(raw)
    viol = balance_violations(clean_df)
    by_type = clean_df.groupby("type").agg(n=("amount", "size"), total=("amount", "sum"),
                                           fraud=("isFraud", "sum"), fraud_rate=("isFraud", "mean"))
    flagged = clean_df
    tp = int(((flagged.isFlaggedFraud == 1) & (flagged.isFraud == 1)).sum())
    fn = int(((flagged.isFlaggedFraud == 0) & (flagged.isFraud == 1)).sum())

    md = [
        "# AuditLens data-quality report", "",
        f"- Rows in: **{clean_log['rows_in']:,}**, rows after cleaning: **{clean_log['rows_out']:,}**",
        f"- Duplicates removed: {clean_log['duplicates_removed']:,}",
        f"- Nulls imputed: {clean_log['nulls_imputed'] or 'none'}",
        f"- Negative values clipped: {clean_log['negative_values_clipped'] or 'none'}",
        f"- Fraud rate: **{clean_df.isFraud.mean():.4%}** ({int(clean_df.isFraud.sum()):,} cases) — heavy class imbalance",
        f"- PaySim's built-in `isFlaggedFraud` rule catches {tp} of {tp + fn} frauds "
        f"(recall {tp / max(tp + fn, 1):.2%}) — the rule-based baseline any model must beat.", "",
        "## Schema validation (Pandera, raw data)", "",
        "No failures." if fails.empty else fails.groupby(["column", "check"]).size()
        .rename("failures").reset_index().to_markdown(index=False), "",
        "## Column profile (raw)", "", prof.to_markdown(index=False), "",
        "## By transaction type", "", by_type.to_markdown(floatfmt=".4f"), "",
        "Fraud only occurs in TRANSFER and CASH_OUT — features and models should reflect that.", "",
        "## Balance-identity violations (oldbalanceOrg - amount != newbalanceOrig)", "",
        viol.to_markdown(floatfmt=".4f"), "",
        "These are frequent in PaySim even for normal rows, so the *error* term is used as a feature rather "
        "than a reason to drop rows.", "",
        "## Figures", "", *[f"![{f}](figures/{f})" for f in figs], "",
    ]
    out = out_dir / "data_quality_report.md"
    out.write_text("\n".join(md))
    return out
