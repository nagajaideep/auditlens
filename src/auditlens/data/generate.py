"""Synthetic PaySim-schema generator.

The real PaySim data (Kaggle: ealaxi/paysim1, ~6.3M rows) needs a Kaggle login,
so this generator produces data with the *same columns and semantics* for CI,
tests and quick demos. Fraud follows PaySim's pattern: a TRANSFER that empties the
origin account, usually followed by a CASH_OUT. Never report metrics from this
data as real results -- train on the real CSV for that.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
TYPE_P = [0.34, 0.08, 0.35, 0.22, 0.01]
TYPE_SCALE = {"PAYMENT": 1.3e4, "TRANSFER": 9e5, "CASH_OUT": 1.8e5, "CASH_IN": 1.7e5, "DEBIT": 5.5e3}


def generate(n_rows: int = 50_000, n_steps: int = 744, fraud_rate: float = 0.004,
             n_customers: int | None = None, n_dest: int | None = None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_customers = n_customers or max(1000, n_rows // 2)
    n_dest = n_dest or max(300, n_rows // 20)

    types = rng.choice(TYPES, size=n_rows, p=TYPE_P)
    # daily cycle + weekly seasonality on step (hours)
    hours = np.arange(n_steps)
    w = 1 + 0.6 * np.sin(2 * np.pi * (hours % 24 - 6) / 24) + 0.25 * np.sin(2 * np.pi * hours / 168)
    w = np.clip(w, 0.05, None)
    step = np.sort(rng.choice(hours, size=n_rows, p=w / w.sum())) + 1

    scale = np.array([TYPE_SCALE[t] for t in types])
    amount = np.round(rng.lognormal(np.log(scale), 0.9), 2)
    old_org = np.round(np.where(rng.random(n_rows) < 0.3, 0.0, rng.lognormal(11, 1.6, n_rows)), 2)
    is_out = np.isin(types, ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
    new_org = np.where(is_out, np.maximum(old_org - amount, 0.0), old_org + amount)

    dest_idx = rng.integers(0, n_dest, n_rows)
    name_dest = np.where(types == "PAYMENT",
                         np.char.add("M", rng.integers(1e8, 2e9, n_rows).astype(str)),
                         np.char.add("C", (1_000_000 + dest_idx).astype(str)))
    old_dest = np.where(types == "PAYMENT", 0.0, np.round(rng.lognormal(12, 1.8, n_rows), 2))
    new_dest = np.where(types == "PAYMENT", 0.0, old_dest + np.where(is_out, amount, -np.minimum(amount, old_dest)))

    df = pd.DataFrame({
        "step": step, "type": types, "amount": amount,
        "nameOrig": np.char.add("C", rng.integers(0, n_customers, n_rows).astype(str)),
        "oldbalanceOrg": old_org, "newbalanceOrig": np.round(new_org, 2),
        "nameDest": name_dest, "oldbalanceDest": old_dest, "newbalanceDest": np.round(new_dest, 2),
        "isFraud": 0, "isFlaggedFraud": 0,
    })

    # inject fraud into TRANSFER / CASH_OUT rows
    cand = np.flatnonzero(np.isin(types, ["TRANSFER", "CASH_OUT"]))
    n_fraud = min(len(cand), max(10, int(fraud_rate * n_rows)))
    fidx = rng.choice(cand, size=n_fraud, replace=False)
    f_amount = np.round(rng.lognormal(np.log(2.5e5), 1.2, n_fraud), 2)
    df.loc[fidx, "amount"] = f_amount
    # ~65% of frauds empty the account exactly; the rest leave a small residual balance
    empties = rng.random(n_fraud) < 0.65
    residual = np.where(empties, 0.0, np.round(f_amount * rng.uniform(0.01, 0.6, n_fraud), 2))
    df.loc[fidx, "oldbalanceOrg"] = f_amount + residual
    df.loc[fidx, "newbalanceOrig"] = residual
    # legitimate look-alikes: some normal transfers also move the exact full balance
    normal_cand = np.setdiff1d(cand, fidx)
    look = rng.choice(normal_cand, size=min(len(normal_cand), n_fraud * 6), replace=False)
    df.loc[look, "oldbalanceOrg"] = df.loc[look, "amount"]
    df.loc[look, "newbalanceOrig"] = 0.0
    # destination balances often not updated for fraud in PaySim
    zero_dest = rng.random(n_fraud) < 0.4
    df.loc[fidx[zero_dest], ["oldbalanceDest", "newbalanceDest"]] = 0.0
    night = rng.random(n_fraud) < 0.4
    df.loc[fidx[night], "step"] = (df.loc[fidx[night], "step"] // 24) * 24 + rng.integers(1, 6, night.sum())
    df["step"] = df["step"].clip(1, n_steps)
    df.loc[fidx, "isFraud"] = 1
    df.loc[fidx[df.loc[fidx, "amount"].to_numpy() > 2e5], "isFlaggedFraud"] = 0
    big = df.index.isin(fidx) & (df["type"] == "TRANSFER") & (df["amount"] > 1e7)
    df.loc[big, "isFlaggedFraud"] = 1

    # a little realistic dirt for the cleaning step
    dirty = rng.choice(n_rows, size=max(1, n_rows // 2000), replace=False)
    df.loc[dirty[: len(dirty) // 2], "oldbalanceDest"] = np.nan
    df = pd.concat([df, df.iloc[: max(1, n_rows // 5000)]], ignore_index=True)  # duplicates
    return df.sort_values("step").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic PaySim-schema transactions")
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=744)
    ap.add_argument("--out", default="data/raw/paysim.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.rows, a.steps, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} rows ({df.isFraud.mean():.3%} fraud) -> {a.out}")


if __name__ == "__main__":
    main()
