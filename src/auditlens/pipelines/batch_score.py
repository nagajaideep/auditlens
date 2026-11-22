"""Nightly batch scoring job.

Reads new transactions (CSV/Parquet path, or PostgreSQL rows with step > last scored),
scores them, writes results to reports/batch/ (and to transaction_scores in Postgres when
--to-postgres), and runs a drift check against the training reference."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from auditlens.config import get_config, resolve
from auditlens.data.clean import clean
from auditlens.logging_utils import get_logger, log_event
from auditlens.pipelines.artifacts import load_bundle
from auditlens.pipelines.scoring import score_frame

log = get_logger("auditlens.batch")


def _read(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def run(input_path: str | None = None, to_postgres: bool = False, drift: bool = True) -> Path:
    cfg = get_config()
    bundle = load_bundle()
    t0 = time.time()
    if input_path:
        raw = _read(input_path)
    else:
        from auditlens.data.load import read_postgres
        raw = read_postgres(cfg["data"]["postgres_url"])
    df, _ = clean(raw)
    scores, X = score_frame(bundle, df)
    out = df[["tx_id", "step", "type", "amount", "nameOrig", "nameDest"]].join(scores)
    out["model_version"] = bundle["model_version"]
    out_dir = resolve("reports/batch"); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scores_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
    out.to_parquet(path, index=False)
    out[out.flagged].sort_values("fraud_score", ascending=False).to_csv(out_dir / "latest_flagged.csv", index=False)

    if to_postgres:
        from auditlens.data.load import engine
        out[["tx_id", "model_version", "fraud_score", "anomaly_score", "flagged"]].to_sql(
            "transaction_scores", engine(cfg["data"]["postgres_url"]), if_exists="append", index=False,
            method="multi", chunksize=10_000)

    if drift:
        from auditlens.monitoring.drift import drift_report
        cur = X[bundle["features"]].assign(score=scores["fraud_score"].to_numpy())
        drift_report(pd.read_parquet(resolve(cfg["monitoring"]["reference"])), cur,
                     resolve(cfg["monitoring"]["drift_report"]))
    log_event(log, "batch scored", rows=len(out), flagged=int(out.flagged.sum()),
              seconds=round(time.time() - t0, 2), output=str(path))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Nightly batch scoring")
    ap.add_argument("--input", help="CSV/Parquet of transactions (default: read from PostgreSQL)")
    ap.add_argument("--to-postgres", action="store_true")
    ap.add_argument("--no-drift", action="store_true")
    a = ap.parse_args()
    run(a.input, a.to_postgres, not a.no_drift)


if __name__ == "__main__":
    main()
