"""Load transactions from CSV or PostgreSQL, and bulk-load CSV into PostgreSQL."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from auditlens.config import get_config, resolve
from auditlens.logging_utils import get_logger, log_event

log = get_logger(__name__)


def read_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def engine(url: str | None = None):
    from sqlalchemy import create_engine
    return create_engine(url or get_config()["data"]["postgres_url"])


COPY_COLS = "step, type, amount, nameorig, oldbalanceorg, newbalanceorig, namedest, oldbalancedest, " \
            "newbalancedest, isfraud, isflaggedfraud"


def csv_to_postgres(csv_path: str | Path, url: str | None = None) -> int:
    """Bulk-load a PaySim CSV into `transactions_raw` (see sql/schema.sql) with PostgreSQL COPY.
    6.3M rows load in about a minute, versus tens of minutes with row-by-row inserts."""
    from sqlalchemy import text
    eng = engine(url)
    schema_sql = resolve("sql/schema.sql").read_text()
    with eng.begin() as conn:
        for stmt in [s for s in schema_sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
        conn.execute(text("TRUNCATE transactions_raw RESTART IDENTITY"))
    raw = eng.raw_connection()
    try:
        with raw.cursor() as cur, open(csv_path) as fh:
            cur.copy_expert(f"COPY transactions_raw ({COPY_COLS}) FROM STDIN WITH (FORMAT csv, HEADER true)", fh)
            cur.execute("ANALYZE transactions_raw")
            cur.execute("SELECT COUNT(*) FROM transactions_raw")
            total = cur.fetchone()[0]
        raw.commit()
    finally:
        raw.close()
    log_event(log, "loaded into postgres", rows=total)
    return total


def read_postgres(url: str | None = None, limit: int | None = None) -> pd.DataFrame:
    q = "SELECT * FROM transactions_raw ORDER BY step"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql(q, engine(url))
    back = {"nameorig": "nameOrig", "oldbalanceorg": "oldbalanceOrg", "newbalanceorig": "newbalanceOrig",
            "namedest": "nameDest", "oldbalancedest": "oldbalanceDest", "newbalancedest": "newbalanceDest",
            "isfraud": "isFraud", "isflaggedfraud": "isFlaggedFraud"}
    return df.rename(columns=back).drop(columns=["id"], errors="ignore")


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or get_config()
    d = cfg["data"]
    if d["source"] == "postgres":
        return read_postgres(d["postgres_url"], d.get("sample_rows"))
    return read_csv(resolve(d["raw_csv"]), d.get("sample_rows"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a PaySim CSV into PostgreSQL")
    ap.add_argument("--csv", default=str(resolve(get_config()["data"]["raw_csv"])))
    ap.add_argument("--url", default=None)
    a = ap.parse_args()
    n = csv_to_postgres(a.csv, a.url)
    print(f"loaded {n:,} rows into transactions_raw")


if __name__ == "__main__":
    main()
