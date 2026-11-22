"""Append-only JSON-lines request log, read by the Streamlit monitoring dashboard."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()


class RequestLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record) -> None:
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()))
        line = json.dumps(record, default=str)
        with _lock, self.path.open("a") as fh:
            fh.write(line + "\n")


def read_log(path: str | Path):
    import pandas as pd
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ts", "path", "status", "latency_ms"])
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"])
    return df
