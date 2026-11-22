"""Save / load the trained model bundle used by the API and the batch job."""
from __future__ import annotations

from pathlib import Path

import joblib

from auditlens.config import get_config, resolve


def model_dir() -> Path:
    d = resolve(get_config()["model"]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_bundle(bundle: dict, name: str = "bundle.joblib") -> Path:
    p = model_dir() / name
    joblib.dump(bundle, p)
    return p


def load_bundle(name: str = "bundle.joblib") -> dict:
    p = model_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `auditlens train` first")
    return joblib.load(p)
