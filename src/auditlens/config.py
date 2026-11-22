"""Config loading: YAML file + environment-variable overrides."""
from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_env_overrides(cfg: dict, environ: dict | None = None) -> dict:
    """AUDITLENS_SECTION__KEY=value overrides cfg[section][key]."""
    environ = os.environ if environ is None else environ
    out = deepcopy(cfg)
    for name, raw in environ.items():
        if not name.startswith("AUDITLENS_") or "__" not in name:
            continue
        section, key = name[len("AUDITLENS_"):].lower().split("__", 1)
        out.setdefault(section, {})[key] = _coerce(raw)
    return out


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> dict:
    path = path or os.environ.get("AUDITLENS_CONFIG", str(DEFAULT_CONFIG))
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return apply_env_overrides(cfg)


def resolve(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
