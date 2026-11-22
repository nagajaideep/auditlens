"""Thin MLflow helpers. Tracking is optional: if MLflow is unavailable or disabled
(AUDITLENS_MLFLOW__ENABLED=false) every call is a no-op so tests stay fast."""
from __future__ import annotations

import contextlib
import os

from auditlens.config import get_config


def _enabled() -> bool:
    cfg = get_config().get("mlflow", {})
    if str(cfg.get("enabled", "true")).lower() == "false":
        return False
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def setup():
    if not _enabled():
        return None
    import mlflow
    cfg = get_config()["mlflow"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["tracking_uri"]))
    mlflow.set_experiment(cfg["experiment"])
    return mlflow


@contextlib.contextmanager
def run(name: str, tags: dict | None = None, nested: bool = False):
    mlflow = setup()
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=name, tags=tags, nested=nested) as r:
        yield r


def log_params(params: dict):
    if _enabled():
        import mlflow
        if mlflow.active_run():
            mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: dict, step: int | None = None):
    if _enabled():
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_artifact(path: str):
    if _enabled():
        import mlflow
        if mlflow.active_run():
            mlflow.log_artifact(path)


def log_and_register_sklearn(model, artifact_path: str, registered_name: str | None, X_example=None):
    """Log a model; register it under `registered_name` (returns version or None)."""
    if not _enabled():
        return None
    import mlflow
    if not mlflow.active_run():
        return None
    if hasattr(model, "get_booster"):
        info = mlflow.xgboost.log_model(model, name=artifact_path, input_example=X_example,
                                        registered_model_name=registered_name)
    else:
        # we trained this model ourselves in this run, so its tree types are trusted
        info = mlflow.sklearn.log_model(model, name=artifact_path, input_example=X_example,
                                        registered_model_name=registered_name,
                                        skops_trusted_types=["sklearn.tree._tree.Tree"])
    return getattr(info, "registered_model_version", None)
